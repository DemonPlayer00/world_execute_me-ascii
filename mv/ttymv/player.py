"""The player: audio clock, terminal, and the frame pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

from . import analyze as A
from . import scenes as S
from . import tears as TEARS
from . import shots as SH
from .caps import Caps
from .canvas import Canvas, DEFAULT, mix, rgb, str_width, truncate
from .lrc import Lyrics, EmptyLyrics

# Terminal control sequences are not hard-coded here: they come from terminfo
# via ttymv.caps, because a Linux virtual console has no alternate screen and
# needs different cursor sequences than xterm.


# --------------------------------------------------------------------------
# audio clock
# --------------------------------------------------------------------------

class Clock:
    """Master time.  Backed by mpv's own playhead when available, so picture
    follows the sound rather than a wall clock that slowly disagrees with it."""

    def __init__(self):
        self.offset = 0.0
        self.duration = 0.0
        self.paused = False
        self.backend = "none"
        self._t = 0.0
        self._wall = time.monotonic()
        self.live = False

    def now(self) -> float:
        if self.paused:
            return self._t + self.offset
        return self._t + (time.monotonic() - self._wall) + self.offset

    def seek(self, t: float) -> None: ...
    def toggle_pause(self) -> None: ...
    def close(self) -> None: ...


class MpvClock(Clock):
    def __init__(self, track: str | Path, start: float = 0.0,
                 ao: str | None = None, volume: int | None = None):
        super().__init__()
        self.sock_path = tempfile.mktemp(prefix="ttymv-", suffix=".sock")
        cmd = ["mpv", "--no-video", "--no-terminal", "--really-quiet",
               "--audio-display=no", "--keep-open=no",
               f"--input-ipc-server={self.sock_path}",
               f"--start={start:.3f}"]
        if ao:
            cmd.append(f"--ao={ao}")
        if volume is not None:
            cmd.append(f"--volume={volume}")
        cmd.append(str(track))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self._rid = 0
        self._lock = threading.Lock()
        self._fp = None
        for _ in range(120):
            if os.path.exists(self.sock_path):
                try:
                    s = socket.socket(socket.AF_UNIX)
                    s.connect(self.sock_path)
                    self._fp = s.makefile("rwb")
                    break
                except OSError:
                    pass
            if self.proc.poll() is not None:
                raise RuntimeError("mpv exited before the IPC socket appeared")
            time.sleep(0.05)
        if self._fp is None:
            raise RuntimeError("could not connect to mpv IPC socket")
        self.backend = "mpv"
        self.live = True
        dur = self._cmd("get_property", "duration")
        if isinstance(dur.get("data"), (int, float)):
            self.duration = float(dur["data"])
        self._t = start
        self._wall = time.monotonic()
        self._stop = False
        threading.Thread(target=self._poll, daemon=True).start()

    def _cmd(self, *args):
        with self._lock:
            rid = self._rid
            self._rid += 1
            try:
                self._fp.write(json.dumps({"command": list(args),
                                           "request_id": rid}).encode() + b"\n")
                self._fp.flush()
            except (OSError, ValueError):
                return {}
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                line = self._fp.readline()
                if not line:
                    return {}
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("request_id") == rid:
                    return obj
            return {}

    def _poll(self):
        while not self._stop:
            t = self._cmd("get_property", "time-pos").get("data")
            p = self._cmd("get_property", "pause").get("data")
            if isinstance(t, (int, float)):
                self._t = float(t)
                self._wall = time.monotonic()
            if isinstance(p, bool):
                self.paused = p
            if self.proc.poll() is not None:
                self.live = False
                break
            time.sleep(0.2)

    def seek(self, t: float) -> None:
        t = max(0.0, t)
        self._cmd("seek", t, "absolute+exact")
        self._t = t
        self._wall = time.monotonic()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._cmd("set_property", "pause", self.paused)
        self._t = self.now() - self.offset
        self._wall = time.monotonic()

    def close(self) -> None:
        self._stop = True
        try:
            self._cmd("quit")
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            if os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
        except OSError:
            pass


class WallClock(Clock):
    """Silent / no-player fallback: monotonic time is the playhead."""

    def __init__(self, duration: float, start: float = 0.0):
        super().__init__()
        self.duration = duration
        self._t = start
        self._wall = time.monotonic()
        self.backend = "wall"

    def seek(self, t: float) -> None:
        self._t = max(0.0, t)
        self._wall = time.monotonic()


def make_clock(track, duration, start, audio: bool, ao: str | None):
    if audio and shutil.which("mpv"):
        try:
            return MpvClock(track, start, ao)
        except Exception as exc:                       # pragma: no cover
            print(f"[ttymv] mpv unavailable ({exc}); running silent",
                  file=sys.stderr)
    return WallClock(duration, start)


# --------------------------------------------------------------------------
# terminal
# --------------------------------------------------------------------------

class Term:
    """Terminal control, driven by terminfo rather than by assumption.

    A Linux virtual console has no alternate screen and needs its own cursor
    sequences, so both are taken from the capability database.  When there is
    no alternate screen the MV clears on entry and on exit instead, which is
    the best a console can do.
    """

    def __init__(self, force_size: tuple[int, int] | None = None,
                 caps: Caps | None = None, altscreen: bool = True,
                 sync_output: bool = True):
        self.caps = caps or Caps.detect()
        self.force = force_size
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.saved = None
        self.resized = True
        self.use_altscreen = bool(altscreen and self.caps.smcup)
        self.use_sync = bool(sync_output and self.caps.sync_output)
        self.bytes_written = 0

    def size(self) -> tuple[int, int]:
        if self.force:
            return self.force
        try:
            sz = os.get_terminal_size(sys.stdout.fileno())
            return max(20, sz.columns), max(6, sz.lines)
        except OSError:
            return 80, 24

    def __enter__(self):
        if self.fd is not None:
            import termios
            import tty
            self.saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        signal.signal(signal.SIGWINCH, self._on_resize)
        c = self.caps
        start = (c.smcup if self.use_altscreen else b"") + c.civis + c.clear
        self._write(start)
        return self

    def __exit__(self, *exc):
        c = self.caps
        # no alternate screen to return to: leave the console tidy instead
        tail = c.cnorm + (c.rmcup if self.use_altscreen else c.clear)
        try:
            self._write(tail)
        except Exception:
            pass
        if self.saved is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except Exception:
                pass

    def _write(self, data) -> None:
        if isinstance(data, str):
            data = data.encode()
        if not data:
            return
        os.write(sys.stdout.fileno(), data)
        self.bytes_written += len(data)

    def begin_frame(self) -> None:
        """Ask the terminal to hold the update until the frame is complete."""
        if self.use_sync:
            self._write(b"\x1b[?2026h")

    def end_frame(self, payload: str) -> None:
        data = payload.encode()
        if self.use_sync:
            data += b"\x1b[?2026l"
        try:
            os.write(sys.stdout.fileno(), data)
            self.bytes_written += len(data)
        except BlockingIOError:
            pass                      # a slow tty: drop, never wedge the loop

    def _on_resize(self, *_):
        self.resized = True

    def read_keys(self, timeout: float = 0.0) -> list[str]:
        if self.fd is None:
            return []
        keys = []
        while True:
            r, _, _ = select.select([self.fd], [], [], timeout if not keys else 0)
            if not r:
                break
            try:
                data = os.read(self.fd, 64)
            except OSError:
                break
            if not data:
                break
            keys.append(data.decode("utf-8", "ignore"))
        return keys


# --------------------------------------------------------------------------
# audio frame preparation
# --------------------------------------------------------------------------

class Audio:
    """The measured track, in *master* time.

    ``offset`` is what the file this was measured from is missing relative to
    the master (see ``MUSIC_OFFSET``).  Every timing in the piece is written
    in master time, so the analysis is read back through the same shift: a
    lookup for master time t reads the frame at t + offset.  Nothing else in
    the renderer has to know the file is not the master.
    """

    def __init__(self, data: dict, offset: float = 0.0):
        self.d = data
        self.offset = float(offset)
        self.fps = float(data["fps"])
        self.n = int(data["n"])
        # the file's own span, shifted into master time and never negative
        self.duration = max(0.1, float(data["duration"]) - self.offset)
        self.audio_end = max(0.1, float(data["audio_end"]) - self.offset)
        self.bpm = float(data["bpm"])
        self.beat_period = 60.0 / self.bpm
        self.beat_phase0 = float(data["beat_phase"])
        self.sections = list(data["sections"])
        self.section_starts = [s["start"] for s in self.sections]
        self.wave = data["wave"]
        self.wave_rate = float(data["wave_rate"])

        raw = data["bands"].astype(np.float32) / 255.0
        # Two normalisations, blended.  Per band keeps every frequency alive
        # through a loud chorus; global preserves the spectral envelope, so a
        # bass note still reads as a bass note instead of a wall of bars.
        lo_b = np.percentile(raw, 8.0, axis=0)
        hi_b = np.percentile(raw, 99.3, axis=0)
        per_band = np.clip((raw - lo_b) / np.maximum(hi_b - lo_b, 1e-3), 0.0, 1.0)
        lo_g = float(np.percentile(raw, 4.0))
        hi_g = float(np.percentile(raw, 99.6))
        glob = np.clip((raw - lo_g) / max(1e-3, hi_g - lo_g), 0.0, 1.0)
        self.bands = (0.45 * per_band + 0.55 * glob).astype(np.float32)
        self.bands **= 1.15
        self.rms = self._norm(data["rms"])
        self.flux = self._norm(data["flux"])
        self.centroid = self._norm(data["centroid"])
        self.low = self._norm(data["low"])
        self.mid = self._norm(data["mid"])
        self.high = self._norm(data["high"])

        self._bass_idx = slice(0, max(1, self.bands.shape[1] // 8))
        self.bass = self.bands[:, self._bass_idx].mean(axis=1)
        lo2 = float(np.percentile(self.bass, 5))
        hi2 = float(np.percentile(self.bass, 99.0))
        self.bass = np.clip((self.bass - lo2) / max(1e-4, hi2 - lo2), 0.0, 1.0)

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        v = a.astype(np.float32) / 255.0
        lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 99.5))
        if hi - lo < 1e-4:
            return np.zeros_like(v)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    def index(self, t: float) -> int:
        """The frame of the file that carries master time ``t``."""
        return max(0, min(self.n - 1, int((t + self.offset) * self.fps)))

    def section_index(self, t: float) -> int:
        if not self.sections:
            return 0
        i = bisect_right(self.section_starts, t + self.offset + 1e-9) - 1
        return max(0, min(len(self.sections) - 1, i))

    def slice(self, i: int, width: int = 48):
        j = min(self.n, i + width)
        b = self.bands[i:j]
        if len(b) < width:
            b = np.vstack([b, np.zeros((width - len(b), self.bands.shape[1]),
                                       dtype=np.float32)])
        return b


# --------------------------------------------------------------------------
# the show
# --------------------------------------------------------------------------

class Show:
    def __init__(self, audio: Audio, *, color: str = "true", fps: float = 30.0,
                 glyphs: str = "braille", caps: Caps | None = None):
        self.a = audio
        self.caps = caps or Caps.detect()
        self.color = color
        self.fps = fps
        self.glyphs = glyphs
        self.stage = S.Stage()
        self.beat = 0.0
        self.onset = 0.0
        self.flash = 0.0
        self.glitch = 0.0
        self._tint = 0.05
        self._tear_plan = None
        self._tear_hold = 0
        # Fixed by default so a dump or a preview sheet is reproducible;
        # --tear-random makes every viewing tear differently.
        self._tear_salt = 0
        self.beat_no = 0
        self._last_beat = -1
        self._prev_flux = 0.0
        self.offset = 0.0
        self.show_text = True
        self.last_layout: S.Layout | None = None
        self.last_ctx: S.Ctx | None = None

    def osd(self, text: str, seconds: float = 1.6) -> None:
        """Flash a short control readout in the corner, then let it go.

        Deliberately not a status line: the piece is meant to have no player
        furniture, so this appears only in response to a keypress whose effect
        is otherwise invisible.
        """
        t = self.last_ctx.t if self.last_ctx is not None else 0.0
        self.stage.readout = text
        self.stage.readout_until = t + seconds

    # -- context ---------------------------------------------------------

    def context(self, t: float, w: int, h: int, dt: float, frame: int) -> S.Ctx:
        a = self.a
        i = a.index(t)
        bands = a.bands[i]
        rms = float(a.rms[i])
        flux = float(a.flux[i])
        act = S.act_at(t)

        # beat grid from the measured tempo
        beats = (t - a.beat_phase0) / a.beat_period
        bno = int(math.floor(beats))
        phase = beats - bno
        if bno != self._last_beat:
            self._last_beat = bno
            self.beat = 1.0
        self.beat *= math.exp(-dt * 6.5)

        # transient accent: rising edge of the onset envelope
        if flux > 0.42 and flux > self._prev_flux + 0.04:
            self.onset = 1.0
        self._prev_flux = flux
        self.onset *= math.exp(-dt * 9.0)
        # When the picture comes apart is decided by `tears.py` and by nothing
        # else.  Two things used to leak in here and both are gone: the
        # roughness of the music (which made the same moment tear differently
        # depending on which analysis was loaded) and the measured dropouts
        # (which the stand-in score does not have at all, so a render with no
        # music tore differently from a render with it).  A fault schedule
        # that shifts with the files present is not a schedule.
        target = TEARS.glitch(t, act)
        self.glitch += (target - self.glitch) * min(1.0, dt * 16.0)

        sec_i = a.section_index(t)

        pal = S.palette_at(t)
        ctx = S.Ctx(
            t=t, frame=frame, dt=dt, w=w, h=h,
            bands=bands,
            bass=float(a.bass[i]),
            mid=float(a.mid[i]), high=float(a.high[i]),
            rms=rms, flux=flux, centroid=float(a.centroid[i]),
            beat=self.beat, onset=self.onset, beat_no=bno, beat_phase=phase,
            duration=a.audio_end,
            act=act, act_t=t - _act_start(act, t),
            section=sec_i, pal=pal,
            show_text=self.show_text,
            glitch=self.glitch, reset=S.reset_level(t), flash=self.flash,
            wave=a.wave, wave_rate=a.wave_rate, glyphs=self.glyphs,
        )
        return ctx

    # -- frame -----------------------------------------------------------

    def frame(self, t: float, w: int, h: int, dt: float, frame_no: int) -> Canvas:
        """One full-bleed frame.  No HUD, no analyser, no caption, no bar."""
        cv = Canvas(w, h)
        layout = S.layout_for(w, h)
        ctx = self.context(t, w, h, dt, frame_no)
        ctx.stage = layout.stage
        ctx.tiny = layout.tiny
        self.last_layout, self.last_ctx = layout, ctx

        # The whole frame is tinted by the bass rather than having a drawn
        # border, because a global colour shift has no glyph artefacts at any
        # size.  It is smoothed hard and disabled on a virtual console: a
        # 16-colour palette quantises small changes into discrete jumps, and
        # repainting every cell's background on a slow console reads as the
        # whole screen flashing.
        if self.caps.is_console:
            bg = ctx.pal["base"]
        else:
            target = 0.05 + 0.17 * ctx.bass
            self._tint += (target - self._tint) * min(1.0, dt * 1.5)
            bg = mix(ctx.pal["base"], ctx.pal["accent"],
                     max(0.0, min(0.18, self._tint)))
        cv.clear(" ", DEFAULT, bg)
        self.stage.draw(cv, ctx)
        self._postfx(cv, ctx)
        return cv

    def _postfx(self, cv: Canvas, c: S.Ctx) -> None:
        """Screen tearing, for the stretches where the machine is failing.

        A terminal has no signal to lose, so the tear is built the way a
        failing frame actually looks: the picture slips vertically, it is
        sliced into pieces that each sit somewhere else horizontally, and a
        band is sometimes repeated down the screen.  All three are pure
        horizontal displacements -- nothing is invented, the frame is simply
        not where it should be.

        Everything about it is randomised, including which of the three
        happens: the pattern is redrawn from a fresh generator that is *not*
        derived from the previous frame's, so there is no period to lock on
        to.  A tear that repeats on a two-frame cycle is a flicker; a tear
        that never draws the same thing twice is a fault.  For the same
        reason the pattern is held for a random one to three frames -- real
        tearing persists briefly, it does not strobe.

        Driven by the song's failure moments (see scenes.TEAR_EVENTS) and by
        how rough the music is, never by the beat grid: a screen that slips on
        every downbeat is a metronome, not a fault.
        """
        lvl = c.glitch
        if lvl <= 0.06 or cv.h < 6 or cv.w < 12:
            self._tear_hold = 0
            self._reset_flash(cv, c)
            return

        if self._tear_hold > 0 and self._tear_plan is not None:
            self._tear_hold -= 1
            plan = self._tear_plan
        else:
            rng = np.random.default_rng(
                (int(c.frame) * 2654435761 + self._tear_salt) & 0xFFFFFFFF)
            h = cv.h
            slices = int(rng.integers(1, 2 + int(lvl * 4)))
            cuts = sorted(int(x) for x in rng.integers(1, h, size=slices - 1)) \
                if slices > 1 else []
            edges = [0] + cuts + [h]
            plan = {
                # which faults happen this time; often more than one, sometimes
                # only one, so the effect never settles into a routine
                "roll": rng.random() < 0.22 * lvl,
                "roll_by": int(rng.integers(1, max(2, h // 5))),
                # a slip is occasionally much bigger than the rest
                "offsets": [int(round(rng.integers(-11, 12) * lvl
                                      * (2.2 if rng.random() < 0.18 else 1.0)))
                            for _ in edges[:-1]],
                "edges": edges,
                "repeats": int(rng.integers(0, 1 + int(3 * lvl))),
                "repeat_rows": [(int(rng.integers(0, h)), int(rng.integers(0, h)))
                                for _ in range(int(rng.integers(0, 1 + int(3 * lvl))))],
            }
            self._tear_plan = plan
            self._tear_hold = int(rng.integers(0, 3))

        h, w = cv.h, cv.w
        if plan["roll"]:
            k = plan["roll_by"] % h
            cut = (h - k) * w
            for arr in (cv.ch, cv.fg, cv.bg):
                arr[:] = arr[cut:] + arr[:cut]

        edges = plan["edges"]
        for (a0, a1), d in zip(zip(edges[:-1], edges[1:]), plan["offsets"]):
            if not d:
                continue
            d %= w
            for y in range(a0, min(a1, h)):
                base = y * w
                for arr in (cv.ch, cv.fg, cv.bg):
                    row = arr[base:base + w]
                    arr[base:base + w] = row[-d:] + row[:-d]

        for src_y, dst_y in plan["repeat_rows"]:
            src, dst = (src_y % h) * w, (dst_y % h) * w
            if src != dst:
                cv.ch[dst:dst + w] = cv.ch[src:src + w]
                cv.fg[dst:dst + w] = cv.fg[src:src + w]
                cv.bg[dst:dst + w] = cv.bg[src:src + w]

        self._reset_flash(cv, c)

    @staticmethod
    def _reset_flash(cv: Canvas, c: S.Ctx) -> None:
        """Blank the whole screen, then let it come back.

        The field goes to the hot colour while the glyphs go to the base
        colour, so the frame washes out rather than simply brightening --
        which is what a reset looks like on a screen that has one colour to
        spare.  It fires three times in the whole piece, at the end of each
        long breakdown.
        """
        k = c.reset
        if k <= 0.02:
            return
        hot = c.pal["hot"]
        base = c.pal["base"]
        for i in range(len(cv.ch)):
            cv.bg[i] = mix(cv.bg[i], hot, k)
            if cv.fg[i] != -1:
                cv.fg[i] = mix(cv.fg[i], base, k * 0.85)
            if k > 0.55 and cv.ch[i] == " ":
                cv.ch[i] = " "


def _act_start(act: str, t: float) -> float:
    """When the act containing `t` began."""
    start = 0.0
    for s0, a in S.ACT_TIMELINE:
        if a == act and s0 <= t:
            start = s0
    return start


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

# ==========================================================================
#  the two things you may need to change
# ==========================================================================
#
# 1. Where the music is.  Any supported format works and any location under
#    the project is found, but a file with this name in the repository root
#    needs no arguments at all -- and this is the name the README gives.
TRACK_NAME = "world_execute(me)_-Mili-9893117-192.ogg"

# 2. How far your copy of the music sits from the master this was measured
#    against, in seconds.
#
#      positive  your file has EXTRA silence (or a longer intro) at the front,
#                so the music starts later than the master did -> the picture
#                waits that long before it begins
#      negative  your file has been trimmed at the front, so the music starts
#                earlier -> the picture starts that much sooner
#
#    Every timing written into this piece -- the act table, the shot grid, the
#    caption cues, the failure moments -- is in *master* time.  This is the one
#    number that maps master time onto your file, and it applies to both the
#    sound (which frame of your file to read) and the picture (which moment of
#    the piece to draw), so setting it lines up the whole thing at once.
#
#    A byte-identical copy needs nothing here.  To trim it by eye, run the
#    piece and use [ and ] (0.1s a press); when it looks right, read the value
#    off the status line and put it here.
#
#    It is NOT the same as the song's own "offset" in a player: it does not
#    move the audio, it moves the piece relative to the audio.
MUSIC_OFFSET = 0.0
AUDIO_SUFFIXES = ("*.ogg", "*.mp3", "*.flac", "*.m4a", "*.wav", "*.opus")


def find_track(explicit: str | None,
               required: bool = True) -> Path | None:
    """Locate the music.

    The recording is not in the repository (it is somebody's commercial
    release), so "not found" is a normal state rather than an error: the piece
    runs on its stand-in score instead.  ``required`` keeps the old strict
    behaviour for callers that genuinely cannot continue without it.
    """
    if explicit:
        path = Path(explicit)
        if path.exists() or required:
            return path
        return None
    root = A.PROJECT_DIR.parent
    for pat in AUDIO_SUFFIXES:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    if required:
        raise SystemExit("no audio file found; pass one explicitly")
    return None


def load_score(explicit: str | None = None, *, required: bool = False,
               fps: float = 50.0, rebuild: bool = False):
    """The analysis this piece runs on, whether or not the music is here.

    One place decides between "measured from the track" and "the stand-in
    score", so every caller -- the player, the video renderer, the stills
    sheet, the character survey -- agrees about what is on screen and about
    whether it was measured.

    Returns ``(track, data)``.  ``track`` is ``None`` exactly when the data is
    the stand-in, which is also how callers know to stay silent.
    """
    track = find_track(explicit, required=required)
    if track is None:
        return None, A.stand_in(fps=fps)
    if rebuild:
        A.cache_path_for(track).unlink(missing_ok=True)
    return track, A.load(prepare_cache(track, fps))


def prepare_cache(track: Path, fps: float, quiet: bool = False) -> Path:
    cache = A.cache_path_for(track)
    if not cache.exists():
        if not quiet:
            print(f"[ttymv] measuring {track.name} ...", file=sys.stderr)
        A.save(A.analyze(track, fps), cache)
    return cache


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ttymv", description="world.execute(me); -- a TTY music video")
    p.add_argument("track", nargs="?", help="audio file (default: newest in tree)")
    p.add_argument("--no-text", dest="no_text", action="store_true",
                   help="no words at all on screen, imagery only")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--ao", default=None, help="mpv audio output, e.g. null, pipewire")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=None,
                   help="frames/sec (default: 30, or 20 on a virtual console)")
    p.add_argument("--offset", type=float, default=MUSIC_OFFSET,
                   help="how far your copy of the music sits from the master, "
                        "in seconds; positive if it starts later (defaults to "
                        "the MUSIC_OFFSET in this file)")
    p.add_argument("--color", choices=("true", "256", "16", "auto"),
                   default="auto",
                   help="auto picks from terminfo; a Linux console gets 16")
    p.add_argument("--glyphs", choices=("braille", "block", "auto"),
                   default="auto",
                   help="block if your terminal font has no Braille Patterns")
    p.add_argument("--altscreen", dest="altscreen", action="store_true",
                   default=True, help="use the alternate screen when available")
    p.add_argument("--no-altscreen", dest="altscreen", action="store_false")
    p.add_argument("--sync", dest="sync", action="store_true", default=True,
                   help="synchronised updates (mode 2026) where supported")
    p.add_argument("--no-sync", dest="sync", action="store_false")
    p.add_argument("--tear-random", action="store_true",
                   help="draw a fresh tear pattern every viewing")
    p.add_argument("--stats", action="store_true",
                   help="report frames dropped and bytes written on exit")
    p.add_argument("--size", default=None, help="force WxH (testing)")
    p.add_argument("--rebuild", action="store_true", help="redo the analysis")
    p.add_argument("--no-music", dest="no_music", action="store_true",
                   help="run on the silent stand-in score, ignoring any "
                        "audio file that is present")
    p.add_argument("--list", action="store_true", help="print structure and exit")
    p.add_argument("--dump", default=None,
                   help="render frames at these comma-separated seconds to stdout")
    p.add_argument("--dump-to", default=None, help="directory for --dump output")
    p.add_argument("--dump-every", type=float, default=None,
                   help="with --dump: step this interval across 0..duration")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    caps = Caps.detect()
    if args.color == "auto":
        args.color = caps.best_color_mode()
    if args.glyphs == "auto":
        args.glyphs = caps.best_glyphs()
    if args.fps is None:
        args.fps = caps.best_fps()
    # With no music, the stand-in score drives the whole piece: everything
    # that was written down still happens -- acts, shots, titles, captions,
    # all on the right beat -- and only the part that answers to the sound
    # goes quiet.  Said out loud, because a stand-in that is not announced is
    # a lie about what is on screen.
    track, data = load_score(
        None if args.no_music else args.track,
        required=False, fps=50.0, rebuild=args.rebuild)
    if track is None:
        print("[ttymv] no music file -- running the stand-in score: the whole "
              "piece, silent.", file=sys.stderr)
        print(f"[ttymv] for the real thing, put {TRACK_NAME} in "
              f"{A.PROJECT_DIR.parent}", file=sys.stderr)

    audio = Audio(data, offset=args.offset)

    # The timed lyric is kept as provenance and as the source the on-screen
    # cue sheet is checked against (mv/tools/selftest.py); it is not shown.
    en = _first(A.PROJECT_DIR.parent / "lyrics", "*.lrc")
    lyrics = Lyrics.load(en) if en else EmptyLyrics()

    if args.list:
        return _print_structure(track, audio, lyrics, data.get("source"))

    show = Show(audio, color=args.color,
                fps=args.fps, glyphs=args.glyphs, caps=caps)
    show.offset = args.offset      # live-trimmable with [ and ]
    show.show_text = not args.no_text
    if args.tear_random:
        show._tear_salt = int.from_bytes(os.urandom(4), "little")

    if args.dump is not None or args.dump_every is not None:
        return _dump(show, args)

    force = None
    if args.size:
        w, h = args.size.lower().split("x")
        force = (int(w), int(h))

    clock = make_clock(track, audio.duration, args.start,
                       (not args.no_audio) and track is not None, args.ao)
    if clock.duration <= 0:
        clock.duration = audio.duration
    try:
        return _run(show, clock, force, args)
    finally:
        clock.close()


def _first(d: Path, pat: str, exclude: str | None = None):
    if not d.exists():
        return None
    hits = sorted(p for p in d.glob(pat)
                  if exclude is None or exclude not in p.name)
    return hits[0] if hits else None


def _print_structure(track: Path | None, a: Audio, ly: Lyrics,
                     source: str | None = None) -> int:
    stand_in = source == "stand-in"
    if track is None:
        print(f"track      (none) -- stand-in score, not a measurement")
    else:
        print(f"track      {track}")
    if stand_in:
        print("score      NOT MEASURED: built from the piece's own act "
              "table and tempo")
    print(f"duration   {a.duration:.2f} s  (music ends {a.audio_end:.2f} s)")
    print(f"tempo      {a.bpm:.3f} BPM   ({a.beat_period:.4f} s/beat)")
    print(f"frames     {a.n} @ {a.fps:g} fps")
    print(f"offset     {a.offset:+.1f} s  "
          f"({'master, unshifted' if a.offset == 0 else 'your file vs master'})")
    print(f"lyric      {len(ly.cues)} published cues (provenance, not displayed)")
    print("\nmeasured sections")
    for i, s in enumerate(a.sections):
        bar = "#" * int(max(0.0, min(1.0, (s["rms_db"] + 40) / 45)) * 34)
        print(f"  {i:2d}  {s['start']:7.2f} -> {s['end']:7.2f}  "
              f"{s['end'] - s['start']:6.2f}s  {s['rms_db']:6.1f} dB  {bar}")
    print("\nacts")
    for i, (t, name) in enumerate(S.ACT_TIMELINE):
        nxt = (S.ACT_TIMELINE[i + 1][0] if i + 1 < len(S.ACT_TIMELINE)
               else a.duration)
        print(f"  {t:7.2f} -> {nxt:7.2f}  {name}")
    print("\nwords on screen (everything else is imagery)")
    for at, text, hold in S.TEXT_CUES:
        print(f"  {at:7.2f}  {text}")
    return 0


def _dump(show: Show, args) -> int:
    if args.dump_every:
        times = [i * args.dump_every
                 for i in range(int(show.a.duration / args.dump_every) + 1)]
    else:
        times = [float(x) for x in args.dump.split(",")]
    times = [t for t in times if 0 <= t <= show.a.duration]
    w, h = (int(x) for x in (args.size or "80x24").lower().split("x"))
    out_dir = Path(args.dump_to) if args.dump_to else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    t_cursor = 0.0
    dt = 1.0 / args.fps
    frame_no = 0
    buf = []
    for target in times:
        # evolve state up to the requested moment so particles and peaks look
        # like they would in a live run
        while t_cursor < target - 1e-9:
            step = min(dt, target - t_cursor)
            show.frame(t_cursor, w, h, step, frame_no)
            t_cursor += step
            frame_no += 1
        cv = show.frame(target, w, h, dt, frame_no)
        frame_no += 1
        if out_dir:
            name = f"t{target:07.3f}".replace(".", "_") + ".ans"
            # previews stand in for the graphical-terminal case; the virtual
            # console path is exercised separately by mv/tools/console_check.py
            (out_dir / name).write_text(
                cv.render(args.color, reserve_corner=False), encoding="utf-8")
            (out_dir / name.replace(".ans", ".txt")).write_text(
                cv.render_plain(), encoding="utf-8")
            print(out_dir / name)
        else:
            buf.append(f"===== t={target:.2f}s  {w}x{h} =====\n"
                       + cv.render(args.color, reserve_corner=False))
    if buf:
        sys.stdout.write("\n".join(buf) + "\n")
    return 0


def _run(show: Show, clock: Clock, force, args) -> int:
    term = Term(force, caps=show.caps, altscreen=args.altscreen,
                sync_output=args.sync)
    dt = 1.0 / args.fps
    frame_no = 0
    dropped = 0
    paused = False
    quitting = False
    trimmed = False
    started = time.monotonic()
    next_frame = time.monotonic()
    with term:
        while not quitting:
            w, h = term.size()
            # the clock runs in the file's time; the piece is written in
            # master time, and show.offset is the difference
            t = clock.now() - show.offset
            if t >= show.a.audio_end + 3.2:
                break
            cv = show.frame(t, w, h, dt, frame_no)
            # Only a virtual console needs the bottom-right cell left alone;
            # on a graphical terminal reserving it just leaves a dark notch.
            payload = cv.render(show.color,
                                reserve_corner=show.caps.is_console)
            term.begin_frame()
            if term.resized or frame_no == 0:
                payload = show.caps.clear.decode() + payload
                term.resized = False
            term.end_frame(payload)

            for k in term.read_keys(0.0):
                if k in ("q", "Q", "\x1b"):
                    quitting = True      # leave via the same path as the end
                    break                # of the song, so --stats still prints
                if k == " ":
                    paused = not paused
                    clock.toggle_pause()
                elif k in ("t", "T"):
                    show.show_text = not show.show_text
                elif k in ("]", "."):
                    show.offset += 0.1
                    trimmed = True
                    show.osd(f"MUSIC OFFSET {show.offset:+.1f}s")
                elif k in ("[", ","):
                    show.offset -= 0.1
                    trimmed = True
                    show.osd(f"MUSIC OFFSET {show.offset:+.1f}s")
                elif k == "\x1b[C":
                    clock.seek(clock.now() + 5)
                elif k == "\x1b[D":
                    clock.seek(max(0.0, clock.now() - 5))
                elif k == "r":
                    clock.seek(0.0)

            frame_no += 1
            next_frame += dt
            now = time.monotonic()
            if next_frame < now:
                # Behind schedule -- a slow tty, a big window, or both.  Skip
                # the frames we missed rather than sprinting through them: the
                # picture is looked up from the audio clock every frame, so
                # dropping frames keeps it in sync, while catching up would
                # only flood the terminal with stale ones.
                missed = int((now - next_frame) / dt) + 1
                next_frame += missed * dt
                dropped += missed
            slack = next_frame - time.monotonic()
            if slack > 0:
                time.sleep(slack)
    if trimmed:
        # the whole point of the readout: the number you settled on is the one
        # that goes in MUSIC_OFFSET
        print(f"[ttymv] music offset left at {show.offset:+.1f}s -- put that "
              f"in MUSIC_OFFSET (mv/ttymv/player.py) to keep it",
              file=sys.stderr)
    if args.stats:
        secs = max(1e-6, time.monotonic() - started)
        print(f"[ttymv] {frame_no} frames, {dropped} dropped, "
              f"{term.bytes_written / 1024:.0f} KB written, "
              f"{term.bytes_written / 1024 / secs:.0f} KB/s, "
              f"{term.caps.summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
