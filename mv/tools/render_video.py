#!/usr/bin/env python3
"""Render the MV to a video file, in this terminal's own font.

The MV is a terminal animation, so the honest way to film it is to use the
terminal's font rather than a font that merely happens to be monospace.  That
font is not a matter of taste on this machine -- it is written down:

    ~/.config/kdeglobals   [General] fixed=文泉驿等宽微米黑,10,-1,5,400,...

Konsole has no profile of its own, so it inherits that; see ``kfont.py`` for
how the family, the face index inside the TTC, the cell and the per-character
fallbacks are resolved.

What follows from it: Konsole's cell at 10pt/96dpi is 8x15 pixels, an aspect
of 1.875:1 -- *not* the 2:1 that the MV's dot geometry assumes when it draws
round figures.  The renderer keeps the terminal's true cell rather than the
assumed one, so what the video shows is what the terminal shows.

Frames are rasterised and piped straight into ffmpeg as raw RGB, so nothing
touches the disk between the renderer and the encoder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "mv"))

from ttymv import analyze as A                                  # noqa: E402
from ttymv.canvas import DEFAULT, WIDE                          # noqa: E402
from ttymv.player import (MUSIC_OFFSET, TRACK_NAME, Audio, Show,  # noqa: E402
                           find_track, load_score)     # noqa: E402

import kfont                                                    # noqa: E402

SCALE = 4                       # cell multiple; see --scale
ROWS = 54


def unpack(c: int) -> tuple[int, int, int]:
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


class Raster:
    """Cell grid -> RGB, in the terminal's font.

    Two things make this fast enough to be worth doing.  Backgrounds collapse
    into one rectangle per run of equal colour.  And glyphs are drawn from an
    atlas: rasterising a glyph costs a fraction of a millisecond, so calling
    ``draw.text`` per run re-renders the same letters thousands of times per
    frame.  Each character is instead rendered once into an 8-bit mask by the
    font stack and composited with ``paste``, a C-level blend.
    """

    def __init__(self, stack: kfont.FontStack, cols: int, rows: int,
                 bg: int = 0x000000):
        self.stack = stack
        self.cw, self.ch = stack.cw, stack.ch
        self.cols, self.rows = cols, rows
        self.size = (cols * self.cw, rows * self.ch)
        self.bg = bg
        self.img = Image.new("RGB", self.size, unpack(bg))
        self.draw = ImageDraw.Draw(self.img)
        self._atlas: dict[str, Image.Image] = {}

    def mask(self, ch: str) -> Image.Image:
        m = self._atlas.get(ch)
        if m is None:
            m = self.stack.mask(ch)
            self._atlas[ch] = m
        return m

    def frame(self, cv) -> Image.Image:
        d, cw, ch, w = self.draw, self.cw, self.ch, cv.w
        img = self.img
        d.rectangle([0, 0, self.size[0] - 1, self.size[1] - 1],
                    fill=unpack(self.bg))
        chs, fgs, bgs = cv.ch, cv.fg, cv.bg

        # backgrounds, one rectangle per run of equal colour
        for y in range(cv.h):
            base = y * w
            top, bottom = y * ch, y * ch + ch - 1
            x = 0
            while x < w:
                col = bgs[base + x]
                x2 = x
                while x2 + 1 < w and bgs[base + x2 + 1] == col:
                    x2 += 1
                if col != DEFAULT:
                    d.rectangle([x * cw, top, (x2 + 1) * cw - 1, bottom],
                                fill=unpack(col))
                x = x2 + 1

        # glyphs, composited through cached masks
        paste = img.paste
        for y in range(cv.h):
            base = y * w
            py = y * ch
            for x in range(w):
                c2 = chs[base + x]
                if c2 == " " or c2 == WIDE:
                    continue
                col = fgs[base + x]
                # the x position comes from the column, never from a running
                # counter: a counter that skips blanks shifts everything after
                # the first space one cell to the left
                # minus the bleed, so an overshooting block or box glyph
                # overlaps its neighbour and the join has no hairline
                paste(unpack(col) if col != DEFAULT else (255, 255, 255),
                      (x * cw - self.stack.BLEED, py - self.stack.BLEED),
                      self.mask(c2))
        return img


def grid_for(stack: kfont.FontStack, rows: int, aspect: float = 16 / 9) -> tuple[int, int]:
    """Columns and rows that come closest to ``aspect`` with square cells."""
    rows += rows % 2
    cols = int(round(rows * aspect * stack.ch / stack.cw))
    cols += cols % 2
    return max(2, cols), rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--scale", type=int, default=SCALE,
                    help="cell size as a multiple of the terminal's own cell")
    ap.add_argument("--dpi", type=float, default=96.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "world.execute(me).tty-mv.mp4"))
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--audio-bitrate", default="256k")
    ap.add_argument("--offset", type=float, default=MUSIC_OFFSET,
                    help="how far your copy of the music sits from the master "
                         "(s); the picture is written in master time and the "
                         "audio is taken from your file, so this lines the two "
                         "up.  Defaults to MUSIC_OFFSET.")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None,
                    help="render only up to this many seconds (default: the end)")
    ap.add_argument("--bench", type=int, default=0,
                    help="render N frames and report the speed, encode nothing")
    ap.add_argument("--sheet", default=None,
                    help="write PNG frames at these timestamps (comma "
                         "separated) and exit; streams the whole show once")
    a = ap.parse_args(argv)

    fixed = kfont.read_kde_fixed()
    stack = kfont.FontStack(fixed, scale=a.scale, dpi=a.dpi)
    cols, rows = grid_for(stack, a.rows)
    W, H = cols * stack.cw, rows * stack.ch

    # The music is not in the repository, so a render has to be possible
    # without it: the stand-in score drives the picture and the file comes out
    # silent.  See README 4.5.
    track, data = load_score(fps=50.0)
    audio = Audio(data, offset=a.offset)
    source = "stand-in" if track is None else track.name
    if track is None:
        print("[ttymv] no music file -- rendering on the stand-in score, "
              f"silent.  Put {TRACK_NAME} in {A.PROJECT_DIR.parent} for the "
              "real thing.", file=sys.stderr)
    clock = audio
    show = Show(clock, color="true", fps=a.fps)
    end = min(clock.audio_end + 3.2, a.end) if a.end else clock.audio_end + 3.2
    start = max(0.0, a.start)
    total = int((end - start) * a.fps)

    print(f"track    {source}" + (f"   offset {a.offset:+.1f}s"
                                  if a.offset else ""))
    print(f"font     {stack.describe()}")
    print(f"grid     {cols}x{rows}   frame {W}x{H}   {W / H:.5f}:1")
    print(f"video    {a.fps:g} fps, {total} frames, {end:.1f}s")

    if a.sheet is not None:
        # The Show is a streaming driver: elements that depend on accumulated
        # state (the title card's reveal, the ghost decimation, the particle
        # fields) are simply absent if you jump straight to a timestamp.  So
        # the sheet streams every frame up to the last requested one, exactly
        # as playback does, and grabs the ones asked for on the way past.
        want = sorted(float(x) for x in str(a.sheet).split(",") if x.strip())
        if not want:
            print("--sheet needs at least one timestamp", file=sys.stderr)
            return 2
        last = max(0, int(round((want[-1] - start) * a.fps)))
        outdir = ROOT / "mv" / "cache" / "preview"
        outdir.mkdir(parents=True, exist_ok=True)
        deadline = 0
        r = None
        written = []
        for i in range(last + 1):
            cv = show.frame(start + i / a.fps, cols, rows, 1 / a.fps, i)
            if i < deadline:
                continue
            while deadline < len(want) and i >= int(round((want[deadline] - start) * a.fps)):
                if r is None:
                    r = Raster(stack, cols, rows, bg=cv.bg[0])
                out = outdir / f"frame-{want[deadline]:08.2f}.png"
                r.frame(cv).save(out)
                written.append(out.name)
                deadline += 1
        print(f"streamed {last + 1} frames; wrote {', '.join(written)}")
        print(f"faces    {', '.join(f'{k} ({v})' for k, v in sorted(stack._usage.items()))}")
        return 0

    if a.bench:
        r = Raster(stack, cols, rows,
                   bg=show.frame(start, cols, rows, 1 / 30, 0).bg[0])
        t0 = time.perf_counter()
        for i in range(a.bench):
            cv = show.frame(start + i / a.fps, cols, rows, 1 / a.fps, i)
            r.frame(cv)
        dt = (time.perf_counter() - t0) / a.bench
        print(f"bench    {dt * 1000:.1f} ms/frame -> {dt * total / 60:.1f} min "
              f"for the whole piece")
        print(f"faces    {', '.join(f'{k} ({v})' for k, v in sorted(stack._usage.items()))}")
        return 0

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", f"{a.fps}", "-i", "-",
    ]
    if track is not None:
        # The picture is master time; the file is not.  Seek the audio input to
        # where master time `start` lives in this particular file.  (This used
        # to insert -ss before the *rawvideo* input, which seeks a pipe -- the
        # audio never moved.)
        at = a.start + a.offset
        cmd += ["-ss", f"{max(0.0, at):.3f}", "-i", str(track)]
    cmd += [
        "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "6.1",
    ]
    cmd += (["-c:a", "aac", "-b:a", a.audio_bitrate] if track is not None
            else ["-an"])
    cmd += ["-movflags", "+faststart", "-shortest", str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    first = show.frame(start, cols, rows, 1 / a.fps, 0)
    r = Raster(stack, cols, rows, bg=first.bg[0])

    t0 = time.perf_counter()
    written = 0
    try:
        for i in range(total):
            t = start + i / a.fps
            cv = show.frame(t, cols, rows, 1 / a.fps, i)
            img = r.frame(cv)
            proc.stdin.write(img.tobytes())
            written += 1
            if i % 150 == 0 or i == total - 1:
                el = time.perf_counter() - t0
                rate = written / max(1e-6, el)
                left = (total - written) / max(1e-6, rate)
                print(f"  {written:5d}/{total}  {100 * written / total:5.1f}%  "
                      f"{rate:5.1f} fps  eta {left / 60:4.1f} min", flush=True)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()

    if proc.returncode != 0:
        print("ffmpeg failed", file=sys.stderr)
        return 1
    size = Path(a.out).stat().st_size
    print(f"\nwrote {a.out}  {size / 1e6:.1f} MB  "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    print(f"faces    {', '.join(f'{k} ({v})' for k, v in sorted(stack._usage.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
