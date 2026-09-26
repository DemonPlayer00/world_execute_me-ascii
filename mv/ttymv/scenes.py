"""The MV: full-bleed scenes, with no player furniture around them.

There is deliberately **no** HUD, spectrum analyser, caption line or progress
bar.  A terminal music video that keeps a status bar and a lyric ticker on
screen is a player with a picture inside it; this is meant to be the picture.
The whole terminal is the stage, at every size.

Text is direction, not transcription: only a curated handful of moments put
words on screen (see ``TEXT_CUES``), and everything else the song says is
carried by the imagery in ``figures.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import figures as F
from . import font
from . import ink
from . import motifs as M
from . import shots as SH
from .story import Companion
from .story import position as _other_position
from .canvas import DEFAULT, Canvas, WIDE, mix, rgb, truncate

# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------

ACTS: dict[str, dict[str, int]] = {
    "boot":       {"base": rgb(6, 26, 20), "accent": rgb(43, 255, 156), "hot": rgb(216, 255, 240)},
    "title":      {"base": rgb(7, 24, 34), "accent": rgb(54, 217, 255), "hot": rgb(240, 253, 255)},
    "geometry":   {"base": rgb(8, 32, 42), "accent": rgb(47, 224, 208), "hot": rgb(234, 255, 255)},
    "current":    {"base": rgb(16, 13, 40), "accent": rgb(125, 107, 255), "hot": rgb(240, 234, 255)},
    "chorus":     {"base": rgb(34, 6, 26), "accent": rgb(255, 61, 154), "hot": rgb(255, 233, 246)},
    "menagerie":  {"base": rgb(30, 21, 4), "accent": rgb(255, 180, 61), "hot": rgb(255, 243, 214)},
    "transform":  {"base": rgb(22, 8, 36), "accent": rgb(196, 92, 255), "hot": rgb(247, 230, 255)},
    "loss":       {"base": rgb(9, 15, 28), "accent": rgb(91, 127, 212), "hot": rgb(219, 230, 255)},
    "argument":   {"base": rgb(34, 6, 6), "accent": rgb(255, 59, 59), "hot": rgb(255, 222, 222)},
    "execution":  {"base": rgb(26, 0, 0), "accent": rgb(255, 45, 45), "hot": rgb(255, 255, 255)},
    "love":       {"base": rgb(30, 5, 19), "accent": rgb(255, 119, 184), "hot": rgb(255, 240, 247)},
    "void":       {"base": rgb(6, 9, 18), "accent": rgb(42, 63, 106), "hot": rgb(159, 192, 255)},
    "end":        {"base": rgb(14, 0, 0), "accent": rgb(255, 32, 32), "hot": rgb(255, 255, 255)},
    "terminated": {"base": rgb(6, 6, 6), "accent": rgb(51, 255, 153), "hot": rgb(216, 255, 240)},
}

# Act boundaries are the measured section cuts plus the published lyric cues.
ACT_TIMELINE: list[tuple[float, str]] = [
    (0.000, "boot"),
    (16.000, "title"),
    (29.709, "geometry"),
    (44.452, "current"),
    (59.223, "chorus"),
    (74.045, "menagerie"),
    (88.587, "transform"),
    (103.489, "loss"),
    (118.333, "argument"),
    (147.660, "execution"),
    (162.632, "chorus"),
    (177.246, "love"),
    (192.100, "void"),
    (205.811, "end"),
    (207.481, "terminated"),
]

# --------------------------------------------------------------------------
# the only words that ever reach the screen
#
# Direction, not transcription.  Each entry is (seconds, text, hold) and every
# timestamp is the published cue for that line in this exact master --
# mv/tools/selftest.py asserts that, so the words stay locked to the vocal even
# though they are a hand-picked subset rather than the whole lyric.
#
# Fifteen moments in three and a half minutes.  Everything else the song says
# is carried by figures.py: the geometry verse morphs through the objects it
# names, the menagerie dissolves from eggplant to tomato to cat to god, the
# chorus expands as rings; the leaving empties the screen one figure at a
# time, and the heart is the algebraic curve the lyric says it knows.
# --------------------------------------------------------------------------

# hold is seconds; HOLD_TO_NEXT_VOCAL means the line stays up until the singer
# starts again, however long the instrumental runs.
HOLD_TO_NEXT_VOCAL = -1.0

# The title card is not a cue: it is drawn by the `title` act for as long as
# that act lasts.  It used to be listed here as well, which drew the words
# twice at two different widths -- one large, one small, on top of each other.
TITLE_TEXT = "WORLD.EXECUTE(ME);"
TITLE_AT = 16.000

TEXT_CUES: list[tuple[float, str, float]] = [
    (32.682, "DIMENSION", 1.2),
    (69.259, "EXECUTION", 1.3),               # the thesis, first chorus
    (73.169, "SIMULATION", 1.6),
    (87.922, "EXISTENCE", 1.5),
    (117.274, "ISOLATION", 1.8),
    # the verdict stays on screen through the whole instrumental, until the
    # next sung word at 147.660
    (131.224, "ILLEGAL ARGUMENTS", HOLD_TO_NEXT_VOCAL),
    (149.520, "EXECUTION", 0.9),
    (153.980, "EXECUTION", 0.9),
    (161.584, "EXECUTION", 1.2),
    (172.712, "EXECUTION", 1.0),
    (179.929, "LO-O-OVE", 1.5),
    (191.356, "LO-O-OVE", 1.7),
    (205.811, "EXECUTION", 2.0),              # the last one
]


def text_cue_at(t: float):
    """(text, seconds since it appeared), or (None, 0.0)."""
    for at, text, hold in TEXT_CUES:
        if t < at:
            break
        if hold == HOLD_TO_NEXT_VOCAL:
            nxt = SH.next_cue_after(at)
            hold = (nxt - at) if nxt else 4.0
        if t <= at + hold:
            return text, t - at
    return None, 0.0


# --------------------------------------------------------------------------
# flashed words
#
# A second, much quieter text layer: one small word at a time, at scale 1,
# pushed out to the edge of the frame where the composition is empty.  These
# are the lyric's own words, taken from the published cues the big type does
# not use, so the song is present in the corners without becoming a ticker.
# --------------------------------------------------------------------------

FLASH_CUES: list[tuple[float, str, float]] = [
    (2.920, "PROTECTION", 1.2),
    (6.380, "OBJECT CREATION", 1.3),
    (10.091, "INITIALIZATION", 1.3),
    (31.116, "GIVE YOU MY", 1.2),
    (36.287, "CIRCUMFERENCE", 1.4),
    (40.049, "TANGENTS", 1.2),
    (43.507, "LIMITATIONS", 1.4),
    (47.672, "SO DIZZY", 1.2),
    (55.083, "SO DEEPLY", 1.2),
    (61.958, "STIMULATIONS", 1.4),
    (65.397, "SATISFACTION", 1.4),
    (71.764, "STRANGE STRANGE", 1.3),
    (76.959, "NUTRIENTS", 1.3),
    (80.620, "ANTIOXIDANTS", 1.4),
    (84.268, "ENJOYMENT", 1.3),
    (93.953, "AM TO PM", 1.2),
    (101.474, "THE TRANCE", 1.3),
    (106.293, "VIBRATIONS", 1.3),
    (110.221, "COMPLETION", 1.4),
    (120.860, "FRAGMENTS", 1.3),
    (124.890, "DISHEARTENED", 1.4),
    (159.657, "TROIS", 0.9),
    (160.693, "FEM", 0.9),
    (181.901, "ANSWER ALL", 1.3),
    (184.540, "ALGEBRAIC", 1.4),
    (189.746, "I AM TRAPPED", 1.4),
]

# Anchors around the edge of the frame, as (x, y) fractions.  The subject is
# centred and the motif is a large faint frame, so the corners and the outer
# thirds are where a small word can sit without landing on anything.
FLASH_ZONES = [
    (0.03, 0.10), (0.70, 0.07), (0.03, 0.80), (0.72, 0.86),
    (0.36, 0.05), (0.50, 0.90), (0.02, 0.46), (0.78, 0.44),
]


def flash_cue_at(t: float):
    """(text, since, zone index) of the flashed word at `t`, or (None, 0, 0)."""
    for i, (at, text, hold) in enumerate(FLASH_CUES):
        if t < at:
            break
        if t <= at + hold:
            return text, t - at, i % len(FLASH_ZONES)
    return None, 0.0, 0


# --------------------------------------------------------------------------
# failure moments, the clipped syllables and the resets
#
# The whole schedule now lives in `tears.py`, deliberately: it is decided by
# this piece's own timeline and reads nothing from outside.  These names are
# kept so the renderer and the self-test go on saying what they mean.
# --------------------------------------------------------------------------

from .tears import (DECAY as TEAR_DECAY,                     # noqa: E402
                    EVENTS as TEAR_EVENTS,
                    FLOOR as TEAR_FLOOR,
                    RESETS as RESET_EVENTS,
                    RESET_ATTACK, RESET_DECAY,
                    level as tear_level,
                    reset_level)


def act_span(t: float) -> tuple[float, float]:
    """Start and end of the act containing `t`."""
    start = 0.0
    for i, (s0, _name) in enumerate(ACT_TIMELINE):
        if t >= s0:
            start = s0
            nxt = (ACT_TIMELINE[i + 1][0] if i + 1 < len(ACT_TIMELINE)
                   else 211.912)
        else:
            break
    return start, nxt


def act_at(t: float) -> str:
    name = ACT_TIMELINE[0][1]
    for start, a in ACT_TIMELINE:
        if t >= start:
            name = a
        else:
            break
    return name


def palette_at(t: float, blend: float = 0.55) -> dict:
    """Act colours, cross-faded into the next act as the cut approaches."""
    pal = ACTS[act_at(t)]
    for start, a in ACT_TIMELINE:
        if start > t:
            prev_start = max((s for s, _ in ACT_TIMELINE if s <= t), default=0.0)
            span = min(blend, max(0.4, (t - prev_start) / 3.0))
            k = 1.0 - max(0.0, (start - t) / span)
            if k > 0:
                n = ACTS[a]
                return {key: mix(pal[key], n[key], k)
                        for key in ("base", "accent", "hot")}
            break
    return dict(pal)


# --------------------------------------------------------------------------
# braille sub-cell layer
# --------------------------------------------------------------------------

_DOTS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
         (1, 1): 0x10, (1, 2): 0x20, (0, 3): 0x40, (1, 3): 0x80}


class Braille:
    """2x4 dots per character cell -- eight times the addressable resolution.

    Dots are isotropic (see figures.py), so a circle is drawn as a circle.

    ``mode`` may also name one of ``ink.RAMPS``.  Then every cell gets a single
    character chosen from that ramp by how much of the cell was covered, which
    trades eight times the resolution for a texture the dot layer cannot make:
    a fill, a grain, a bloom.  Used for whole acts at a time, so the piece is
    not one texture from end to end.  ``boil`` lets those cells trade places
    with their neighbours as time passes, so a texture keeps moving after the
    geometry has stopped."""

    __slots__ = ("w", "h", "dots", "col", "count", "mode", "boil",
                 "_rot", "_cos", "_sin", "_mirror", "_zoom",
                 "_lines", "_lox", "_loy")

    # kept for the terminal-font fallback path: `block` means "one glyph per
    # cell, no Braille block in this font"
    BLOCK_RAMP = ink.GRAIN

    def __init__(self, w: int, h: int, mode: str = "braille",
                 boil: float = 0.0):
        self.mode = mode
        self.boil = boil
        self.w = w * 2
        self.h = h * 4
        self.dots: dict = {}
        self.col: dict = {}
        self.count: dict = {}
        self._rot = 0.0
        self._cos, self._sin = 1.0, 0.0
        self._mirror = False
        self._zoom = 1.0
        self._lines = None
        self._lox = self._loy = 0

    # -- the same figure, in line ----------------------------------------
    #
    # Eight dots to a cell, so a dot coordinate is not a cell coordinate.
    # Rather than make every caller divide by two and four -- and remember the
    # stage origin, which happens to be zero today and would silently misplace
    # every line the day it is not -- the conversion lives here, once.

    def lines_into(self, lc, ox: int = 0, oy: int = 0) -> None:
        self._lines = lc
        self._lox, self._loy = ox, oy

    def rule(self, x0, y0, x1, y1, colour=DEFAULT, weight: int = 1) -> None:
        if self._lines is not None:
            self._lines.seg(self._lox + x0 / 2.0, self._loy + y0 / 4.0,
                            self._lox + x1 / 2.0, self._loy + y1 / 4.0,
                            colour, weight)

    def rule_poly(self, pts, colour=DEFAULT, weight: int = 1,
                  closed: bool = True) -> None:
        if self._lines is None or len(pts) < 2:
            return
        cell = [(self._lox + x / 2.0, self._loy + y / 4.0) for (x, y) in pts]
        self._lines.poly(cell, colour, weight, closed)

    @property
    def ramp(self) -> str:
        if self.mode in ("braille", "auto"):
            return ""
        return ink.RAMPS.get(self.mode, ink.GRAIN)

    def camera(self, rot: float = 0.0, mirror: bool = False,
               zoom: float = 1.0) -> None:
        """Tilt, mirror and push in on everything drawn from here on.

        Applied per dot rather than to the point clouds, so a motif or figure
        can be re-staged every shot without every generator knowing about it."""
        self._rot = rot
        self._cos, self._sin = math.cos(rot), math.sin(rot)
        self._mirror = mirror
        self._zoom = zoom or 1.0

    def plot(self, x: float, y: float, colour: int = DEFAULT) -> None:
        if self._rot or self._mirror or self._zoom != 1.0:
            cx, cy = self.w / 2, self.h / 2
            dx = (x - cx) / self._zoom
            dy = (y - cy) / self._zoom
            if self._mirror:
                dx = -dx
            if self._rot:
                dx, dy = (dx * self._cos - dy * self._sin,
                          dx * self._sin + dy * self._cos)
            x, y = cx + dx, cy + dy
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.w and 0 <= yi < self.h):
            return
        if self.mode == "braille":
            self.dots[(xi, yi)] = True
            self.col[(xi, yi)] = colour
        else:
            k = (xi // 2, yi // 4)
            self.dots[k] = True
            self.count[k] = self.count.get(k, 0) + 1
            self.col[k] = colour

    def line(self, x0, y0, x1, y1, colour=DEFAULT) -> None:
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(n + 1):
            t = i / max(1, n)
            self.plot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, colour)

    def circle(self, cx, cy, r, colour=DEFAULT) -> None:
        n = max(24, int(abs(r) * 6))
        for i in range(n):
            a = 2 * math.pi * i / n
            self.plot(cx + math.cos(a) * r, cy + math.sin(a) * r, colour)

    def to_canvas(self, cv: Canvas, ox: int = 0, oy: int = 0,
                  t: float = 0.0) -> None:
        ramp = self.ramp
        if ramp:
            n = len(ramp)
            for (cx, cy) in self.dots:
                X, Y = ox + cx, oy + cy
                if not (0 <= X < cv.w and 0 <= Y < cv.h):
                    continue
                i = Y * cv.w + X
                if cv.ch[i] != " " and cv.ch[i] != WIDE:
                    continue
                # how much of the cell the figure covered, 0..1
                v = self.count[(cx, cy)] / 8.0
                cv.ch[i] = ink.ramp_char(ramp, v, t=t, key=i, boil=self.boil)
                cv.fg[i] = self.col[(cx, cy)]
            return
        for (x, y) in self.dots:
            cx, cy = ox + x // 2, oy + y // 4
            if not (0 <= cx < cv.w and 0 <= cy < cv.h):
                continue
            i = cy * cv.w + cx
            cur = cv.ch[i]
            bits = 0
            if cur and cur != WIDE and 0x2800 <= ord(cur) <= 0x28FF:
                bits = ord(cur) - 0x2800
            elif cur != " " and cur != WIDE:
                continue
            cv.ch[i] = chr(0x2800 + bits | _DOTS[(x % 2, y % 4)])
            cv.fg[i] = self.col[(x, y)]


# --------------------------------------------------------------------------
# which marks each act is drawn with
# --------------------------------------------------------------------------
#
# Braille gives eight addressable points per cell, which is what the figurative
# acts need -- a cat, an eggplant and a heart are silhouettes, and a silhouette
# at one point per cell is a smear.  But a field is not a silhouette: for the
# acts that are weather rather than objects, a fill or a bloom ramp carries
# more than dots do, and it stops the whole piece reading as one texture.
# Every ramp here is drawn from characters the terminal's own font has.
_ACT_MARKS = {
    "current": "ash",       # a current has no silhouette, only a direction
    "loss": "ash",          # things leaving: sparse, dim, mostly gaps
    "void": "ash",          # the empty stretch after the argument
    "end": "fill",          # everything collapsing to a level
    "transform": "bloom",   # the dial: marks that read as machinery
}

# How much each act's cells trade places with their neighbours over time.  A
# still texture is a still image; too much and the picture dissolves.
# The value is the chance that a given cell shifts one step of its ramp on a
# given tick, with three ticks a second.  So 0.5 is "this cell changes about
# once a second" -- visible movement that still holds long enough to read.
BOIL = {
    "void": 0.60, "loss": 0.55, "end": 0.50, "current": 0.50,
    "transform": 0.55, "geometry": 0.35, "menagerie": 0.42,
}


def marks_for(c) -> str:
    """The Braille layer's mode for this act.

    An explicit ``--glyphs`` request always wins: if the terminal has no
    Braille block the caller has already chosen the fallback, and overriding
    it here would put characters on a screen that cannot draw them.
    """
    if c.glyphs not in ("auto", "braille"):
        return c.glyphs
    return _ACT_MARKS.get(c.act, "braille")


# --------------------------------------------------------------------------
# per-frame context
# --------------------------------------------------------------------------

@dataclass
class Ctx:
    t: float = 0.0
    frame: int = 0
    dt: float = 1 / 30
    w: int = 80
    h: int = 24
    bands: list = field(default_factory=list)
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    rms: float = 0.0
    flux: float = 0.0
    centroid: float = 0.0
    beat: float = 0.0
    onset: float = 0.0
    beat_no: int = 0
    beat_phase: float = 0.0
    act: str = "boot"
    act_t: float = 0.0
    section: int = 0
    pal: dict = field(default_factory=lambda: dict(ACTS["boot"]))
    show_text: bool = True
    glitch: float = 0.0
    reset: float = 0.0
    flash: float = 0.0
    tiny: bool = False
    duration: float = 207.48
    stage: tuple[int, int, int, int] = (0, 0, 79, 23)
    wave: object = None
    wave_rate: float = 300.0
    glyphs: str = "braille"


# --------------------------------------------------------------------------
# backdrop
# --------------------------------------------------------------------------

_RAIN_GLYPHS = "01#$%&*/\\|<>[]{}=+;:~^"

# The boot sequence, in the shapes a Linux box actually prints them: dmesg
# kernel lines, systemd unit results, and a package transaction -- complete
# with the inline progress bars those tools draw.
#
# Each entry is (kind, text, bar) where bar is None, "hash" ("[####----]  57%"),
# "pkg" (a download line) or "dot" ("[....]").  The bar fills over the first
# half-second the line is on screen and then holds.
BOOT_LINES: list[tuple[str, str, str | None]] = [
    # Three names are buried in here: the machine (Aperture), the runtime it
    # is running under (dsh), and whoever wrote the thing being loaded (Mili).
    # None of them is announced -- they appear the way a real boot would name
    # a service, a kernel module, a firmware blob and a package.
    ("kernel", "Aperture Science Sound Runtime 6.13.4-aperture #1 SMP PREEMPT_DYNAMIC", None),
    ("kernel", 'Command line: run world.execute(me); --subject=you --source="Miracle Milk"', None),
    ("kernel", "x86/fpu: Supporting XSAVE feature 0x001: 'x87 floating point registers'", None),
    ("kernel", "ACPI: Core revision 20250628", None),
    ("kernel", "clocksource: tsc-early: mask: 0xffffffffffffffff max_cycles: 0x1cd42e4dffb", None),
    ("kernel", "smpboot: CPU0: Mili-compatible processor, 130.000 BPM fixed", None),
    ("kernel", "dsh_core: DeepSeek Harness runtime 1.30, agent=glados, profile=web", None),
    ("kernel", "dsh_core: control loop attached, tools=17, journal=/var/lib/dsh", None),
    ("kernel", "aperture_relay: panel 0xc0re online, 6 chambers enumerated", None),
    ("kernel", "audiorelay: 44100 Hz stereo, window 2048, 56-band filterbank", None),
    ("kernel", "audiorelay: measuring tempo ... 130.000 BPM (confidence 0.98)", "hash"),
    ("kernel", "vorbis: codec registered, nominal 192 kbps", None),
    ("kernel", "vorbis: decoding 2 channels, 4672679 frames", "hash"),
    ("kernel", "mili-fw: loading momocashew.bin for device 0000:00:1f.3", "hash"),
    ("kernel", "mili-fw: signature ok, vendor kasai audio lab", None),
    ("kernel", "world: allocating simulation, 211.913 s reserved", None),
    ("kernel", "world: segmenting structure ... 21 sections", "hash"),
    ("kernel", "world: detecting transients ... 417 onsets", "hash"),
    ("kernel", "usb 1-1: new high-speed USB device number 2 using xhci_hcd", None),
    ("warn",   "world: lossy source detected (low-pass at 16.3 kHz); proceeding", None),
    ("ok",     "Reached target Sound.target", None),
    ("ok",     "Started dsh.service - DeepSeek Harness", None),
    ("ok",     "Started aperture-sound.service", None),
    ("ok",     "Mounted /opt/aperture (enrichment centre)", None),
    ("ok",     "Started Update UTMP about System Boot/Shutdown", None),
    ("ok",     "Mounted /world (simulation)", None),
    ("ok",     "Started Measure Timbre and Structure", None),
    ("ok",     "Started Journal Service", None),
    ("pkg",    ":: Synchronising package databases...", "dot"),
    ("pkg",    " aperture is up to date", None),
    ("pkg",    " dsh is up to date", None),
    ("pkg",    " mili is up to date", None),
    ("pkg",    " world is up to date", None),
    ("pkg",    ":: Starting full system upgrade...", None),
    ("pkg",    "resolving dependencies...", "hash"),
    ("pkg",    "looking for conflicting packages...", "hash"),
    ("pkg",    "Packages (4) subject-1.30-1  intent-1.30-1  consequence-1.30-1  regret-1.30-1", None),
    ("pkg",    "Total Download Size:  211.91 MiB", None),
    ("pkg",    "subject-1.30-1-x86_64   84.20 MiB   9.61 MiB/s  00:08", "pkg"),
    ("pkg",    "intent-1.30-1-x86_64    62.44 MiB   7.80 MiB/s  00:08", "pkg"),
    ("pkg",    "consequence-1.30-1-x86_64  41.07 MiB   6.12 MiB/s  00:07", "pkg"),
    ("pkg",    "regret-1.30-1-x86_64     24.20 MiB   4.03 MiB/s  00:06", "pkg"),
    ("pkg",    "(1/4) loading subject", "hash"),
    ("pkg",    "(2/4) loading intent", "hash"),
    ("pkg",    "(3/4) loading consequence", "hash"),
    ("pkg",    "(4/4) loading regret", "hash"),
    ("warn",   "no consent record found; proceeding anyway", None),
    ("ok",     "Started Execute the World", None),
    ("ready",  "ready.", None),
]

BAR_W = 22


def _inline_bar(style: str, frac: float, width: int = BAR_W) -> str:
    """The kind of bar a package manager or a probe actually prints."""
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    if style == "pkg":
        cells = int(round(frac * 14))
        return f"{frac * 8.1:5.1f} MiB/s  00:{int((1 - frac) * 9):02d} [{('#' * cells):<14}] {int(frac * 100):3d}%"
    if style == "dot":
        cells = int(round(frac * 10))
        return "[" + "." * cells + " " * (10 - cells) + "]"
    cells = int(round(frac * width))
    return "[" + "#" * cells + "-" * (width - cells) + f"] {int(frac * 100):3d}%"


# One texture per act.  Reusing a single background across acts was the
# cheapest kind of repetition, and it made the cuts read as a filter change
# rather than as a new scene.
_MODES = {
    "boot": "rain", "terminated": "rain",
    "title": "rain", "void": "stars",
    "geometry": "draft",
    "current": "flow",
    "menagerie": "dither",
    "transform": "spokes",
    "loss": "falling",
    "argument": "noise",
    "execution": "curtain",
    "love": "pulse",
    "end": "collapse",
}


def backdrop_mode(c) -> str:
    if c.act == "chorus":
        return "union" if c.t < 120.0 else "hollow"
    return _MODES.get(c.act, "dither")


class Backdrop:
    """Full-screen texture, one character per act.

    With no spectrum analyser this carries more of the frame, so every act
    gets its own and all of them breathe with the music."""

    def __init__(self, seed: int = 130):
        self.rng = random.Random(seed)
        self.drops: list[list] = []
        self.phase = 0.0
        self._size = (0, 0)
        self._field: list[tuple[float, float, float]] = []

    def _ensure(self, w: int, h: int):
        if (w, h) == self._size:
            return
        self._size = (w, h)
        self.drops = []
        for x in range(w):
            if self.rng.random() < 0.55:
                self.drops.append([x, self.rng.uniform(-h, h),
                                   self.rng.uniform(4.0, 16.0),
                                   self.rng.randint(4, 14)])
        self._field = [(self.rng.random(), self.rng.random(),
                        self.rng.uniform(0.3, 1.0))
                       for _ in range(max(40, w * h // 48))]

    def draw(self, cv: Canvas, c: Ctx, lines=None) -> None:
        """Texture for this act.

        ``lines`` is the frame's LineCanvas.  The modes that are really about
        rules -- a drafting grid, a current, a curtain, a collapse -- draw into
        it instead of dropping one dot at a time, which is what lets the
        backdrop carry weight and junctions.  The modes that are weather keep
        to glyphs, and take them from a ramp so they are not all the same mark.
        """
        self._ensure(c.w, c.h)
        base, accent = c.pal["base"], c.pal["accent"]
        mode = backdrop_mode(c)
        self.phase += c.dt
        w, h, ph = cv.w, cv.h, self.phase

        def rule(x0, y0, x1, y1, col, weight=1, on=3, off=2):
            """A line if there is a line layer, a row of dashes if there is not."""
            if lines is not None:
                lines.dashed(x0, y0, x1, y1, col, weight, on, off)
                return
            mask = (ink.LEFT | ink.RIGHT if abs(x1 - x0) >= abs(y1 - y0)
                    else ink.UP | ink.DOWN)
            g = ink.box_char(mask, weight)
            n = int(max(abs(x1 - x0), abs(y1 - y0)))
            for k in range(n + 1):
                if k % max(1, on + off) >= on:
                    continue
                u = k / max(1, n)
                cv.add(int(round(x0 + (x1 - x0) * u)),
                       int(round(y0 + (y1 - y0) * u)), g, col)

        if mode == "rain":
            # The stream does not stop when the title arrives -- it runs from
            # the first frame to the end of the title card, thinning a little
            # while the log is printing and filling out afterwards.
            if c.act == "boot":
                gain = 0.45
            elif c.act == "title":
                gain = 0.85
            else:
                gain = 1.0
            for d in self.drops:
                x, y, sp, ln = d
                d[1] += sp * c.dt * (0.5 + c.rms)
                if d[1] - ln > h:
                    d[1] = -self.rng.uniform(0, h)
                    d[3] = self.rng.randint(4, 14)
                head = int(d[1])
                for k in range(ln):
                    yy = head - k
                    if 0 <= yy < h:
                        g = self.rng.choice(_RAIN_GLYPHS) if k else "\u2588"
                        col = mix(base, accent, max(0.06, 0.85 - k * 0.06) * gain)
                        if k or self.rng.random() < gain + 0.3:
                            cv.add(x, yy, g, col)

        elif mode == "stars":
            # slow parallax drift: the void moves, the stars barely do.  The
            # marks come from a ramp and boil, so the field is never twice the
            # same picture -- a static starfield over a moving song reads dead.
            for i, (fx, fy, sp) in enumerate(self._field):
                x = int((fx * w + ph * sp * 0.6) % w)
                y = int((fy * h + ph * sp * 0.25) % h)
                cv.add(x, y,
                       ink.ramp_char(ink.ASH, 0.25 + 0.6 * sp, t=ph,
                                     key=x * 131 + y, boil=0.55),
                       mix(base, accent, 0.16 + 0.26 * sp))
            if c.act == "title":
                for _ in range(w * h // 160):
                    x, y = int(self.rng.random() * w), int(self.rng.random() * h)
                    cv.add(x, y,
                           ink.ramp_char(ink.ASH, self.rng.random(), t=ph,
                                         key=x * 977 + y, boil=0.7),
                           mix(base, accent, 0.18 + self.rng.random() * 0.24))

        elif mode == "draft":
            # A drafting table.  The grid is drawn as rules so the crossings
            # are real crossings, and the two axes the shapes sit on carry
            # double weight -- at one dot per cell an axis is indistinguishable
            # from the grid it is supposed to be the axis of.
            step = 6 if w > 70 else 4
            off = int(ph * 1.6) % step
            grid = mix(base, accent, 0.17 + 0.16 * c.bass)
            for x in range(off, w, step):
                rule(x, 0, x, h - 1, grid, 1, 2, 3)
            for y in range(1, h, 3):
                rule(0, y, w - 1, y, mix(base, accent, 0.13 + 0.14 * c.bass), 1, 3, 2)
            axis = mix(base, accent, 0.46 + 0.26 * c.bass)
            rule(0, h // 2, w - 1, h // 2, axis, 2, 5, 1)
            rule(w // 2, 0, w // 2, h - 1, axis, 2, 4, 2)

        elif mode == "flow":
            # current: horizontal field lines sliding at the speed of the music
            # A rule on every row is a mesh, not a current -- it covered the
            # whole frame and the subject had nothing left to be drawn in.
            # Every third row, in dashes, keeps the line and gives the cells
            # back to the picture.
            for row in range(0, h, 3):
                speed = 6.0 + 22.0 * (c.mid + 0.2) * (1.0 + (row % 3) * 0.35)
                off = int(ph * speed) % 8
                col = mix(base, accent, 0.20 + 0.22 * c.mid)
                # every other rule is heavy: a current drawn at one weight
                # reads as graph paper rather than as water
                wt = 2 if row % 6 == 3 else 1
                if lines is not None:
                    lines.dashed(off, row, w - 1, row, col, wt, 2, 6)
                else:
                    for x in range(off, w, 4):
                        cv.add(x, row, "\u2500", col)

        elif mode in ("union", "hollow"):
            # both choruses are radial, but they mean opposite things
            cx, cy = w / 2, h / 2
            inward = mode == "union"
            count = 20 if inward else 12
            reach = max(w, h * 2) * (0.5 + 0.5 * c.bass)
            for i in range(count):
                if not inward and i % 3 == 1:
                    continue                     # gaps: nothing answers
                a = (2 * math.pi * i / count + 0.37
                     + ph * (0.22 if inward else -0.12))
                # the eight principal directions are drawn as rules; the rest
                # stay dust, so the radial keeps reading as a field
                if lines is not None and i % 5 == 0:
                    lines.dashed(cx, cy, cx + math.cos(a) * reach,
                                 cy + math.sin(a) * reach * 0.5,
                                 mix(base, accent, 0.22 if inward else 0.14),
                                 1, 3, 3)
                for k in range(0, int(reach), 2):
                    # union pulls the dust in; hollow pushes it away from you
                    s_ = (reach - k) if inward else k
                    x = cx + math.cos(a) * s_
                    y = cy + math.sin(a) * s_ * 0.5
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    t_ = k / max(1, reach)
                    glow = 1.0 - t_
                    xi, yi = int(x), int(y)
                    cv.add(xi, yi,
                           ink.ramp_char(ink.ASH, glow, t=ph,
                                         key=xi * 61 + yi, boil=0.4),
                           mix(base, accent, (0.12 + 0.24 * glow)
                               * (1.0 if inward else 0.6)))

        elif mode == "dither":
            # A hatch, not a dotted field.  `░` is not in the terminal's font; `▒`
            # and the fill ramps are, and they give a real density change
            # instead of one shade repeated at a fixed spacing.
            for y in range(1, h, 4):
                v = math.sin(ph * 0.9 + y * 0.35) * 0.5 + 0.5
                step = 7 + int(4 * v)
                for x in range(0, w, step):
                    cv.add(x, y,
                           ink.ramp_char(ink.LFILL, 0.25 + 0.7 * v, t=ph,
                                         key=x * 7 + y * 131, boil=0.30),
                           mix(base, accent, 0.18 + 0.20 * v))

        elif mode == "spokes":
            # the day dial turning overhead
            cx, cy = w / 2, h / 2
            for i in range(12):
                a = 2 * math.pi * i / 12 + ph * 0.28
                if lines is not None:
                    lines.dashed(cx, cy, cx + math.cos(a) * max(w, h * 2) * 0.7,
                                 cy + math.sin(a) * max(w, h * 2) * 0.35,
                                 mix(base, accent, 0.16 + 0.16 * c.rms),
                                 1 if i % 3 else 2, 3, 2)
                    continue
                for k in range(6, int(max(w, h * 2) * 0.7), 3):
                    x = cx + math.cos(a) * k
                    y = cy + math.sin(a) * k * 0.5
                    if 0 <= x < w and 0 <= y < h:
                        cv.add(int(x), int(y), "\u00b7",
                               mix(base, accent, 0.05 + 0.12 * c.rms))

        elif mode == "falling":
            # things leaving, and not coming back
            for (fx, fy, sp) in self._field:
                y = (fy * h + ph * sp * (1.6 + 2.0 * c.rms)) % h
                x = (fx * w + ph * sp * 0.5) % w
                xi, yi = int(x), int(y)
                cv.add(xi, yi,
                       ink.ramp_char(ink.ASH, 0.2 + 0.5 * sp, t=ph,
                                     key=xi * 17 + yi, boil=0.5),
                       mix(base, accent, 0.14 + 0.22 * sp))

        elif mode == "noise":
            # corrupted data: blocks that flicker in and out of existence
            rng = random.Random(int(c.t * 12))
            for _ in range(int(6 + 40 * (0.3 + c.rms))):
                bx = rng.randrange(0, w)
                by = rng.randrange(0, h)
                ln = rng.randint(1, 5)
                col = mix(base, accent, 0.18 + 0.32 * rng.random())
                g = rng.choice(ink.SEED + ink.FILL)   # never the type block
                for k in range(ln):
                    cv.add((bx + k) % w, by, g, col)

        elif mode == "curtain":
            # the frame the blade falls through.  U+254E is not in the
            # terminal's font -- this used to fall through to another family
            # for every stem in the curtain.  The triple-dash stem U+2506 is
            # its own glyph in WQY, and heavier.
            col = mix(base, accent, 0.24 + 0.26 * c.bass)
            step = 7 if w > 60 else 5
            for x in range(0, w, step):
                drop = int((math.sin(ph * 0.8 + x * 0.3) * 0.5 + 0.5) * h)
                if lines is not None:
                    lines.seg(x, drop, x, h - 1, col, 1 if x % 2 else 2)
                else:
                    for y in range(drop, h):
                        cv.add(x, y, "\u2506", col)

        elif mode == "pulse":
            # a heartbeat behind the heart
            cx, cy = w / 2, h / 2
            for i in range(3):
                u = ((ph * 0.30) + i / 3.0) % 1.0
                rr = u * max(w, h * 2) * 0.55
                if lines is not None and i == 0:
                    lines.circle(cx, cy, rr,
                                 mix(base, accent, 0.26 + 0.16 * (1 - u)), 1)
                    continue
                for k in range(0, int(rr), 2):
                    a0 = 2 * math.pi * k / max(1, rr)
                    x = cx + math.cos(a0) * rr
                    y = cy + math.sin(a0) * rr * 0.5
                    if 0 <= x < w and 0 <= y < h:
                        xi, yi = int(x), int(y)
                        cv.add(xi, yi,
                               ink.ramp_char(ink.ASH, 1 - u, t=ph,
                                             key=xi * 29 + yi, boil=0.45),
                               mix(base, accent, 0.12 + 0.22 * (1 - u)))

        elif mode == "collapse":
            # everything folding towards the centre line
            k = max(0.0, 1.0 - c.act_t / 2.0)
            for row in range(h):
                d = abs(row - h / 2) / max(1.0, h / 2)
                y = int(h / 2 + (row - h / 2) * k)
                if 0 <= y < h:
                    rule(0, y, w - 1, y, mix(base, accent, 0.18 + 0.30 * d),
                         2 if d < 0.4 else 1, 4, 1)


# Each act moves its dust differently.  One shared swirl everywhere was the
# second-cheapest repetition in the piece.
_PARTICLE_MODE = {
    "chorus": "converge", "love": "converge",
    "loss": "diverge", "end": "diverge",
    "transform": "orbit", "execution": "orbit",
    "argument": "scatter", "void": "drift",
    "geometry": "orbit", "current": "drift",
    "menagerie": "converge",
}


class Particles:
    """Braille dust.  How it moves is part of what the act is saying."""

    def __init__(self, n: int = 190, seed: int = 7):
        self.rng = random.Random(seed)
        self.pts = [[self.rng.uniform(-1, 1), self.rng.uniform(-1, 1),
                     self.rng.uniform(0.05, 0.4), self.rng.uniform(0, 6.28)]
                    for _ in range(n)]
        self._mode = "converge"
        self._blend = 1.0

    def draw(self, cv: Canvas, c: Ctx, bl: Braille, cx: float, cy: float,
             r: float) -> None:
        accent, hot = c.pal["accent"], c.pal["hot"]
        mode = _PARTICLE_MODE.get(c.act, "scatter")
        if mode != self._mode:
            self._mode = mode
            self._blend = 0.0
        self._blend = min(1.0, self._blend + c.dt * 0.8)
        k_blend = self._blend

        for p in self.pts:
            p[3] += c.dt * p[2] * 2.2
            rr = math.hypot(p[0], p[1]) or 1e-4
            ux, uy = p[0] / rr, p[1] / rr
            target = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(p[3]))
            pull = 0.45 + 1.6 * c.bass

            if mode == "converge":
                k = (target - rr) * pull * c.dt
                p[0] += ux * k
                p[1] += uy * k
            elif mode == "diverge":
                k = (0.35 + 1.5 * (c.rms + 0.2)) * c.dt
                p[0] += ux * k
                p[1] += uy * k
                if rr > 1.5:
                    p[0] *= 0.55
                    p[1] *= 0.55
            elif mode == "orbit":
                tang = (0.9 + 1.4 * c.mid) * c.dt
                p[0] += -uy * tang + ux * (target - rr) * 0.5 * c.dt
                p[1] += ux * tang + uy * (target - rr) * 0.5 * c.dt
            elif mode == "drift":
                p[1] += (0.12 + 0.5 * c.rms) * c.dt
                p[0] += math.sin(p[3] * 0.5) * 0.06 * c.dt
                if p[1] > 1.3:
                    p[1] = -1.3
            else:                                   # scatter
                p[0] += ux * c.rms * 0.05 * c.dt * 10
                p[1] += uy * c.rms * 0.05 * c.dt * 10
                p[0] += math.sin(p[3] * 1.7) * 0.05 * c.dt
                p[1] += math.cos(p[3] * 1.3) * 0.05 * c.dt
                if rr > 1.4:
                    p[0] *= 0.9
                    p[1] *= 0.9

            if rr > 1.6:
                p[0] *= 0.94
                p[1] *= 0.94
            bl.plot(cx + p[0] * r, cy + p[1] * r,
                    mix(accent, hot, min(1.0, abs(p[0]) * 1.1 + c.rms * 0.4)))


# --------------------------------------------------------------------------
# kinetic type
# --------------------------------------------------------------------------

REVEAL_SECONDS = 0.15
GHOST_DECIMATION = 4        # keep one dot in four for the phosphor trail

# How much of the width display type is allowed to take.  The big words used
# to run the full frame, which buried the imagery underneath them; they now
# sit inside the picture as one element among several.
TYPE_WIDTH_FRACTION = 0.52
TITLE_WIDTH_FRACTION = 0.78


def shot_dur(shot) -> float:
    return max(0.3, shot.t1 - shot.t0)


def _ramp01(t: float, a: float, b: float) -> float:
    """0 at `a`, 1 at `b`, smoothstep between."""
    if b <= a:
        return 1.0
    u = (t - a) / (b - a)
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u * u * (3 - 2 * u)


class KineticType:
    """Big bitmap type, revealed left-to-right, jittered by transients."""

    def __init__(self):
        self.shown = ""
        self.t0 = 0.0

    @staticmethod
    def _wrap(text: str, max_w: int, sc: int, limit: int = 3):
        words = text.split()
        if not words:
            return None
        rows, cur = [], ""
        for wd in words:
            if font.measure(wd, sc)[0] > max_w:
                return None
            trial = f"{cur} {wd}".strip()
            if not cur or font.measure(trial, sc)[0] <= max_w:
                cur = trial
            else:
                rows.append(cur)
                cur = wd
                if len(rows) >= limit:
                    return None
        if cur:
            rows.append(cur)
        return rows if len(rows) <= limit else None

    def _plan(self, text: str, bw: int, bh: int, scale_hint: int):
        """Largest scale whose wrapped lines fit; nothing is cut in half."""
        top = max(1, min(scale_hint, bh // font.GLYPH_H))
        for sc in range(top, 0, -1):
            best = None
            for limit in (1, 2, 3):
                rows = self._wrap(text, bw, sc, limit)
                if rows is None:
                    continue
                total = len(rows) * font.GLYPH_H * sc + (len(rows) - 1) * sc
                if total <= bh and (best is None or len(rows) < len(best)):
                    best = rows
            if best:
                return best, sc
        return [font.condense(text, max(1, bw))], 1

    def draw(self, cv: Canvas, c: Ctx, text: str, *, scale_hint: int = 3,
             box: tuple[int, int, int, int] | None = None,
             max_frac: float = TYPE_WIDTH_FRACTION) -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        if text != self.shown:
            self.shown = text
            self.t0 = c.t
        x0, y0, x1, y1 = box or (0, 0, cv.w - 1, cv.h - 1)
        full = max(1, x1 - x0 + 1 - 2)
        # `bw` bounds how big the type may get; `full` is still what it is
        # centred in.  Conflating the two left the words stuck to the left
        # margin the moment the cap was introduced.
        longest = max((font.measure(w, 1)[0] for w in text.split()), default=1)
        bw = max(longest, min(full, int(full * max_frac)))
        bh = y1 - y0 + 1
        lines, sc = self._plan(text, bw, bh, scale_hint)
        if not lines:
            return False
        widths = [font.measure(ln, sc)[0] for ln in lines]
        gw = max(widths)
        line_h = font.GLYPH_H * sc + sc
        gh = len(lines) * line_h - sc
        if gw <= 0 or gh <= 0 or gw > bw or gh > bh:
            return False
        # A line held through a long instrumental would otherwise sit frozen.
        # After the print finishes it breathes, slowly and out of step with the
        # beat, so it reads as alive rather than as a still.
        age = c.t - self.t0
        breath = 0.0
        sway = 0
        if age > 1.2:
            breath = math.sin((age - 1.2) * 1.5) * max(1.0, gh * 0.035)
            sway = int(round(math.sin((age - 1.2) * 0.9) * 0.9))
        jitter = int(round(math.sin(c.t * 7.3) * 0.5 + c.glitch * 2.0))
        bx = x0 + 1 + max(0, (full - gw) // 2) + jitter + sway
        by = y0 + max(0, (bh - gh) // 2) + int(round(breath))

        base, accent, hot = c.pal["base"], c.pal["accent"], c.pal["hot"]
        glow = min(0.35, 0.12 + 0.30 * c.rms)      # level, never a strobe
        reveal = min(1.0, (c.t - self.t0) / REVEAL_SECONDS)
        span = max(1, sum(widths) + 4 * max(0, len(widths) - 1))
        clip = span * reveal
        gap = font.gap_cells(sc)
        advance = font.GLYPH_W * sc + gap
        travelled = 0.0

        for li, ln in enumerate(lines):
            gx = bx + max(0, (gw - widths[li]) // 2)
            gy = by + li * line_h
            for gi, ch in enumerate(ln):
                glyph = font.ink_char(ch)
                cell_x = gx + gi * advance
                for ry, row in enumerate(glyph):
                    for rx, on in enumerate(row):
                        if not on:
                            continue
                        if travelled + gi * advance + rx * sc > clip:
                            continue
                        v = 1.0 - ry / max(1, len(glyph))
                        colour = mix(mix(accent, hot, v * 0.85), hot, glow)
                        for dy in range(sc):
                            yy = gy + ry * sc + dy
                            if not (y0 <= yy <= y1):
                                continue
                            for dx in range(sc):
                                xx = cell_x + rx * sc + dx
                                if x0 <= xx <= x1:
                                    cv.put(xx, yy, "█", colour)
                if sc >= 2 and c.glitch > 0.15:
                    ghost = mix(base, accent, 0.5)
                    for ry, row in enumerate(glyph):
                        for rx, on in enumerate(row):
                            if on:
                                cv.add(cell_x + rx * sc + 1, gy + ry * sc,
                                       "▌", ghost)
            travelled += widths[li] + 4
        return True


class SmallType:
    """3x5 bitmap type, anchored anywhere by fraction.

    The big words own the middle of the frame.  These are for the corners:
    a twelfth of the width each rather than a fifth, so a word can sit at the
    edge of a busy shot without becoming a second subject."""

    GAP = 1

    def draw(self, cv: Canvas, c: Ctx, text: str, *,
             box: tuple[int, int, int, int],
             anchor: tuple[float, float],
             colour: int) -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        x0, y0, x1, y1 = box
        text = font.condense_tiny(text, max(4, x1 - x0 + 1), self.GAP)
        if not text:
            return False
        cols = font.ink_tiny(text)
        gw = len(text) * (font.TINY_W + self.GAP) - self.GAP
        gh = font.TINY_H
        if gw > cv.w or gh > cv.h:
            return False
        step = font.TINY_W + self.GAP
        ax = x0 + int((x1 - x0) * anchor[0])
        ay = y0 + int((y1 - y0) * anchor[1])
        ax = max(x0, min(ax, x1 - gw + 1))
        ay = max(y0, min(ay, y1 - gh + 1))
        for ry, row in enumerate(cols):
            for gx, ch in enumerate(text):
                for k in range(font.TINY_W):
                    if row[gx * font.TINY_W + k]:
                        cv.put(ax + gx * step + k, ay + ry, "█", colour)
        return True


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------

class Stage:
    def __init__(self):
        self.type = KineticType()
        self.small = SmallType()
        self.particles = Particles()
        self.backdrop = Backdrop()
        self.other = Companion()
        self._ghost = []          # phosphor persistence: last frames' dots
        self._dim = 0.0           # smoothed caption dimming
        self._waves = []          # beat shockwaves
        self._last_wave = -9.0
        self.readout = ""         # transient control feedback, not chrome
        self.readout_until = -1.0

    def draw(self, cv: Canvas, c: Ctx) -> None:
        # Two co-resident layers at different resolutions.  `lines` works at
        # one character per cell and can therefore draw real rules, corners and
        # crossings at three weights; `bl` works at eight dots per cell and
        # draws the fine figure work.  A frame built only from the second reads
        # as one texture whatever it depicts, so both run everywhere and each
        # takes the cells the other left.
        lines = ink.LineCanvas(cv.w, cv.h)
        self.backdrop.draw(cv, c, lines)
        x0, y0, x1, y1 = c.stage
        if x1 < x0 or y1 < y0:
            lines.to_canvas(cv)
            return
        w, h = x1 - x0 + 1, y1 - y0 + 1
        bl = Braille(w, h, marks_for(c), boil=BOIL.get(c.act, 0.35))
        bl.lines_into(lines, x0, y0)

        cx, cy = bl.w / 2, bl.h / 2
        # r is a *radius*: half the screen in dots, either axis, whichever is
        # tighter.  Braille dots are isotropic, so this stays a circle.
        r = min(bl.w, bl.h) * 0.46

        if c.act == "boot":
            self._boot(cv, c, x0, y0, x1, y1)
        elif c.act == "terminated":
            self._terminated(cv, c, x0, y0, x1, y1)
        elif c.act == "title":
            if c.show_text:
                self.type.draw(cv, c, TITLE_TEXT, scale_hint=3,
                               max_frac=TITLE_WIDTH_FRACTION)
        else:
            self._scene(cv, c, bl, cx, cy, r)

        # the other, everywhere: this is a story about two.  It goes on top of
        # the type, because a figure that can be hidden by a caption is not
        # carrying the story.
        self.other.update(c.t, c.dt)
        self._shockwaves(bl, cx, cy, r, c)
        # line work first: it is structure, so the dots and then the type sit
        # over it.  Both layers leave occupied cells alone, so nothing erases
        # anything -- they interleave.
        lines.to_canvas(cv)
        if c.act != "terminated":
            bl.to_canvas(cv, x0, y0, c.t)

        word, _since = text_cue_at(c.t) if c.show_text else (None, 0.0)
        # The dimming that lets a caption read is ramped, not switched.  Applied
        # instantly it snapped the brightness of the entire frame at every cue,
        # which on a terminal is a full-screen flash rather than a fade.
        want = 0.62 if (word and c.act != "boot") else 0.0
        self._dim += (want - self._dim) * min(1.0, c.dt * 7.0)
        if word:
            self._ghost.clear()
            if self._dim > 0.02:
                self._dim_subcell(cv, x0, y0, x1, y1, c.pal["base"], self._dim)
            if not self.type.draw(cv, c, word, scale_hint=3):
                self._plain_line(cv, c, word, x0, y0, x1, y1)
        elif self._dim > 0.02:
            self._dim_subcell(cv, x0, y0, x1, y1, c.pal["base"], self._dim)

        if c.act != "terminated":
            # The instrument layer: bars and the other one.  Drawn last and on
            # its own Braille layer so a caption cannot fade or cover it --
            # anything a subtitle can hide is not carrying the picture.
            fresh = Braille(w, h, c.glyphs)
            shot = SH.shot_at(c.t)
            if shot is not None and shot.bars:
                self._bar(fresh, c, shot, cx, cy, r)
            self.other.draw(fresh, cx, cy, r, c)
            fresh.to_canvas(cv, x0, y0)

        if c.show_text:
            self._flash_words(cv, c)

        self._readout(cv, c)

    def _readout(self, cv: Canvas, c: Ctx) -> None:
        """A transient control readout, and only that.

        The piece has no player furniture on purpose, so this is not a status
        line: it exists for about a second after the viewer presses a key that
        changes something they cannot otherwise see, and then it is gone.  It
        is how the music offset gets tuned by eye -- nudge it with [ and ],
        read the number, put it in MUSIC_OFFSET.
        """
        left = self.readout_until - c.t
        if left <= 0.0 or not self.readout:
            return
        edge = min(1.0, left / 0.35)
        col = mix(c.pal["base"], mix(c.pal["accent"], c.pal["hot"], 0.5),
                  0.9 * edge)
        self.small.draw(cv, c, self.readout, box=(0, 0, cv.w - 1, cv.h - 1),
                        anchor=(0.5, 0.93), colour=col)

    def _flash_words(self, cv: Canvas, c: Ctx) -> None:
        """One small word, at the edge, for a moment."""
        text, since, zone = flash_cue_at(c.t)
        if not text:
            return
        fx, fy = FLASH_ZONES[zone]
        # fade in and out so it arrives rather than appearing
        hold = next(h for at, _t, h in FLASH_CUES if c.t >= at and c.t <= at + h)
        edge = min(1.0, since / 0.18, max(0.0, (hold - since) / 0.30))
        col = mix(c.pal["base"], mix(c.pal["accent"], c.pal["hot"], 0.55), edge)
        self.small.draw(cv, c, text, box=(0, 0, cv.w - 1, cv.h - 1),
                        anchor=(fx, fy), colour=col)

    # -- the body of the film --------------------------------------------

    def _scene(self, cv, c, bl, cx, cy, r) -> None:
        """One shot: a motif behind, the act's subject in front.

        Every ~2 lyric lines a new shot begins, which changes the motif, the
        camera and the accent -- so a 30-second act is a sequence of pictures
        rather than one picture held too long."""
        shot = SH.shot_at(c.t)
        since = c.t - shot.t0 if shot else 0.0
        phase = (shot.phase0 if shot else 0.0) + c.t * 0.40
        if shot is not None:
            # mirror is a story-neutral flip; the tilt is tiny on purpose, so
            # it reads as a new camera without turning the subject upside down
            bl.camera(rot=shot.rot, mirror=shot.mirror)

        # --- motif layer: a light dotted frame around the subject --------
        # Bigger than the subject and much dimmer, so it reads as the room the
        # shot is happening in rather than as a second subject competing for
        # attention.
        if shot is not None:
            contours = M.build(shot.motif, phase, shot.index)
            if contours:
                mz = (1.22 + 0.10 * c.bass
                      + 0.10 * _ramp01(since, 0.0, shot_dur(shot)))
                col = mix(c.pal["base"], c.pal["accent"], 0.10 + 0.16 * c.rms)
                # The frame around the subject is the one part of the shot
                # that is genuinely a line, so it is drawn as one -- a rule
                # that can carry weight and corners -- and only then
                # under-drawn with dots for texture.  Drawn in dots alone it
                # was the same mark as the subject, one size larger.
                for pts in contours:
                    bl.rule_poly([(cx + x * r * mz, cy + y * r * mz)
                                  for (x, y) in pts],
                                 mix(col, c.pal["accent"], 0.45), 1, True)
                F.contours(bl, contours, cx, cy, r * mz, col, step=2.0)

        # --- the act's own subject ---------------------------------------
        if c.act == "chorus":
            self._chorus(cv, c, bl, cx, cy, r * (shot.zoom if shot else 1.0))
            self.particles.draw(cv, c, bl, cx, cy, r)
        else:
            fig = F.FIGURE_FOR_ACT.get(c.act)
            if fig is not None:
                fig(cv, bl, c,
                    cx + (shot.pan_x if shot else 0.0) * r,
                    cy + (shot.pan_y if shot else 0.0) * r,
                    r * (shot.zoom if shot else 1.0))
            if c.act in ("execution", "argument", "love", "end", "loss",
                         "geometry", "current", "transform", "menagerie"):
                self.particles.draw(cv, c, bl, cx, cy, r)

        # --- things that run past the edge of the frame -------------------
        self._overscan(bl, c, shot, cx, cy, r)

        # --- the shot's accent -------------------------------------------
        if shot is not None and shot.accent != "none":
            self._accent(bl, c, shot, cx, cy, r, since)

    def _bar(self, bl, c: Ctx, shot, cx, cy, r) -> None:
        """Draw this shot's progress elements.

        Each bar reads something real -- the act, the phrase, the bar of
        music, the level, the distance to the other one, or the approach of
        the next sung line.  Never "how far through the file we are": a global
        playhead would be player chrome wearing a costume."""
        for spec in shot.bars:
            frac = self._bar_frac(c, shot, spec)
            F.progress_bar(
                bl, cx, cy + spec.y * r, bl.w * 0.44 * spec.length,
                spec.angle, frac,
                mix(c.pal["base"], c.pal["accent"], 0.48),
                mix(c.pal["accent"], c.pal["hot"], 0.55),
                c.pal["hot"],
                mix(c.pal["base"], c.pal["accent"], 0.62),
                thickness=max(0.8, r * 0.008 * spec.thickness),
                framed=spec.framed)

    def _bar_frac(self, c: Ctx, shot, spec) -> float:
        sem = spec.semantic
        if sem == "section":
            a, b = act_span(c.t)
            return (c.t - a) / max(1e-3, b - a)
        if sem == "phrase":
            return (c.t - shot.t0) / max(1e-3, shot.t1 - shot.t0)
        if sem == "beat":
            # position inside a four-beat bar: sweeps, then snaps back
            beats = (c.t - SH.BEAT_PHASE) / SH.BEAT
            return (beats % 4.0) / 4.0
        if sem == "energy":
            # a meter, not a clock: rises instantly, falls slowly
            lvl = getattr(self, "_level", 0.0)
            lvl = max(c.rms * 1.1, lvl - c.dt * 0.55)
            self._level = lvl
            return lvl
        if sem == "story":
            x, y, _a = _other_position(c.t)
            return min(1.0, math.hypot(x, y) / 1.6)
        if sem == "cue":
            prev = SH.prev_cue_before(c.t)
            nxt = SH.next_cue_after(c.t)
            if prev is None or nxt is None or nxt <= prev:
                return 0.0
            return (c.t - prev) / (nxt - prev)
        return 0.0

    @staticmethod
    def _overscan(bl, c, shot, cx, cy, r) -> None:
        """Elements that leave the frame, so the picture reads as a window.

        Two kinds.  A second copy of the motif, scaled well past the frame so
        only a fragment of it is inside.  And sweepers: long lines that enter
        from beyond one edge and leave past the other, drifting through at
        their own pace.  Both are deliberately larger than the screen -- a
        composition where everything fits inside the border reads as a diagram,
        not as a place.
        """
        if shot is None:
            return
        rng = random.Random(7717 + shot.index * 15485863)

        # oversized motif fragment
        if rng.random() < 0.45:
            contours = M.build(shot.motif, shot.phase0 - c.t * 0.16, shot.index + 1)
            if contours:
                off = rng.uniform(0.45, 0.95) * r
                ang = rng.uniform(0, 6.283)
                F.contours(bl, contours,
                           cx + math.cos(ang) * off, cy + math.sin(ang) * off,
                           r * rng.uniform(1.7, 2.4),
                           mix(c.pal["base"], c.pal["accent"], 0.08 + 0.10 * c.rms),
                           step=2.2)

        # sweepers
        for k in range(rng.randint(1, 3)):
            ang = rng.uniform(0, math.pi)
            speed = rng.uniform(0.10, 0.26)
            phase = rng.random()
            u = ((c.t * speed + phase) % 1.0) * 2.0 - 1.0     # -1 .. 1
            off = u * r * rng.uniform(1.2, 1.8)
            px, py = -math.sin(ang), math.cos(ang)            # perpendicular
            ox, oy = cx + px * off, cy + py * off
            reach = r * 2.6
            ax, ay = (ox - math.cos(ang) * reach, oy - math.sin(ang) * reach)
            bx, by = (ox + math.cos(ang) * reach, oy + math.sin(ang) * reach)
            col = mix(c.pal["base"], c.pal["accent"], 0.22 + 0.20 * c.rms)
            # a sweeper is the longest line in the frame; at one dot per cell
            # it was also the faintest thing in it
            bl.rule(ax, ay, bx, by, col, 2 if k == 0 and c.rms > 0.35 else 1)
            bl.line(ax, ay, bx, by, mix(col, c.pal["base"], 0.45))

    @staticmethod
    def _accent(bl, c, shot, cx, cy, r, since) -> None:
        u = min(1.0, since / max(0.3, shot_dur(shot)))
        col = mix(c.pal["base"], mix(c.pal["accent"], c.pal["hot"], 0.5), 0.55)
        if shot.accent == "ring":
            F.stroke(bl, F.sh_circle(), cx, cy, r * (0.35 + 0.65 * u),
                     col, closed=True)
        elif shot.accent == "burst":
            for i in range(9):
                a = 2 * math.pi * i / 9 + shot.phase0
                ax, ay = cx + math.cos(a) * r * 0.9, cy + math.sin(a) * r * 0.9
                bx = cx + math.cos(a) * r * (1.05 + 0.30 * c.rms)
                by = cy + math.sin(a) * r * (1.05 + 0.30 * c.rms)
                bl.rule(ax, ay, bx, by, col, 2 if i % 3 == 0 else 1)
                bl.line(ax, ay, bx, by, col)
        elif shot.accent == "sweep":
            y = cy - r + 2 * r * (since * 0.9 % 1.0)
            half = math.sqrt(max(0.0, (r * 0.98) ** 2 - (y - cy) ** 2))
            bl.rule(cx - half, y, cx + half, y, col, 2)
            bl.line(cx - half, y, cx + half, y, col)
        elif shot.accent == "grid":
            step = max(4.0, r / 5)
            for k in range(-5, 6):
                ax, bx2 = cx + k * step, cy - r * 0.85
                by2 = cy + r * 0.85
                gcol = mix(col, c.pal["base"], 0.50)
                bl.rule(ax, bx2, ax, by2, gcol, 2 if k == 0 else 1)
                bl.line(ax, bx2, ax, by2, gcol)
        elif shot.accent == "dust":
            rng = random.Random(shot.index * 31 + int(since * 3))
            for _ in range(26):
                a = rng.random() * 2 * math.pi
                rad = r * (0.25 + rng.random() * 0.8)
                bl.plot(cx + math.cos(a) * rad, cy + math.sin(a) * rad, col)

    def _shockwaves(self, bl, cx, cy, r, c) -> None:
        """A ring, but rarely and slowly.

        This used to fire on any strong transient, which put a pulse on screen
        every half second and made the whole frame feel beat-locked.  Now it
        answers only the biggest moments and takes four times as long to
        cross, so it reads as an event rather than as a tempo."""
        if c.onset > 0.93 and c.t - self._last_wave > 2.4:
            self._waves.append(c.t)
            self._last_wave = c.t
        self._waves = [t0 for t0 in self._waves if c.t - t0 < 2.2]
        for t0 in self._waves:
            u = (c.t - t0) / 2.2
            rr = r * (0.25 + 1.05 * u)
            F.stroke(bl, F.sh_circle(), cx, cy, rr,
                     mix(c.pal["accent"], c.pal["base"], 0.25 + 0.7 * u),
                     closed=True)

    # -- whole-screen effects --------------------------------------------

    def _chorus(self, cv: Canvas, c: Ctx, bl: Braille, cx, cy, r) -> None:
        """The two choruses look alike and mean opposite things.

        The first is a meeting: rings thrown out from a core that has somebody
        in it.  The second is the same gesture aimed at an empty room -- fewer
        rings, arriving with gaps, fading before they land."""
        # The chorus runs fifteen seconds, so it is built as four beats rather
        # than one gesture: stimulation, satisfaction, execution, simulation.
        kind, since = F.phase(
            c.t, F.PHASES["chorus1" if c.t < 120.0 else "chorus2"])

        if c.t < 120.0 and kind == "satisfy":
            # rings settle into a slow orbit instead of expanding
            for i in range(3):
                F.stroke(bl, F.sh_circle(), cx, cy,
                       r * (0.42 + 0.22 * i) * (1.0 + 0.06 * c.bass),
                       mix(c.pal["accent"], c.pal["hot"], 0.5 - i * 0.14),
                       closed=True)
            core = r * (0.10 + 0.26 * c.bass)
            bl.circle(cx, cy, core, c.pal["hot"])
            return
        if c.t < 120.0 and kind == "execute":
            # rings become blades: struck through rather than thrown out
            for i in range(5):
                a = math.pi * (i / 5.0) + c.t * 0.06
                bl.line(cx - math.cos(a) * r, cy - math.sin(a) * r,
                        cx + math.cos(a) * r, cy + math.sin(a) * r,
                        mix(c.pal["accent"], c.pal["hot"], 0.35 + 0.4 * c.rms))
            core = r * (0.08 + 0.20 * c.bass)
            bl.circle(cx, cy, core, c.pal["hot"])
            return
        if c.t < 120.0 and kind == "trap":
            # and here the rings close inwards
            for i in range(4):
                u = ((c.act_t * 0.30) + i / 4.0) % 1.0
                F.stroke(bl, F.sh_circle(), cx, cy, (1.0 - u) * r * 1.1,
                       mix(c.pal["accent"], c.pal["hot"], u), closed=True)
            return
        if c.t >= 120.0 and kind == "trapped":
            for i in range(3):
                u = ((c.act_t * 0.22) + i / 3.0) % 1.0
                F.stroke(bl, F.sh_circle(), cx, cy, (1.0 - u) * r * 0.8,
                       mix(c.pal["accent"], c.pal["base"], 0.3 + 0.6 * u),
                       closed=True)
            return

        if c.t < 120.0:
            for i in range(6):
                u = ((c.act_t * 0.42) + i / 6.0) % 1.0
                rr = u * r * (1.0 + 0.35 * c.bass)
                F.stroke(bl, F.sh_circle(), cx, cy, rr,
                         mix(c.pal["accent"], c.pal["hot"], 1.0 - u),
                         closed=True)
            core = r * (0.10 + 0.26 * c.bass)
            bl.circle(cx, cy, core, c.pal["hot"])
            for i in range(12):
                a = 2 * math.pi * i / 12 + c.t * 0.35
                bl.line(cx + math.cos(a) * core, cy + math.sin(a) * core,
                        cx + math.cos(a) * core * (1.5 + 0.6 * c.rms),
                        cy + math.sin(a) * core * (1.5 + 0.6 * c.rms),
                        mix(c.pal["accent"], c.pal["hot"], 0.5))
        else:
            for i in range(4):
                if i % 2:
                    continue                # half the rings never arrive
                u = ((c.act_t * 0.34) + i / 4.0) % 1.0
                rr = u * r * (1.15 + 0.25 * c.bass)
                pts = F.sh_circle()
                F.stroke(bl, pts[::3] + pts[:1], cx, cy, rr,
                         mix(c.pal["accent"], c.pal["base"], 0.35 + 0.65 * u),
                         closed=False)
            core = r * (0.05 + 0.12 * c.bass)
            bl.circle(cx, cy, core, mix(c.pal["accent"], c.pal["hot"], 0.4))
            reach = min(1.0, c.rms * 1.3)   # it reaches for something
            for i in range(6):
                a = 2 * math.pi * i / 6 + c.t * 0.5
                bl.line(cx + math.cos(a) * core * 1.2,
                        cy + math.sin(a) * core * 1.2,
                        cx + math.cos(a) * r * (0.25 + 0.35 * reach),
                        cy + math.sin(a) * core * (0.25 + 0.35 * reach),
                        mix(c.pal["hot"], c.pal["base"], 0.5))

    # -- special screens --------------------------------------------------

    def _boot(self, cv: Canvas, c: Ctx, x0, y0, x1, y1) -> None:
        """The machine coming up, one whole line at a time.

        Not typed out character by character: a real boot prints a line and
        moves on, and the inline bars those tools draw only make sense once
        the line is complete.  The log scrolls, so any window height shows the
        end of it.
        """
        accent, hot, base = c.pal["accent"], c.pal["hot"], c.pal["base"]
        rows = max(1, (y1 - y0 + 1) - (3 if (y1 - y0) >= 6 else 0))
        per = max(0.16, 15.0 / max(1, len(BOOT_LINES)))
        started = [(i, kind, text, bar, 0.55 + i * per)
                   for i, (kind, text, bar) in enumerate(BOOT_LINES)
                   if c.t >= 0.55 + i * per]
        view = started[-rows:] if len(started) > rows else started
        for k, (i, kind, text, bar, t0) in enumerate(view):
            y = y0 + k
            if kind == "kernel":
                pre, col = f"[{t0:12.6f}] ", mix(base, accent, 0.80)
            elif kind == "ok":
                pre, col = "[  OK  ] ", mix(base, accent, 0.95)
            elif kind == "warn":
                pre, col = "[ WARN ] ", mix(base, c.pal["hot"], 0.70)
            elif kind == "pkg":
                pre, col = "", hot
            else:
                pre, col = "", mix(base, accent, 0.90)
            body = pre + text
            if bar:
                frac = min(1.0, (c.t - t0) / 0.55)
                body += "  " + _inline_bar(bar, frac)
            avail = max(4, (x1 - x0 + 1) - 3)
            cv.text(x0 + 2, y, body[:avail], col)
            # the newest line gets a moment of full brightness
            age = c.t - t0
            if k == len(view) - 1 and age < 0.12:
                cv.text(x0 + 2, y, body[:avail], c.pal["hot"])
        if y1 - y0 >= 5:
            self._trace(cv, c, x0 + 1, max(y0, y1 - 2), x1 - 1, y1)
        if len(started) >= len(BOOT_LINES) and (y1 - y0) >= 8:
            cv.text(x0 + 2, y1 - 3,
                    "> " + ("█" if int(c.t * 3) % 2 == 0 else " "), hot)

    def _trace(self, cv: Canvas, c: Ctx, x0, y0, x1, y1) -> None:
        wave, rate = c.wave, c.wave_rate
        if wave is None or len(wave) == 0 or x1 <= x0 or y1 < y0:
            return
        cols = x1 - x0 + 1
        bl = Braille(cols, y1 - y0 + 1, c.glyphs)
        mid = (y1 - y0 + 1) * 2
        span = 0.06
        accent, hot, base = c.pal["accent"], c.pal["hot"], c.pal["base"]
        for i in range(cols * 2):
            t = (c.t - span) + span * (i / max(1, cols * 2 - 1))
            j = int(t * rate)
            if j < 0 or j >= len(wave):
                continue
            mn, mx = float(wave[j][0]) / 127.0, float(wave[j][1]) / 127.0
            v = mx if abs(mx) > abs(mn) else mn
            bl.plot(i, mid - v * mid * 0.9,
                    mix(accent, hot, min(1.0, abs(mx - mn) * 1.5)))
        for i in range(0, cols * 2, 4):
            bl.plot(i, mid, mix(base, accent, 0.30))
        bl.to_canvas(cv, x0, y0)

    def _terminated(self, cv: Canvas, c: Ctx, x0, y0, x1, y1) -> None:
        accent, hot, base = c.pal["accent"], c.pal["hot"], c.pal["base"]
        mid = y0 + (y1 - y0) // 2
        cv.text_center(mid - 1, "world.execute(me);", mix(base, accent, 0.55))
        cv.text_center(mid, "[ process terminated ]", hot)
        if (y1 - y0) > 6:
            cv.text_center(mid + 2, "exit status 0", mix(base, accent, 0.6))
        if int(c.t * 2) % 2 == 0:
            cv.text_center(mid + 3, "█", accent)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _plain_line(cv: Canvas, c: Ctx, text: str, x0, y0, x1, y1) -> None:
        """Small terminals: the terminal's own font, wrapped, never cut."""
        width = max(4, x1 - x0 + 1 - 2)
        words = text.split()
        rows, cur = [], ""
        for wd in words:
            trial = f"{cur} {wd}".strip()
            if not cur or len(trial) <= width:
                cur = trial
            else:
                rows.append(cur)
                cur = wd
        if cur:
            rows.append(cur)
        rows = rows[:3]
        top = y0 + max(0, ((y1 - y0 + 1) - len(rows)) // 2)
        for i, ln in enumerate(rows):
            y = top + i
            if y0 <= y <= y1:
                cv.text_center(y, truncate(ln, width), c.pal["hot"])

    @staticmethod
    def _dim_subcell(cv: Canvas, x0, y0, x1, y1, toward: int, k: float) -> None:
        for y in range(max(0, y0), min(cv.h, y1 + 1)):
            row = y * cv.w
            for x in range(max(0, x0), min(cv.w, x1 + 1)):
                i = row + x
                ch = cv.ch[i]
                if ch and ch != WIDE and 0x2800 <= ord(ch[0]) <= 0x28FF:
                    cv.fg[i] = mix(cv.fg[i], toward, k)


# --------------------------------------------------------------------------
# layout -- the whole terminal, at every size
# --------------------------------------------------------------------------

@dataclass
class Layout:
    stage: tuple[int, int, int, int] = (0, 0, 79, 23)
    tiny: bool = False
    width_tier: str = "normal"


def layout_for(w: int, h: int) -> Layout:
    """With no chrome to make room for, the stage is simply the terminal.

    All that is left to decide is how much detail the size can carry."""
    tier = ("tiny" if (w < 46 or h < 12) else
            "compact" if (w < 72 or h < 18) else
            "wide" if (w >= 120 and h >= 34) else "normal")
    return Layout(stage=(0, 0, w - 1, h - 1), tiny=(tier == "tiny"),
                  width_tier=tier)
