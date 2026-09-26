#!/usr/bin/env python3
"""The marks the MV draws with.

Braille dots give eight addressable points per cell, which is why every figure
in this piece is built from them -- but a picture made only of dots reads as
one texture no matter what it depicts.  This module adds the other half of a
character grid's vocabulary:

  * **Line art**, in box-drawing characters.  ``LineCanvas`` remembers which
    of a cell's four sides a line arrived from and how heavy it was, and picks
    the glyph that matches -- so corners, tees and crossings come out right
    without the caller knowing they exist.
  * **Weight.**  A character grid cannot draw a thicker stroke; what it can do
    is switch to a heavier *character* for the same cell.  ``─`` and ``━`` are
    the same line at two weights, and the double set ``═`` is a third.  That
    is the only honest kind of "thicker line" here, and it is what ``weight``
    selects.
  * **Ramps** for texture, so density can be shown as shade, fill or grain
    instead of only as an accumulation of dots.

Everything named here was checked against the terminal's own font first.  The
video renders in Konsole's configured face, so a glyph that face lacks would
either fall back to a stranger's shapes or come out as a tofu box.  Of the 128
characters in the box-drawing block WenQuanYi Micro Hei Mono has 113 -- all of
the light, heavy, double and light/heavy-mixture sets, at exactly 0.6016em.
It is missing only the half-line stubs (U+2575-257B) and the dashed stems
U+254C-254F.  That last gap is why the backdrop used to reach for ``╎`` and
fall through to Noto; ``┆`` (U+2506) is present and does the same job.
"""

from __future__ import annotations

import math

from .canvas import DEFAULT, WIDE

# --------------------------------------------------------------------------
# glyph sets, light to heavy
# --------------------------------------------------------------------------

# Everything below is present in WenQuanYi Micro Hei Mono at one cell width.
#
# None of them contains the full block U+2588, and that is a rule rather than
# an accident: this MV uses the full block for type and nothing else -- the
# title card, the captions, the log's progress bars, the head of a rain column.
# A texture that also emitted U+2588 would put something on screen that reads
# as writing, and would make "how much type is on screen" unmeasurable from
# the outside.  The ramps below therefore stop one step short of solid.
GRAIN = "·:;+*x#%@"
ASH = "·˙•°○●"
FILL = "▁▂▃▄▅▆▇"             # height of fill, so it reads as a level bar
LFILL = "▏▎▍▌▋▊▉"            # width of fill, for horizontal meters

#: The one character reserved for the type layer.  See the note above.
TYPE_BLOCK = "\u2588"
BLOOM = "·+×○◇◆●■"
DIAG = "·╱╲╳"
SEED = ".;*oO0@"
ASCII = "01#$%&*/\\|<>[]{}=+;:~^"

RAMPS: dict[str, str] = {
    "grain": GRAIN,
    "ash": ASH,
    "fill": FILL,
    "lfill": LFILL,
    "bloom": BLOOM,
    "diag": DIAG,
    "seed": SEED,
    "rain": ASCII,
}


def _mix32(a: int, b: int) -> int:
    """A small deterministic hash: same inputs, same value, everywhere."""
    h = (a * 0x9E3779B1) ^ (b * 0x85EBCA6B)
    h &= 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x2545F491) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def assert_no_type_block(*ramps: str) -> None:
    """Guard: a texture must never emit the character reserved for type.

    Checked at import, so the rule is enforced by the module rather than kept
    alive by a comment.
    """
    for r in ramps:
        if TYPE_BLOCK in r:
            raise ValueError(
                f"ramp {r!r} contains the type block {TYPE_BLOCK!r}; "
                f"type would stop being readable")


def ramp_char(ramp: str, v: float, *, t: float = 0.0, key: int = 0,
              boil: float = 0.0, rate: float = 3.0) -> str:
    """Pick a character for a density ``v`` in 0..1, optionally boiling.

    ``boil`` lets a cell trade places with its neighbours over time: the level
    still comes from ``v``, so the picture keeps its shape, but neighbouring
    steps of the ramp swap back and forth and the texture moves.  Without it
    a ramp is a still image the moment the geometry stops changing.

    Each cell is hashed from its own key at the current tick, so the shifts
    are independent across the frame and the result shimmers rather than
    pulsing in blocks -- a synchronised swap would read as a strobe, which is
    the one thing this renderer is not allowed to do to a whole screen.
    ``boil`` is the chance a given cell shifts on a given tick, and ``rate``
    is how many ticks there are per second; together they set how alive the
    texture looks.
    """
    if not ramp:
        return " "
    n = len(ramp)
    i = int(round(max(0.0, min(1.0, v)) * (n - 1)))
    if boil > 0.0 and v > 0.02:
        h = _mix32(key, int(t * rate)) / 0xFFFFFFFF
        if h < boil:
            i += 1 if h < boil * 0.5 else -1
    return ramp[max(0, min(n - 1, i))]


# --------------------------------------------------------------------------
# box drawing
# --------------------------------------------------------------------------

# Sides a line can enter a cell from.
UP, RIGHT, DOWN, LEFT = 1, 2, 4, 8

# WenQuanYi has no half-line stubs (U+2575-257B), so a line that ends inside a
# cell keeps the full stem instead of a stub.  It reads as the line running to
# the cell edge, which is what a terminal would show anyway.
_FAMILIES: tuple[tuple[str, dict[int, str]], ...] = (
    ("─│┌┐└┘├┤┬┴┼", {
        1: "│", 2: "─", 3: "└", 4: "│", 5: "│", 6: "┌", 7: "├",
        8: "─", 9: "┘", 10: "─", 11: "┴", 12: "┐", 13: "┤", 14: "┬",
        15: "┼",
    }),
    ("━┃┏┓┗┛┣┫┳┻╋", {
        1: "┃", 2: "━", 3: "┗", 4: "┃", 5: "┃", 6: "┏", 7: "┣",
        8: "━", 9: "┛", 10: "━", 11: "┻", 12: "┓", 13: "┫", 14: "┳",
        15: "╋",
    }),
    ("═║╔╗╚╝╠╣╦╩╬", {
        1: "║", 2: "═", 3: "╚", 4: "║", 5: "║", 6: "╔", 7: "╠",
        8: "═", 9: "╝", 10: "═", 11: "╩", 12: "╗", 13: "╣", 14: "╦",
        15: "╬",
    }),
)

WEIGHT_NAMES = ("light", "heavy", "double")


def box_char(mask: int, weight: int = 1) -> str:
    """The box-drawing character for a set of connected sides."""
    if not mask:
        return " "
    w = max(1, min(len(_FAMILIES), int(weight))) - 1
    return _FAMILIES[w][1][mask]


def _steps(dx: float, dy: float) -> int:
    return max(1, int(math.ceil(max(abs(dx), abs(dy)) * 2.0)))


class LineCanvas:
    """Line art at cell resolution, in box-drawing characters.

    Coordinates are in cells.  A line is drawn as a run of small steps; each
    step that crosses into a new cell tells both cells which side the line
    left through, and ``to_canvas`` then turns the accumulated sides into the
    right glyph.  Two lines crossing therefore produce a crossing glyph by
    themselves -- nothing has to know a crossing happened.
    """

    __slots__ = ("w", "h", "mask", "wt", "col")

    def __init__(self, w: int, h: int):
        self.w, self.h = max(1, w), max(1, h)
        self.mask = bytearray(self.w * self.h)
        self.wt = bytearray(self.w * self.h)
        self.col: dict[int, int] = {}

    # -- drawing ---------------------------------------------------------

    def _mark(self, cx: int, cy: int, side: int, weight: int,
              colour: int) -> None:
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return
        i = cy * self.w + cx
        self.mask[i] |= side
        if weight >= self.wt[i]:
            self.wt[i] = min(255, weight)
            self.col[i] = colour

    def _walk(self, x0: float, y0: float, x1: float, y1: float,
              colour: int, weight: int) -> None:
        n = _steps(x1 - x0, y1 - y0)
        px, py = int(math.floor(x0)), int(math.floor(y0))
        for k in range(1, n + 1):
            u = k / n
            cx = int(math.floor(x0 + (x1 - x0) * u))
            cy = int(math.floor(y0 + (y1 - y0) * u))
            if (cx, cy) == (px, py):
                continue
            dx, dy = cx - px, cy - py
            # a diagonal step connects through both axes, which the grid shows
            # as a staircase of corners -- correct, and what a terminal does
            if dx > 0:
                self._mark(px, py, RIGHT, weight, colour)
                self._mark(cx, cy, LEFT, weight, colour)
            elif dx < 0:
                self._mark(px, py, LEFT, weight, colour)
                self._mark(cx, cy, RIGHT, weight, colour)
            if dy > 0:
                self._mark(px, py, DOWN, weight, colour)
                self._mark(cx, cy, UP, weight, colour)
            elif dy < 0:
                self._mark(px, py, UP, weight, colour)
                self._mark(cx, cy, DOWN, weight, colour)
            px, py = cx, cy

    def seg(self, x0: float, y0: float, x1: float, y1: float,
            colour: int = DEFAULT, weight: int = 1) -> None:
        """A straight line between two points, in cell coordinates."""
        self._walk(x0, y0, x1, y1, colour, weight)

    def dashed(self, x0: float, y0: float, x1: float, y1: float,
               colour: int = DEFAULT, weight: int = 1,
               on: int = 2, off: int = 3) -> None:
        """A rule drawn as dashes: ``on`` cells lit, then ``off`` skipped.

        Worth having because the backdrop is drawn before everything else and
        can only write into cells the figures have not taken.  A solid rule
        across the frame would eat a whole row or column of the picture; a
        dashed one keeps the line -- and its weight -- while leaving the gaps
        for the dots to fill.
        """
        n = int(round(max(abs(x1 - x0), abs(y1 - y0)))) + 1
        if n < 2:
            self.seg(x0, y0, x1, y1, colour, weight)
            return
        # A dash one cell long still has to show, so each lit cell is marked
        # directly with the axis it runs along rather than being collected
        # into a run: a run of one drew nothing at all.  Marking the axis
        # rather than the neighbours also means a dash crossing a solid rule
        # still produces a crossing glyph -- the masks simply OR together.
        mask = (LEFT | RIGHT if abs(x1 - x0) >= abs(y1 - y0)
                else UP | DOWN)
        period = max(1, on + off)
        for k in range(n):
            if k % period >= on:
                continue
            u = k / (n - 1)
            self._mark(int(round(x0 + (x1 - x0) * u)),
                       int(round(y0 + (y1 - y0) * u)), mask, weight, colour)

    def poly(self, pts, colour: int = DEFAULT, weight: int = 1,
             closed: bool = False) -> None:
        n = len(pts)
        if n < 2:
            return
        span = n if closed else n - 1
        for i in range(span):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            self._walk(x0, y0, x1, y1, colour, weight)

    def rect(self, x0: float, y0: float, x1: float, y1: float,
             colour: int = DEFAULT, weight: int = 1) -> None:
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                  colour, weight, closed=True)

    def circle(self, cx: float, cy: float, r: float, colour: int = DEFAULT,
               weight: int = 1, n: int | None = None) -> None:
        # y is compressed by the cell's aspect: a cell is about twice as tall
        # as it is wide, so a circle costs half as many rows as columns
        k = n or max(24, int(abs(r) * 5))
        pts = [(cx + math.cos(2 * math.pi * i / k) * r,
                cy + math.sin(2 * math.pi * i / k) * r * 0.5)
               for i in range(k + 1)]
        self.poly(pts, colour, weight)

    def arc(self, cx: float, cy: float, r: float, a0: float, a1: float,
            colour: int = DEFAULT, weight: int = 1, n: int | None = None) -> None:
        k = n or max(8, int(abs(r) * abs(a1 - a0) * 2))
        pts = [(cx + math.cos(a0 + (a1 - a0) * i / k) * r,
                cy + math.sin(a0 + (a1 - a0) * i / k) * r * 0.5)
               for i in range(k + 1)]
        self.poly(pts, colour, weight)

    # -- output ----------------------------------------------------------

    def to_canvas(self, cv, ox: int = 0, oy: int = 0,
                  respect: bool = True) -> None:
        """Stamp the line art onto a canvas.

        ``respect`` leaves cells that already hold something alone, so line
        work drawn first stays behind the type and the figures instead of
        punching through them.
        """
        for cy in range(self.h):
            Y = oy + cy
            if not (0 <= Y < cv.h):
                continue
            base = cy * self.w
            row = Y * cv.w
            for cx in range(self.w):
                m = self.mask[base + cx]
                if not m:
                    continue
                X = ox + cx
                if not (0 <= X < cv.w):
                    continue
                i = row + X
                if respect and cv.ch[i] not in (" ", WIDE):
                    continue
                cv.ch[i] = box_char(m, self.wt[base + cx])
                cv.fg[i] = self.col.get(base + cx, DEFAULT)


assert_no_type_block(*RAMPS.values())
