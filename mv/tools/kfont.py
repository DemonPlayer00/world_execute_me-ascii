#!/usr/bin/env python3
"""Render text the way this machine's terminal renders it.

The MV was written for a terminal, so the honest way to put it in a video is
to use the terminal's own font -- not a font that merely happens to be
monospace.  On this machine "the terminal's own font" is knowable:

  * Konsole has no profile in ``~/.local/share/konsole/``, so it falls back to
    its built-in default profile, which carries no font of its own and
    therefore inherits KDE's global font from ``~/.config/kdeglobals``::

        fixed=文泉驿等宽微米黑,10,-1,5,400,...,1,,0,0

    That is Qt's serialised font description: family, pointSize, pixelSize
    (``-1`` means "derive from the point size"), styleHint, weight, ...

  * The family is a TrueType collection.  Face 0 is "WenQuanYi Micro Hei"
    (proportional); face 1 is "WenQuanYi Micro Hei Mono".  A terminal needs
    the monospace face, so the face index is not optional.

  * Qt resolves 10pt at 96dpi to 13px, and the cell is whatever Qt reports
    there: advance 8px, height 15px.  That 8x15 cell is the unit this module
    scales by, so the video keeps Konsole's true cell shape (1.875:1, which
    is *not* the 2:1 the MV's dot geometry assumes).

  * That face covers almost everything the MV draws, but not Braille, not
    U+254E, and not the light-shade and quadrant blocks.  For those, Qt asks
    fontconfig for a substitute -- so this module walks fontconfig's ordered
    candidate list and takes the first face that is actually *usable* in a
    terminal cell.  Usable is decided by measurement, not by preference:

      - the face must cover the codepoint at all;
      - for Braille, a glyph with one dot set must have exactly one blob of
        ink.  FreeMono looks ideal on paper -- braille at exactly 0.6000em,
        a perfect fit for the cell -- but it draws the seven *unset* dots as
        hollow rings, which would turn every cell of the MV's dot graphics
        into a ring of circles.  It is rejected here, which leaves DejaVu
        Sans: wider than the cell at 0.7324em, but its dots tile a clean 2x4
        grid with only ~4.6% pitch ripple once anchored at the cell origin;
      - the glyph's ink must fit inside one cell.  This is what rejects Noto
        Sans CJK for the shade blocks -- its glyphs are a full em wide, so a
        cell would show only their left 60% -- and leaves FreeMono, whose
        shade and quadrant glyphs are exactly one cell wide and correct.

Glyphs are drawn at the cell origin and clipped to the cell, never squashed:
a terminal does not rescale a fallback glyph, and squashing DejaVu's braille
into the narrower cell would modulate the dot pitch by ~50%, which on an MV
made entirely of dots is a visible two-dot-period ripple.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KDEGLOBALS = Path.home() / ".config" / "kdeglobals"

# Qt's QFont::StyleHint enum, in declaration order -- which is the order the
# serialised form uses.  Note the aliases collapse: Helvetica/SansSerif share a
# value, as do Times/Serif, Courier/TypeWriter and OldEnglish/Decorative.  A 5
# here therefore means AnyStyle, not TypeWriter.
_STYLE_HINTS = (
    "SansSerif",   # 0  Helvetica
    "Serif",       # 1  Times
    "TypeWriter",  # 2  Courier
    "OldEnglish",  # 3  Decorative
    "System",      # 4
    "AnyStyle",    # 5
    "Cursive",     # 6
    "Monospace",   # 7
    "Fantasy",     # 8
)

BRAILLE_LO, BRAILLE_HI = 0x2800, 0x28FF
_MAX_CANDIDATES = 32          # bound the fontconfig walk


# --------------------------------------------------------------------------
# Konsole's configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedFont:
    """The terminal's configured font, as read from KDE's config."""

    family: str
    point_size: float
    pixel_size: int          # -1 when the point size governs
    style_hint: str
    weight: int

    def px(self, dpi: float = 96.0) -> int:
        if self.pixel_size > 0:
            return self.pixel_size
        return max(1, int(round(self.point_size * dpi / 72.0)))


def parse_qt_font(spec: str) -> FixedFont:
    """Parse Qt's serialised font description.

    ``family,pointSize,pixelSize,styleHint,weight,italic,underline,
    strikeOut,fixedPitch,...``.  Qt escapes commas inside a family, but no
    family on this machine needs that, so a plain split is used.
    """
    parts = [p.strip() for p in spec.split(",")]

    def num(i: int, default: int) -> int:
        try:
            return int(float(parts[i]))
        except (IndexError, ValueError):
            return default

    point = float(parts[1]) if len(parts) > 1 and parts[1] else 10.0
    hint = num(3, 0)
    return FixedFont(
        family=parts[0],
        point_size=point,
        pixel_size=num(2, -1),
        style_hint=_STYLE_HINTS[hint] if 0 <= hint < len(_STYLE_HINTS) else "AnyStyle",
        weight=num(4, 400),
    )


def read_kde_fixed(path: Path | None = None) -> FixedFont:
    """KDE's global font -- what Konsole uses when it has no profile."""
    p = path or KDEGLOBALS
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read {p}: {exc}") from exc
    in_general = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_general = line == "[General]"
            continue
        if in_general and line.startswith("fixed="):
            return parse_qt_font(line[len("fixed="):])
    raise SystemExit(f"no [General] fixed= entry in {p}")


# --------------------------------------------------------------------------
# The cell, as Qt reports it
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """Konsole's character cell: columns and rows of pixels per glyph."""

    w: int
    h: int
    ascent: float
    native_px: int
    source: str

    @property
    def aspect(self) -> float:
        return self.h / self.w

    def scaled(self, n: int) -> "Cell":
        k = max(1, int(n))
        return Cell(self.w * k, self.h * k, self.ascent * k,
                    self.native_px * k, f"{self.source} x{k}")


_QT_APP = None          # module-level: PyQt does not keep QApplication alive


def qt_cell(family: str, px: int) -> Cell | None:
    """Ask Qt for the metrics Konsole itself would use.

    Qt rounds the advance and the line height to whole pixels, and it is that
    rounding -- not the font's design -- that gives Konsole's 8x15 cell at
    13px.  Reproducing it from PIL's unrounded metrics would give 8x17 and
    quietly change every figure in the MV, so Qt is asked directly when it is
    available.

    The platform plugin has to be chosen before QtGui is imported, and the
    QApplication has to be kept in a module global: a local one is collected
    by Python's GC, and Qt then aborts the process the next time it touches
    the font database.
    """
    global _QT_APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtGui import QFont, QFontMetricsF
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return None
    if QApplication.instance() is None:
        _QT_APP = QApplication([])
    f = QFont(family)
    f.setFixedPitch(True)
    f.setStyleHint(QFont.StyleHint.TypeWriter)
    f.setPixelSize(px)
    fm = QFontMetricsF(f)
    return Cell(int(round(fm.horizontalAdvance("M"))), int(round(fm.height())),
                float(fm.ascent()), px, "Qt")


def pil_fallback_cell(font: ImageFont.FreeTypeFont, px: int) -> Cell:
    """Last resort when Qt is absent: keep the native cell's *shape*.

    PIL reports the font's typographic line height (1.1724em), which is 9%
    taller than the 1.1538em Qt uses after hinting.  Using it directly would
    stretch the cell and make the MV's round figures elliptical, so the
    advance is taken as measured and the height is scaled by the ratio Qt
    gives at the native size.
    """
    a, d = font.getmetrics()
    w = int(round(font.getlength("M")))
    h = int(round((a + d) * (15 / 17)))
    return Cell(max(1, w), max(1, h), float(a), px, "PIL")


# --------------------------------------------------------------------------
# Which face draws which character
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def fc_candidates(family: str, codepoint: int) -> tuple[tuple[str, int], ...]:
    """fontconfig's ordered answer for one character, best first.

    ``spacing=100`` asks for a fixed-pitch face, because that is what Konsole
    asks for: its font is fixed-pitch, so a substitute has to be too.  Without
    it fontconfig happily offers the *proportional* member of a CJK family,
    and Braille resolves to a face whose dots are spaced for its own 0.73em
    advance rather than for a terminal cell.
    """
    pattern = f"{family}:charset={codepoint:04X}:spacing=100"
    try:
        out = subprocess.run(
            ["fc-match", "-s", "-f", "%{file}\t%{index}\n", pattern],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"fc-match failed for {pattern}: {exc}") from exc
    seen, ordered = set(), []
    for line in out.splitlines():
        path, _, index = line.partition("\t")
        if not path:
            continue
        key = (path, int(index or 0))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
        if len(ordered) >= _MAX_CANDIDATES:
            break
    if not ordered:
        raise SystemExit(f"fc-match returned nothing for {pattern}")
    return tuple(ordered)


@functools.lru_cache(maxsize=512)
def _cmap(path: str, index: int) -> frozenset[int]:
    """Codepoints a face actually covers.

    Not the same as "renders something": a face with no glyph yields .notdef,
    which has ink, so a bitmap test would call every font a match.
    """
    from fontTools.ttLib import TTFont, TTCollection
    try:
        if path.lower().endswith((".ttc", ".otc")):
            with TTCollection(path, lazy=True) as coll:
                font = coll.fonts[index] if index < len(coll.fonts) else coll.fonts[0]
                return frozenset(font.getBestCmap())
        with TTFont(path, fontNumber=index, lazy=True) as font:
            return frozenset(font.getBestCmap())
    except Exception:
        return frozenset()


def _blobs(mask: np.ndarray) -> int:
    """Count 4-connected components of ink."""
    seen = np.zeros_like(mask, dtype=bool)
    n, H, W = 0, *mask.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                n += 1
                stack = [(y, x)]
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < H and 0 <= nx < W
                                and mask[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return n


_BRAILLE_DOTS = ("\u2801", "\u2802", "\u2804", "\u2808",
                 "\u2810", "\u2820", "\u2840", "\u2880")


@functools.lru_cache(maxsize=256)
def _braille_quality(path: str, index: int, px: int,
                     cw: int, ch: int, ascent: int) -> tuple[int, float, float, float]:
    """Measure whether a face's Braille actually tiles a terminal cell.

    Three things can go wrong, and all three are present on this machine:

      * FreeMono draws the seven *unset* dots as hollow rings, so a one-dot
        glyph has eight blobs of ink.  The MV composes every figure out of
        dots; a ring around each one would fill the screen with circles.
      * Cascadia Code draws only the set dots, but its cell is short: the
        four dot rows span 55% of the line box, so the dots bunch in the
        middle and the vertical pitch no longer matches the horizontal one
        (41% ripple), which would show up as banding in a dot figure.
      * A face can simply not fill the cell.

    Returns (blobs for a one-dot glyph, pitch ripple, x coverage, y coverage).
    """
    scale = max(px, 48)
    font = ImageFont.truetype(path, scale, index=index)
    cell = (max(cw, 1) * scale / max(px, 1), max(ch, 1) * scale / max(px, 1))
    a, _ = font.getmetrics()
    dy = ascent * scale / max(px, 1) - a

    def ink(ch_: str) -> tuple[np.ndarray, float | None, float | None]:
        box = scale * 3
        img = Image.new("L", (box, box), 0)
        ImageDraw.Draw(img).text((box // 3, box // 3 + dy), ch_, font=font,
                                 fill=255)
        arr = np.asarray(img) > 24
        if not arr.any():
            return arr, None, None
        ys, xs = np.nonzero(arr)
        w = np.asarray(img)[ys, xs].astype(float)
        return arr, (xs * w).sum() / w.sum() - box // 3, \
            (ys * w).sum() / w.sum() - box // 3 - dy

    single, blobs = {}, _blobs(ink("\u2801")[0])
    for ch_ in _BRAILLE_DOTS:
        _, cx, cy = ink(ch_)
        if cx is None:
            return blobs, 99.0, 0.0, 0.0
        single[ch_] = (cx, cy)

    left = np.mean([single[c][0] for c in _BRAILLE_DOTS[0:3]]
                   + [single["\u2840"][0]])
    right = np.mean([single[c][0] for c in _BRAILLE_DOTS[3:6]]
                    + [single["\u2880"][0]])
    within = (right - left) / cell[0]
    across = 1.0 - within
    ripple = abs(within - across) / ((within + across) / 2) if within > 0 else 9.9

    full, _, _ = ink("\u28ff")
    ys, xs = np.nonzero(full)
    cov_x = (xs.max() - xs.min() + 1) / cell[0]
    cov_y = (ys.max() - ys.min() + 1) / cell[1]
    return blobs, float(ripple), float(cov_x), float(cov_y)

@dataclass
class Face:
    path: str
    index: int
    font: ImageFont.FreeTypeFont
    ascent: int
    reason: str
    ink_w: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class FontStack:
    """The terminal's font, its cell, and a resolved face per character."""

    def __init__(self, fixed: FixedFont, scale: int = 4, dpi: float = 96.0,
                 verbose: bool = False):
        self.fixed = fixed
        self.dpi = dpi
        native_px = fixed.px(dpi)

        base_cell = qt_cell(fixed.family, native_px)
        self.primary_path, self.primary_index = fc_candidates(
            fixed.family, ord("M"))[0]
        self._probe_px = max(native_px, 64)
        self._probe = ImageFont.truetype(self.primary_path, self._probe_px,
                                         index=self.primary_index)
        if base_cell is None:
            base_cell = pil_fallback_cell(self._probe, native_px)

        self.native_cell = base_cell
        self.cell = base_cell.scaled(scale)
        self.px = native_px * max(1, int(scale))
        self.cw, self.ch = self.cell.w, self.cell.h
        self.verbose = verbose

        self._faces: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._metrics: dict[tuple[str, int], tuple[int, int]] = {}
        self._face_of: dict[str, Face] = {}
        self._usage: dict[str, int] = {}

    # -- rasterising one glyph -------------------------------------------

    def _font(self, path: str, index: int) -> ImageFont.FreeTypeFont:
        key = (path, index)
        f = self._faces.get(key)
        if f is None:
            f = ImageFont.truetype(path, self.px, index=index)
            self._faces[key] = f
            self._metrics[key] = f.getmetrics()
        return f

    def _ink(self, font: ImageFont.FreeTypeFont, ch: str,
             box: int | None = None) -> tuple[int, int, int, int]:
        """Bounding box of a glyph's ink, relative to its draw origin."""
        box = box or max(self.cw * 3, self.ch * 2)
        img = Image.new("L", (box, box), 0)
        ImageDraw.Draw(img).text((box // 3, box // 3), ch, font=font, fill=255)
        a = np.asarray(img)
        ys, xs = np.nonzero(a > 24)
        if not len(xs):
            return (0, 0, 0, 0)
        off = box // 3
        return (int(xs.min()) - off, int(xs.max()) - off,
                int(ys.min()) - off, int(ys.max()) - off)

    def _usable(self, path: str, index: int, cp: int,
                font: ImageFont.FreeTypeFont) -> tuple[bool, str, int]:
        if cp not in _cmap(path, index):
            return False, "no coverage", 0
        if BRAILLE_LO <= cp <= BRAILLE_HI:
            blobs, ripple, cov_x, cov_y = _braille_quality(
                path, index, self.px, self.cw, self.ch, int(self.cell.ascent))
            if blobs != 1:
                return False, f"{blobs} blobs for a one-dot glyph", 0
            if ripple >= 0.15:
                return False, f"dot pitch ripple {ripple * 100:.0f}%", 0
            if cov_x < 0.6 or cov_y < 0.6:
                return False, (f"dots cover only {cov_x:.2f}x{cov_y:.2f} "
                               f"of the cell"), 0
        x0, x1, y0, y1 = self._ink(font, chr(cp))
        w = x1 - x0 + 1
        # one pixel of slack: box-drawing glyphs deliberately meet the edges
        if x0 < -1 or x1 > self.cw:
            return False, f"ink {w}px overflows {self.cw}px cell", w
        # Vertical placement matters as much as width.  A face with a taller
        # ascent is pushed up to share the baseline, and a box-drawing stem
        # then stops short of the cell bottom -- visible as gaps in a grid
        # line drawn one character per row.  Noto Sans CJK does exactly this
        # to U+254E, so the test is not hypothetical.
        dy = self.cell.ascent - self._metrics[(path, index)][0]
        top, bottom = y0 + dy, y1 + dy
        if top < -1 or bottom > self.ch:
            return False, f"ink rows [{top}..{bottom}] leave {self.ch}px cell", w
        return True, "ink fits cell", w

    def face_for(self, ch: str) -> Face:
        got = self._face_of.get(ch)
        if got is not None:
            return got
        cp = ord(ch)

        # The configured face always wins for characters it covers, whatever
        # its metrics look like: it *defines* the cell.  WQY's full block and
        # box stems are drawn to the em box (1.1724em) and so overshoot Qt's
        # hinted 1.1538em line height by a few pixels -- which is exactly what
        # Konsole does too, and it clips.  Applying the fit tests here would
        # reject the terminal's own font in favour of a stranger's.
        if cp in _cmap(self.primary_path, self.primary_index):
            x0, x1 = self._ink(self._font(self.primary_path,
                                          self.primary_index), ch)[:2]
            chosen = Face(self.primary_path, self.primary_index,
                          self._font(self.primary_path, self.primary_index),
                          self._metrics[(self.primary_path,
                                         self.primary_index)][0],
                          "primary face", x1 - x0 + 1)
            self._face_of[ch] = chosen
            self._usage[chosen.name] = self._usage.get(chosen.name, 0) + 1
            return chosen

        chosen: Face | None = None
        rejections: list[str] = []
        for path, index in fc_candidates(self.fixed.family, cp):
            font = self._font(path, index)
            ok, reason, ink_w = self._usable(path, index, cp, font)
            if ok:
                chosen = Face(path, index, font, self._metrics[(path, index)][0],
                              reason, ink_w)
                break
            rejections.append(f"{os.path.basename(path)}#{index}: {reason}")
        if chosen is None:                     # nothing usable; take the best
            path, index = fc_candidates(self.fixed.family, cp)[0]
            font = self._font(path, index)
            x0, x1, _, _ = self._ink(font, chr(cp))
            chosen = Face(path, index, font, self._metrics[(path, index)][0],
                          "no candidate fit; clipped", x1 - x0 + 1)
        if self.verbose and rejections:
            print(f"    U+{cp:04X} {ch!r} -> {chosen.name}#{chosen.index}"
                  f"  (skipped {'; '.join(rejections[:3])})")
        self._face_of[ch] = chosen
        self._usage[chosen.name] = self._usage.get(chosen.name, 0) + 1
        return chosen

    # -- compositing ------------------------------------------------------

    # Box and block characters are drawn to overshoot the cell by about a
    # pixel at each edge -- that is deliberate in the font, so that a line or
    # a solid block drawn one character per cell joins its neighbours.  Clip
    # the overshoot away and every join becomes a hairline: the edge pixels
    # are only part-covered (156/255 and 176/255 for the full block here), so
    # two cells meeting show a seam about a third darker than the block.
    # Keeping the overshoot and letting adjacent tiles overlap restores the
    # join the glyph was designed for.  Nothing else is affected: glyphs that
    # do not overshoot simply have empty margin.
    BLEED = 1

    def mask(self, ch: str) -> Image.Image:
        """An L-mode tile for one character, with a pixel of bleed.

        The glyph is drawn at the cell origin on the primary face's baseline.
        It is never squashed into the cell -- the terminal does not rescale a
        fallback glyph, and squashing DejaVu's Braille into a narrower cell
        would modulate the dot pitch by about 50%.
        """
        face = self.face_for(ch)
        b = self.BLEED
        tile = Image.new("L", (self.cw + 2 * b, self.ch + 2 * b), 0)
        ImageDraw.Draw(tile).text(
            (b, b + self.cell.ascent - face.ascent), ch, font=face.font,
            fill=255)
        return tile

    def describe(self) -> str:
        faces = ", ".join(f"{n} ({c})" for n, c in sorted(self._usage.items()))
        return (f"{self.fixed.family} @ {self.fixed.point_size:g}pt "
                f"-> {self.native_cell.w}x{self.native_cell.h}px native cell "
                f"[{self.native_cell.source}], rendered at {self.px}px "
                f"as {self.cw}x{self.ch} "
                f"(aspect {self.cell.aspect:.4f}:1); faces: {faces or 'n/a'}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--dpi", type=float, default=96.0)
    ap.add_argument("--rows", type=int, default=54)
    ap.add_argument("--chars", default="M█─╭╯░▒▚▞╎│·⣿⡇⠁")
    a = ap.parse_args(argv)

    fixed = read_kde_fixed()
    print(f"kdeglobals   {KDEGLOBALS}")
    print(f"family       {fixed.family!r}  hint={fixed.style_hint} "
          f"weight={fixed.weight}")
    print(f"size         {fixed.point_size:g}pt pixelSize={fixed.pixel_size} "
          f"dpi={a.dpi:g} -> {fixed.px(a.dpi)}px native")

    st = FontStack(fixed, scale=a.scale, dpi=a.dpi, verbose=True)
    print(f"primary      {st.primary_path} index={st.primary_index}")
    print(f"cell         native {st.native_cell.w}x{st.native_cell.h} "
          f"({st.native_cell.aspect:.4f}:1, {st.native_cell.source})"
          f" -> {st.cw}x{st.ch} ({st.cell.aspect:.4f}:1)")
    rows = a.rows
    cols = int(round(rows * (16 / 9) * st.ch / st.cw))
    cols += cols % 2
    rows += rows % 2
    print(f"grid         {cols}x{rows} -> {cols * st.cw}x{rows * st.ch} "
          f"({cols * st.cw / (rows * st.ch):.5f}:1)")
    print("\nper-character resolution:")
    for ch in a.chars:
        f = st.face_for(ch)
        print(f"  U+{ord(ch):04X} {ch}  {f.name}#{f.index}  "
              f"ink={f.ink_w}px  dy={st.cell.ascent - f.ascent:+.0f}  {f.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
