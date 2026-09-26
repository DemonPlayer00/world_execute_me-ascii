#!/usr/bin/env python3
"""Render a dumped ANSI frame to PNG so the MV can be inspected as an image.

This is a development tool.  It parses exactly the SGR subset Canvas.render
emits (38;2 / 48;2 / 38;5 / 48;5 / 39 / 49 / 0), so what you see here is what
the terminal was actually sent.

    python3 mv/tools/ansi2png.py frames/*.ans --out previews/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SGR = re.compile(r"\x1b\[([0-9;]*)m")
# the player addresses every row absolutely (ESC[r;1H) and never emits a line
# feed, so row breaks are cursor moves, not newlines
CUP = re.compile(r"\x1b\[(\d+);(\d+)H")

FONT_CANDIDATES = [
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
    "/usr/share/fonts/TTF/CascadiaMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
]

# cell aspect: terminal cells are roughly twice as tall as they are wide
CELL_H_FACTOR = 2.0


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size), path
    raise SystemExit("no monospace font found")


_XTERM_CUBE = [0, 95, 135, 175, 215, 255]


def xterm256(n: int):
    """xterm palette index -> RGB (16..231 cube, 232..255 greyscale)."""
    n = int(n)
    if n < 16:
        return (200, 200, 200)
    if n < 232:
        q = n - 16
        return (_XTERM_CUBE[q // 36], _XTERM_CUBE[(q // 6) % 6], _XTERM_CUBE[q % 6])
    v = 8 + 10 * (n - 232)
    return (v, v, v)


def parse_ansi(text: str, width: int | None = None):
    """-> list of rows; each row is a list of (char, fg, bg).

    Understands both layouts: newline-separated (older dumps) and the player's
    absolute cursor addressing, which contains no line feeds at all."""
    if "\n" not in text.rstrip("\n") and CUP.search(text):
        return _parse_addressed(text, width)
    rows = []
    fg = bg = None
    for line in text.split("\n"):
        row = []
        i = 0
        while i < len(line):
            m = SGR.match(line, i)
            if m:
                params = [int(x) for x in m.group(1).split(";") if x != ""]
                j = 0
                while j < len(params):
                    p = params[j]
                    if p == 0:
                        fg = bg = None
                    elif p == 39:
                        fg = None
                    elif p == 49:
                        bg = None
                    elif p in (38, 48):
                        if params[j + 1] == 2:
                            col = tuple(params[j + 2:j + 5])
                            j += 4
                        elif params[j + 1] == 5:
                            col = xterm256(params[j + 2])
                            j += 2
                        else:
                            col = (200, 200, 200)
                            j += 1
                        if p == 38:
                            fg = col
                        else:
                            bg = col
                    j += 1
                i = m.end()
                continue
            row.append((line[i], fg, bg))
            i += 1
        rows.append(row)
    return rows


def _parse_addressed(text: str, width: int | None):
    """Replay a cursor-addressed frame into a row grid."""
    hits = list(CUP.finditer(text))
    if not hits:
        return []
    grid: dict[int, list] = {}
    fg = bg = None
    for n, m in enumerate(hits):
        r = int(m.group(1)) - 1
        end = hits[n + 1].start() if n + 1 < len(hits) else len(text)
        body = text[m.end():end]
        cells = []
        i = 0
        while i < len(body):
            sm = SGR.match(body, i)
            if sm:
                params = [int(x) for x in sm.group(1).split(";") if x != ""]
                j = 0
                while j < len(params):
                    p = params[j]
                    if p == 0:
                        fg = bg = None
                    elif p == 39:
                        fg = None
                    elif p == 49:
                        bg = None
                    elif p in (38, 48):
                        if params[j + 1] == 2:
                            col = tuple(params[j + 2:j + 5]); j += 4
                        elif params[j + 1] == 5:
                            col = xterm256(params[j + 2]); j += 2
                        else:
                            col = (200, 200, 200); j += 1
                        if p == 38:
                            fg = col
                        else:
                            bg = col
                    j += 1
                i = sm.end()
                continue
            cells.append((body[i], fg, bg))
            i += 1
        grid[r] = cells
    h = max(grid) + 1
    w = width or max(len(v) for v in grid.values())
    return [grid.get(y, []) + [(" ", None, None)] * (w - len(grid.get(y, [])))
            for y in range(h)]


def render(rows, font, cell_w: int, cell_h: int, pad: int = 8,
           default_fg=(200, 210, 215), default_bg=(8, 8, 12)):
    width = max((len(r) for r in rows), default=0)
    img = Image.new("RGB", (width * cell_w + pad * 2,
                            len(rows) * cell_h + pad * 2), default_bg)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        for x, (ch, fg, bg) in enumerate(row):
            px, py = pad + x * cell_w, pad + y * cell_h
            if bg is not None:
                d.rectangle([px, py, px + cell_w - 1, py + cell_h - 1], fill=bg)
            if ch == " ":
                continue
            colour = fg if fg is not None else default_fg
            d.text((px, py), ch, font=font, fill=colour)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None, help="output dir (default: alongside)")
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--scale", type=int, default=1)
    a = ap.parse_args(argv)

    files: list[Path] = []
    for p in a.paths:
        path = Path(p)
        files.extend(sorted(path.glob("*.ans")) if path.is_dir() else [path])
    if not files:
        raise SystemExit("no .ans files")

    font, font_path = load_font(a.font_size)
    cell_w = int(round(font.getlength("M")))
    cell_h = int(round(cell_w * CELL_H_FACTOR))
    print(f"font {font_path}  cell {cell_w}x{cell_h}", file=sys.stderr)

    for f in files:
        rows = parse_ansi(f.read_text(encoding="utf-8"))
        img = render(rows, font, cell_w, cell_h)
        if a.scale > 1:
            img = img.resize((img.width * a.scale, img.height * a.scale),
                             Image.NEAREST)
        out_dir = Path(a.out) if a.out else f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (f.stem + ".png")
        img.save(out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
