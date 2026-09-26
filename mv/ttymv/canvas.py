"""A character-cell canvas with an ANSI encoder.

Everything the MV draws goes through here.  The encoder emits the smallest
escape stream it can: colour is only re-emitted when it actually changes, and
CJK cells are marked so the second column does not overwrite the glyph.
"""

from __future__ import annotations

from wcwidth import wcwidth

DEFAULT = -1          # "terminal default colour"
KEEP = -2             # leave this cell's background exactly as it is
WIDE = "\0"           # right half of a double-width glyph: emit nothing


def rgb(r: int, g: int, b: int) -> int:
    return (max(0, min(255, int(r))) << 16
            | max(0, min(255, int(g))) << 8
            | max(0, min(255, int(b))))


def unpack(c: int) -> tuple[int, int, int]:
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


def mix(a: int, b: int, t: float) -> int:
    """Linear blend between two packed colours, t in [0,1]."""
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    ar, ag, ab = unpack(a)
    br, bg, bb = unpack(b)
    return rgb(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def scale(c: int, k: float) -> int:
    r, g, b = unpack(c)
    return rgb(r * k, g * k, b * k)


# --- 16 / 256 colour fallbacks -------------------------------------------

_ANSI16 = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
]


def to_ansi16(c: int) -> int:
    r, g, b = unpack(c)
    best, bi = 1 << 30, 0
    for i, (ar, ag, ab) in enumerate(_ANSI16):
        d = (r - ar) ** 2 + (g - ag) ** 2 + (b - ab) ** 2
        if d < best:
            best, bi = d, i
    return bi


def to_ansi256(c: int) -> int:
    r, g, b = unpack(c)
    # 6x6x6 colour cube
    def q(v: int) -> int:
        return int(round((v / 255.0) * 5))
    ri, gi, bi_ = q(r), q(g), q(b)
    cube = 16 + 36 * ri + 6 * gi + bi_
    cr = [0, 95, 135, 175, 215, 255]
    er = abs(r - cr[ri]) + abs(g - cr[gi]) + abs(b - cr[bi_])
    gray = int(round((((r + g + b) / 3.0) - 8) / 10))
    if 0 <= gray <= 23:
        gv = 8 + 10 * gray
        eg = abs(r - gv) + abs(g - gv) + abs(b - gv)
        if eg < er:
            return 232 + gray
    return cube


class Canvas:
    __slots__ = ("w", "h", "ch", "fg", "bg", "_efg", "_ebg", "_mode")

    def __init__(self, w: int, h: int):
        self.w = max(1, w)
        self.h = max(1, h)
        n = self.w * self.h
        self.ch: list[str] = [" "] * n
        self.fg: list[int] = [DEFAULT] * n
        self.bg: list[int] = [DEFAULT] * n
        self._efg: dict[int, str] = {}
        self._ebg: dict[int, str] = {}
        self._mode = "true"

    # -- geometry ---------------------------------------------------------

    def clear(self, ch: str = " ", fg: int = DEFAULT, bg: int = DEFAULT) -> None:
        n = self.w * self.h
        self.ch = [ch] * n
        self.fg = [fg] * n
        self.bg = [bg] * n

    def put(self, x: int, y: int, ch: str, fg: int = DEFAULT,
            bg: int = KEEP) -> None:
        """Draw a cell.  `bg` defaults to KEEP, so drawing text or a glyph does
        not punch the terminal's default background through the frame colour
        the canvas was cleared with -- which shows up as hairlines between
        adjacent block glyphs."""
        if 0 <= x < self.w and 0 <= y < self.h:
            i = y * self.w + x
            self.ch[i] = ch
            self.fg[i] = fg
            if bg != KEEP:
                self.bg[i] = bg

    def put_wide(self, x: int, y: int, ch: str, fg: int = DEFAULT,
                 bg: int = KEEP) -> None:
        """Place a double-width glyph, reserving the following cell."""
        if y < 0 or y >= self.h:
            return
        if 0 <= x < self.w:
            self.put(x, y, ch, fg, bg)
        if 0 <= x + 1 < self.w:
            i = y * self.w + x + 1
            self.ch[i] = WIDE
            self.fg[i] = fg
            if bg != KEEP:
                self.bg[i] = bg

    def add(self, x: int, y: int, ch: str, fg: int = DEFAULT) -> None:
        """Write only where the canvas is currently blank."""
        if 0 <= x < self.w and 0 <= y < self.h:
            i = y * self.w + x
            if self.ch[i] == " ":
                self.ch[i] = ch
                self.fg[i] = fg

    def text(self, x: int, y: int, s: str, fg: int = DEFAULT,
             bg: int = KEEP) -> int:
        if not (0 <= y < self.h):
            return x
        cx = x
        for ch in s:
            w = wcwidth(ch)
            if w < 0:
                w = 1
            if cx >= self.w:
                break
            if cx >= 0:
                if w == 2:
                    self.put_wide(cx, y, ch, fg, bg)
                else:
                    self.put(cx, y, ch, fg, bg)
            cx += w
        return cx

    def text_right(self, x_end: int, y: int, s: str, fg: int = DEFAULT,
                   bg: int = KEEP) -> int:
        return self.text(x_end - str_width(s), y, s, fg, bg)

    def text_center(self, y: int, s: str, fg: int = DEFAULT,
                    bg: int = KEEP) -> int:
        return self.text((self.w - str_width(s)) // 2, y, s, fg, bg)

    def fill(self, x0: int, y0: int, x1: int, y1: int, ch: str = " ",
             fg: int = DEFAULT, bg: int = KEEP) -> None:
        x0, x1 = max(0, min(x0, x1)), min(self.w - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.h - 1, max(y0, y1))
        for y in range(y0, y1 + 1):
            base = y * self.w
            for x in range(x0, x1 + 1):
                self.ch[base + x] = ch
                self.fg[base + x] = fg
                if bg != KEEP:
                    self.bg[base + x] = bg

    def hline(self, x0: int, x1: int, y: int, ch: str = "─",
              fg: int = DEFAULT, bg: int = KEEP) -> None:
        self.fill(x0, y, x1, y, ch, fg, bg)

    def vline(self, x: int, y0: int, y1: int, ch: str = "│",
              fg: int = DEFAULT, bg: int = KEEP) -> None:
        self.fill(x, y0, x, y1, ch, fg, bg)

    def box(self, x0: int, y0: int, x1: int, y1: int, fg: int = DEFAULT,
            bg: int = KEEP, rounded: bool = False) -> None:
        if x1 <= x0 or y1 <= y0:
            return
        tl, tr, bl, br = ("╭", "╮", "╰", "╯") if rounded else ("┌", "┐", "└", "┘")
        self.hline(x0 + 1, x1 - 1, y0, "─", fg, bg)
        self.hline(x0 + 1, x1 - 1, y1, "─", fg, bg)
        self.vline(x0, y0 + 1, y1 - 1, "│", fg, bg)
        self.vline(x1, y0 + 1, y1 - 1, "│", fg, bg)
        self.put(x0, y0, tl, fg, bg)
        self.put(x1, y0, tr, fg, bg)
        self.put(x0, y1, bl, fg, bg)
        self.put(x1, y1, br, fg, bg)

    # -- bitmap helpers ---------------------------------------------------

    def bitmap(self, rows: list[list[bool]], x: int, y: int,
               fg: int, scale: int = 1, on: str = "█", off: str | None = None,
               off_fg: int = DEFAULT, shade=None) -> None:
        """Blit a bool bitmap.  `shade` maps (row, col) -> colour when given."""
        for ry, row in enumerate(rows):
            for rx, v in enumerate(row):
                if not v and off is None:
                    continue
                ch = on if v else off
                colour = shade(ry, rx) if (v and shade) else fg
                for dy in range(scale):
                    yy = y + ry * scale + dy
                    for dx in range(scale):
                        self.put(x + rx * scale + dx, yy, ch,
                                 colour if v else off_fg)

    # -- output -----------------------------------------------------------

    def _fg_seq(self, fg: int) -> str:
        """Foreground half of the pen, cached per frame."""
        hit = self._efg.get(fg)
        if hit is not None:
            return hit
        if fg == DEFAULT:
            esc = "\x1b[39m"
        elif self._mode == "true":
            esc = f"\x1b[38;2;{(fg >> 16) & 255};{(fg >> 8) & 255};{fg & 255}m"
        elif self._mode == "256":
            esc = f"\x1b[38;5;{to_ansi256(fg)}m"
        else:
            i = to_ansi16(fg)
            esc = f"\x1b[{30 + i}m" if i < 8 else f"\x1b[{90 + i - 8}m"
        self._efg[fg] = esc
        return esc

    def _bg_seq(self, bg: int) -> str:
        """Background half of the pen.

        Kept separate from the foreground so a foreground change does not drag
        a redundant background escape along with it: most frames use a single
        background colour for thousands of cells."""
        hit = self._ebg.get(bg)
        if hit is not None:
            return hit
        if bg == DEFAULT:
            esc = "\x1b[49m"
        elif self._mode == "true":
            esc = f"\x1b[48;2;{(bg >> 16) & 255};{(bg >> 8) & 255};{bg & 255}m"
        elif self._mode == "256":
            esc = f"\x1b[48;5;{to_ansi256(bg)}m"
        else:
            i = to_ansi16(bg)
            esc = f"\x1b[{40 + i}m" if i < 8 else f"\x1b[{100 + i - 8}m"
        self._ebg[bg] = esc
        return esc

    def render(self, mode: str = "true", reserve_corner: bool = True) -> str:
        """Encode the frame for a terminal.

        Two terminal behaviours are designed around rather than trusted:

        *Rows are addressed absolutely* (ESC[r;1H) and separated by no line
        feed at all.  The player runs the tty in raw mode, which clears OPOST,
        so a bare LF is a line feed with no carriage return: the cursor stays
        in the last column with the auto-wrap flag armed, and terminals
        disagree about whether to resolve that wrap before or after the LF.
        Konsole resolves it first and eats an extra line per row, which doubles
        the picture vertically.  A cursor-move escape also cancels any pending
        wrap, so this is safe at any width.

        *The bottom-right cell is never written* when `reserve_corner` is set.
        A VT that scrolls when that cell is touched (the Linux console among
        them) would otherwise push the whole screen up one line per frame --
        which reads as continuous scrolling instead of playback.  The cost is
        one background-coloured cell in the extreme corner.
        """
        self._mode = mode
        self._efg = {}
        self._ebg = {}
        out: list[str] = []
        push = out.append
        w = self.w
        last_row = self.h - 1
        cur_fg = cur_bg = -2
        for y in range(self.h):
            push(f"\x1b[{y + 1};1H")
            limit = w - 1 if (reserve_corner and y == last_row) else w
            base = y * w
            for x in range(limit):
                i = base + x
                ch = self.ch[i]
                if ch == WIDE:
                    continue
                fg = self.fg[i]
                if fg != cur_fg:
                    push(self._fg_seq(fg))
                    cur_fg = fg
                bg = self.bg[i]
                if bg != cur_bg:
                    push(self._bg_seq(bg))
                    cur_bg = bg
                push(ch)
        push("\x1b[0m")
        return "".join(out)

    def render_plain(self) -> str:
        """No escapes -- for dumps, diffs and the PNG verification tool."""
        rows = []
        for y in range(self.h):
            base = y * self.w
            rows.append("".join(c for c in self.ch[base:base + self.w]
                                if c != WIDE))
        return "\n".join(rows)


def str_width(s: str) -> int:
    return sum(max(1, wcwidth(c)) for c in s)


def truncate(s: str, width: int) -> str:
    out, used = [], 0
    for ch in s:
        w = max(1, wcwidth(ch))
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)
