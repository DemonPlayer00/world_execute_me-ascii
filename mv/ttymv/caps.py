"""What the terminal can actually do.

Read from terminfo rather than guessed from $TERM, because the differences
that matter here are exactly the ones a Linux virtual console gets wrong:

* ``linux`` has **no** ``smcup``/``rmcup`` -- there is no alternate screen, so
  ``ESC[?1049h`` is silently ignored and every frame lands in the scrollback.
* ``linux`` reports 8 (sometimes 16) colours.  True-colour SGR is not merely
  ignored: the kernel parses ``ESC[38;2;R;G;Bm`` parameter by parameter, so
  the ``2`` is taken as "dim" and the rest is dropped -- the picture goes dim
  and colourless.
* ``linux`` needs ``ESC[?1c`` / ``ESC[?0c`` to hide and restore the cursor;
  the plain ``ESC[?25l`` alone is not enough.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _decode(term) -> bytes:
    if isinstance(term, bytes):
        return term
    return term.encode() if term else b""


@dataclass
class Caps:
    term: str = ""
    colors: int = 0
    smcup: bytes = b""
    rmcup: bytes = b""
    civis: bytes = b""
    cnorm: bytes = b""
    clear: bytes = b"\x1b[2J"
    is_console: bool = False          # Linux/BSD virtual console
    truecolor: bool = False
    sync_output: bool = False

    # -- construction -----------------------------------------------------

    @classmethod
    def detect(cls) -> "Caps":
        term = os.environ.get("TERM", "")
        colorterm = os.environ.get("COLORTERM", "").lower()
        caps = cls(term=term)
        caps.is_console = term.startswith("linux") or term in ("cons25", "vt100")

        try:
            import curses
            curses.setupterm(term or None)
            caps.colors = int(curses.tigetnum("colors") or 0)
            caps.smcup = _decode(curses.tigetstr("smcup"))
            caps.rmcup = _decode(curses.tigetstr("rmcup"))
            caps.civis = _decode(curses.tigetstr("civis"))
            caps.cnorm = _decode(curses.tigetstr("cnorm"))
            caps.clear = _decode(curses.tigetstr("clear")) or caps.clear
        except Exception:
            # terminfo unavailable (TERM=dumb, no database): fall back to the
            # safe subset that every ANSI terminal understands.
            caps.colors = 8
            caps.civis = b"\x1b[?25l"
            caps.cnorm = b"\x1b[?25h"

        if not caps.civis:
            caps.civis = b"\x1b[?25l"
        if not caps.cnorm:
            caps.cnorm = b"\x1b[?25h"

        caps.truecolor = (not caps.is_console) and colorterm in ("truecolor", "24bit")
        caps.sync_output = (not caps.is_console) and caps.colors >= 256
        return caps

    # -- derived policy ---------------------------------------------------

    def best_color_mode(self) -> str:
        """Colour depth to use when the user did not ask for one.

        Only downgrade on positive evidence.  An unknown terminal (no terminfo
        entry, `TERM=dumb`, a bare pty) must not be treated as a poor one --
        guessing "256" there silently threw away true colour everywhere the
        database had nothing to say."""
        if self.is_console or 0 < self.colors <= 16:
            return "16"
        if self.truecolor:
            return "true"
        if self.colors > 16 and not self.truecolor:
            return "256"
        return "true"          # colours unknown: assume a modern terminal

    def best_glyphs(self) -> str:
        """The kernel console font has no Braille Patterns block."""
        return "block" if self.is_console else "braille"

    def best_fps(self) -> float:
        """A virtual console pushes far fewer cells per second than a GPU
        terminal; the loop drops frames either way, but starting lower keeps
        the first second from looking like a scroll."""
        return 20.0 if self.is_console else 30.0

    def summary(self) -> str:
        return (f"TERM={self.term or '?'} colors={self.colors or '?'} "
                f"altscreen={'yes' if self.smcup else 'no'} "
                f"console={'yes' if self.is_console else 'no'}")
