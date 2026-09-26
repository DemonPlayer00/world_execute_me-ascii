"""Timed lyrics.

Reads standard LRC.  The copy shipped in ``lyrics/`` is the published timed
lyric for this exact recording (NetEase track 435278010, 211906 ms -- the
same master as the .ogg in this workspace), so cue times are authoritative
rather than estimated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIME = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")

# the song's own punctuation, kept out of the "is this an emphasised word" test
_STRIP = " \t.,;:!?'\"()[]-–—"


@dataclass
class Cue:
    t: float
    end: float
    text: str
    kind: str = "line"        # line | word | meta | title
    phrase: str = ""          # the full sentence this fragment belongs to

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.t)


def _is_word(text: str) -> bool:
    core = text.strip(_STRIP)
    if len(core) < 2:
        return False
    letters = [c for c in core if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def parse(path: str | Path) -> list[tuple[float, str]]:
    out = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        m = _TIME.match(raw.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2).replace(":", "."))
        text = raw.strip()[m.end():].strip()
        out.append((t, text))
    out.sort(key=lambda x: x[0])
    return out


class Lyrics:
    """A cue list with a moving playhead."""

    def __init__(self, cues: list[Cue]):
        self.cues = cues
        self.times = [c.t for c in cues]
        self._i = -1

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, en: str | Path) -> "Lyrics":
        en_rows = parse(en)
        cues: list[Cue] = []
        for t, text in en_rows:
            kind = "line"
            if text.startswith(("作曲", "作词", "编曲", "作詞")):
                kind = "meta"
            elif text.lower().startswith("world.execute"):
                kind = "title"
            elif _is_word(text):
                kind = "word"
            cues.append(Cue(t=t, end=t, text=text, kind=kind))

        # a cue lasts until the next one; a fragment inherits the sentence it
        # belongs to -- everything since the previous emphasised word
        for i, c in enumerate(cues):
            c.end = cues[i + 1].t if i + 1 < len(cues) else c.t + 4.0
        start = 0
        for i, c in enumerate(cues):
            if c.kind in ("meta", "title"):
                c.phrase = c.text
                start = i + 1
                continue
            c.phrase = " ".join(x.text for x in cues[start:i + 1]
                                if x.kind in ("line", "word"))
            if c.kind == "word":
                start = i + 1
        return cls(cues)

    # -- lookup -----------------------------------------------------------

    def index_at(self, t: float) -> int:
        """Index of the cue playing at `t` (-1 before the first one)."""
        import bisect
        i = bisect.bisect_right(self.times, t + 1e-6) - 1
        return i

    def at(self, t: float) -> Cue | None:
        i = self.index_at(t)
        if i < 0:
            return None
        c = self.cues[i]
        if c.kind == "meta":
            return None          # credits are markers, never shown
        return c if t < c.end + 1.2 else None

    def word_at(self, t: float, hold: float = 0.0) -> Cue | None:
        """Most recent emphasised word, held briefly after it stops."""
        i = self.index_at(t)
        while i >= 0:
            c = self.cues[i]
            if c.kind == "word":
                return c if t < c.end + hold else None
            if c.kind in ("meta", "title"):
                return None
            i -= 1
        return None

    def upcoming(self, t: float, n: int = 3) -> list[Cue]:
        i = max(0, self.index_at(t))
        return [c for c in self.cues[i:i + n] if c.kind != "meta"]

    def next_word_after(self, t: float) -> Cue | None:
        for c in self.cues:
            if c.t > t and c.kind == "word":
                return c
        return None


class EmptyLyrics(Lyrics):
    def __init__(self):
        super().__init__([])
