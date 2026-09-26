"""The shot list: a new composition roughly every two lyric lines.

The acts are 5-30 seconds long, which is a long time to hold one picture.  A
music video cuts far more often than that, so the lyric cue sheet is grouped
into shots of about two fragments each (~60 of them across the song) and every
shot re-stages the act: a different motif behind it, a different camera
(zoom / tilt / mirror) and a different accent.

Nothing here invents timing.  Shot boundaries *are* lyric cues, taken from the
published timed lyric for this master, so a cut always lands on a syllable the
singer actually delivers.  `mv/tools/selftest.py` asserts that.

Motifs are dealt from a shuffled bag rather than by index, so the same motif
cannot come round twice in quick succession and the pattern never settles into
a cycle the eye can predict.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import tears as TEARS

from . import motifs

# How a shot is bounded.  A shot ends when the *sentence* ends -- the moment
# the lyric reaches the word it has been building towards -- so the picture
# changes with the meaning rather than every fixed number of fragments.  The
# count and the clock are only there to stop a sentence running away.
FRAGMENTS_PER_SHOT = 3
MIN_SHOT = 1.0
MAX_SHOT = 3.6

_STRIP = " \t.,;:!?'\"()[]-–—"
import re as _re
# a letter either side of a hyphen: "LO-O-OVE".  No \b, because the
# stutter starts mid-word -- "L" then "O-O".
_STUTTER_RE = _re.compile(r"[A-Za-z]-[A-Za-z]")


def _is_caps_word(text: str) -> bool:
    core = text.strip(_STRIP)
    if len(core) < 2:
        return False
    letters = [c for c in core if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

# Measured from the master (see mv/ttymv/analyze.py).  Used to snap the cuts
# inside instrumental stretches onto the beat, so a section with no words still
# cuts musically rather than at an arbitrary fraction.
BPM = 130.0
BEAT_PHASE = 0.1254
BEAT = 60.0 / BPM


def snap_to_beat(t: float) -> float:
    return BEAT_PHASE + round((t - BEAT_PHASE) / BEAT) * BEAT

PROJECT_DIR = Path(__file__).resolve().parent.parent
LYRICS_DIR = PROJECT_DIR.parent / "lyrics"


@dataclass
class BarSpec:
    """One progress element inside a shot.

    ``semantic`` is what the bar *means*, and it is deliberately never "how far
    through the file we are".  Each one tracks something the music or the story
    is actually doing:

    section  how far through the current act
    phrase   how far through this shot's own lyric phrase
    beat     position inside a four-beat bar -- it sweeps and resets
    energy   a level meter with slow release, driven by the low end
    story    how far away the other one is
    cue      filling towards the next sung line
    """
    angle: float = 0.0
    y: float = 0.62
    length: float = 0.86
    thickness: float = 1.0
    semantic: str = "phrase"
    framed: bool = True


@dataclass
class Shot:
    index: int
    t0: float
    t1: float
    text: str = ""
    motif: str = "orbit"
    zoom: float = 1.0
    rot: float = 0.0
    mirror: bool = False
    pan_x: float = 0.0
    pan_y: float = 0.0
    accent: str = "ring"
    phase0: float = 0.0
    # the progress elements, in a minority of shots.  angle 0 is level; the
    # rest are tilted, as asked: "can be placed level, or tilted".  A shot may
    # carry more than one, of different lengths and weights.
    bars: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def bar(self) -> bool:
        return bool(self.bars)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def read_lrc(path) -> list[tuple[float, str]]:
    """(time, text) pairs from one lyric file."""
    import re
    try:
        raw_text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return []
    out = []
    for raw in raw_text.splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", raw.strip())
        if m:
            out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3)))
    return sorted(out)


def lyric_path():
    """The bundled lyric, if there is one."""
    return next(iter(sorted(LYRICS_DIR.glob("*.lrc"))), None)


def lyric_has_words(path=None) -> bool:
    """Does this lyric still carry the lyric, or only the initials?

    The copy that ships here has each cue reduced to its first letter; the
    full text is kept locally and not redistributed.  Checks that need real
    words have to know which one they are looking at rather than quietly
    failing on a redacted file.
    """
    p = path or lyric_path()
    if not p:
        return False
    # credit lines are kept whole even in the redacted copy, so they are not
    # evidence that the lyric itself is still here
    credits = ("作曲", "作词", "编曲", "作詞")
    return any(len(t.strip()) > 3 and not t.strip().startswith(credits)
               for _at, t in read_lrc(p))


def _cue_times() -> list[tuple[float, str]]:
    """Published timed lyric, if the file is present."""
    path = lyric_path()
    return read_lrc(path) if path else []


_CUES: list[tuple[float, str]] | None = None


def cues() -> list[tuple[float, str]]:
    global _CUES
    if _CUES is None:
        _CUES = _cue_times() or _fallback_grid()
    return _CUES


def sung_spans(tail: float = 0.35, merge: float = 0.8,
               max_line: float = 3.0) -> list[tuple[float, float]]:
    """When a voice is present, according to the published timed lyric.

    Used to tell a vocal break from an instrumental dropout: both look like
    the same hole in the mid band, and only one of them is a stutter.  The
    lyric supplies *when* someone is singing; the audio still supplies *where*
    the line breaks.
    """
    rows = [t for t, text in cues()
            if text and not text.startswith(("作曲", "作词", "编曲", "作詞"))]
    if not rows:
        return []
    spans = []
    for i, t in enumerate(rows):
        end = rows[i + 1] if i + 1 < len(rows) else t + 2.5
        # A cue is not a licence to call everything until the next cue "sung":
        # the instrumental stretches are exactly the gaps between distant
        # cues, and they are 14 and 16 seconds long.
        spans.append([t, min(end, t + max_line) + tail])
    out = [spans[0]]
    for a, b in spans[1:]:
        if a - out[-1][1] <= merge:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


#: Where the vocal stutters, as (time, strength).
#:
#: Kept here as a name because the renderer and the self-test ask for
#: "stutters"; the table itself lives in ``tears.py`` with the rest of the
#: schedule, and is a copy of ``tears.STUTTERS`` -- assigned, not duplicated,
#: so the two can never drift apart.
STUTTER_TIMES: tuple[tuple[float, float], ...] = TEARS.STUTTERS


def stutter_cues() -> list[tuple[float, float]]:
    """Where the vocal stutters.  See ``tears.STUTTERS``."""
    return list(STUTTER_TIMES)


def stutter_cues_from_lyric(path=None) -> list[tuple[float, float]]:
    """Re-derive the stutters from the lyric -- the oracle for the table.

    Only meaningful with the full text; with the initials that ship in this
    repository it finds nothing, which is exactly why ``STUTTER_TIMES`` is
    baked in.
    """
    rows = [(t, text) for t, text in _rows_from(path)
            if text and not text.startswith(("作曲", "作词", "编曲", "作詞"))]
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(rows):
        t, text = rows[i]
        low = text.lower()
        words = [w.strip(_STRIP) for w in low.replace("-", " ").split()]
        words = [w for w in words if w]
        repeats = len(words) - len(set(words))
        hyphen = bool(_STUTTER_RE.search(text))

        run = 1
        while (i + run < len(rows)
               and _similar(rows[i + run][1], text)):
            run += 1
        if run >= 3:
            repeats += min(4, run - 2)
        if repeats >= 1 or hyphen:
            strength = min(1.0, 0.42 + 0.16 * repeats + (0.20 if hyphen else 0.0))
            out.append((round(t, 3), round(strength, 3)))
        i += run if run >= 3 else 1
    return out


def _rows_from(path=None) -> list[tuple[float, str]]:
    """(time, text) from a lyric file, or from the bundled one."""
    return read_lrc(path) if path is not None else cues()


def _similar(a: str, b: str) -> bool:
    wa = {w.strip(_STRIP) for w in a.lower().replace("-", " ").split()}
    wb = {w.strip(_STRIP) for w in b.lower().replace("-", " ").split()}
    wa.discard("")
    wb.discard("")
    if not wa or not wb:
        return False
    return len(wa & wb) / max(1, min(len(wa), len(wb))) >= 0.75


def next_cue_after(t: float) -> float | None:
    """When the next line is sung after `t`, or None if the song is done."""
    for at, text in cues():
        if at > t + 1e-6 and text and not text.startswith(("作曲", "作词")):
            return at
    return None


def prev_cue_before(t: float) -> float | None:
    prev = None
    for at, text in cues():
        if at <= t and text and not text.startswith(("作曲", "作词")):
            prev = at
        elif at > t:
            break
    return prev


def _fallback_grid() -> list[tuple[float, str]]:
    """No lyric file: fall back to an even grid so the shot system still runs."""
    return [(i * 3.4, "") for i in range(0, 62)]


def _group(cues: list[tuple[float, str]]) -> list[tuple[float, float, str]]:
    out = []
    cur: list[tuple[float, str]] = []
    for t, text in cues:
        if text.startswith(("作曲", "作词", "编曲", "作詞")):
            continue
        closes = bool(cur) and _is_caps_word(cur[-1][1])
        if cur and (closes
                    or len(cur) >= FRAGMENTS_PER_SHOT
                    or t - cur[0][0] >= MAX_SHOT):
            out.append((cur[0][0], t, " ".join(x[1] for x in cur)))
            cur = []
        cur.append((t, text))
    if cur:
        out.append((cur[0][0], cues[-1][0] + 2.0,
                    " ".join(x[1] for x in cur)))
    # merge anything that came out too short to read
    merged: list[list] = []
    for t0, t1, txt in out:
        if merged and t1 - t0 < MIN_SHOT:
            merged[-1][1] = t1
            merged[-1][2] = (merged[-1][2] + " " + txt).strip()
        else:
            merged.append([t0, t1, txt])

    # and split anything too long.  The instrumental stretch has no lyric for
    # 14 seconds; without this it becomes one shot held for 14 seconds, which
    # is the exact monotony the shot system exists to remove.
    split: list[list] = []
    for t0, t1, txt in merged:
        dur = t1 - t0
        if dur <= MAX_SHOT * 1.3:
            split.append([t0, t1, txt])
            continue
        parts = max(2, int(math.ceil(dur / MAX_SHOT)))
        step = dur / parts
        edges = [t0] + [snap_to_beat(t0 + k * step) for k in range(1, parts)] \
            + [t1]
        for k in range(parts):
            # only the first part still belongs to the sentence that just
            # finished; the rest are the instrumental after it
            split.append([edges[k], edges[k + 1], txt if k == 0 else ""])
    return [(a, b, c) for a, b, c in split]


def _deal(n: int, seed: int = 20260926) -> list[str]:
    """Deal motif names from reshuffled bags, never repeating across a seam."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        bag = list(motifs.NAMES)
        rng.shuffle(bag)
        if out and bag and bag[0] == out[-1]:
            bag.append(bag.pop(0))
        out.extend(bag)
    return out[:n]


ACCENTS = ["ring", "burst", "sweep", "grid", "none", "dust"]

# Level is the plain reading, so it stays in the mix -- but as the minority.
# Placeholder kept only to draw the same number of random values as before, so
# that reworking the angles cannot disturb the semantic / length / weight /
# framing draws that follow it.  The real angle is set afterwards by
# _randomize_bar_geometry.
_ANGLE_PLACEHOLDER = [0.0] * 10

LEVEL_CHANCE = 0.14
MIN_TILT = 0.13          # ~7.5 degrees: below this it just reads as level
MAX_TILT = 0.85          # ~49 degrees
BAR_ROWS = [0.70, 0.62, 0.54, -0.62, -0.54, -0.70]
# Deliberately excludes anything beat-locked.  "beat" made the bar sweep and
# reset like a level meter and "energy" made it jump on transients; both made
# it read as a VU meter rather than as an instrument showing position.  What
# is left only ever moves smoothly or monotonically.
SEMANTICS = ["section", "phrase", "story", "cue"]
# roughly two fifths of shots carry at least one: often enough to read as a
# motif, rare enough not to become the chrome that was deliberately removed
BAR_CHANCE = 0.42


def _randomize_bar_geometry(shots: list) -> None:
    """Scatter every bar's angle and height; leave everything else alone.

    What a bar *shows* (its semantic, its length, its weight) and when it is on
    screen are decided above and must not move.  Only the placement is
    re-rolled, and it is re-rolled continuously rather than from a short list
    of angles and rows -- with ten angles and six rows the same handful of
    combinations came round often enough to notice.

    The count of level bars is fixed by construction rather than drawn per
    bar.  Drawing each one independently left the realized set at 27% level
    against a 14% target purely through sampling luck at n=52, which is exactly
    what the eye picks up on.

    The two bands keep bars clear of the middle of the frame, where the act's
    subject is drawn, and the slot spread keeps bars in the same shot from
    stacking on top of each other.
    """
    entries = [(si, bi, b) for si, sh in enumerate(shots)
               for bi, b in enumerate(sh.bars)]
    if not entries:
        return
    n = len(entries)
    flags = [True] * int(round(n * LEVEL_CHANCE)) + [False] * (n - int(round(n * LEVEL_CHANCE)))
    rng = random.Random(4242)
    rng.shuffle(flags)

    per_shot: dict = {}
    for si, _bi, _b in entries:
        per_shot[si] = per_shot.get(si, 0) + 1
    band_rng = {si: random.Random(9001 + si * 104729) for si in per_shot}

    seen: dict = {}
    for k, (si, _bi, b) in enumerate(entries):
        r = band_rng[si]
        idx = seen.get(si, 0)
        seen[si] = idx + 1
        bands = [[-0.88, -0.46], [0.46, 0.88]]
        if idx == 0:
            r.shuffle(bands)
        lo, hi = bands[idx % 2]
        per_band = max(1, (per_shot[si] + 1) // 2)
        slot = (idx // 2 + r.random()) / per_band
        b.y = lo + (hi - lo) * min(1.0, slot)
        # Tilted bars are kept out of the shallow band too: one at three
        # degrees is indistinguishable from a level one, so it would inflate
        # the count of bars that read as level without adding any variety.
        if flags[k]:
            b.angle = 0.0
        else:
            mag = r.uniform(MIN_TILT, MAX_TILT)
            b.angle = mag if r.random() < 0.5 else -mag


def _bars_for(rng: random.Random) -> list:
    """One to three bars, of different lengths and weights."""
    if rng.random() >= BAR_CHANCE:
        return []
    n = 1 if rng.random() < 0.58 else (2 if rng.random() < 0.72 else 3)
    rows = rng.sample(BAR_ROWS, min(n, len(BAR_ROWS)))
    out = []
    for i in range(n):
        out.append(BarSpec(
            angle=rng.choice(_ANGLE_PLACEHOLDER),
            y=rows[i],
            length=rng.choice([0.42, 0.55, 0.68, 0.80, 0.92, 1.0]),
            thickness=rng.choice([0.7, 0.9, 1.1, 1.4, 1.8]),
            semantic=rng.choice(SEMANTICS),
            framed=rng.random() < 0.82,
        ))
    return out


def build_shots() -> list[Shot]:
    cues = _cue_times() or _fallback_grid()
    spans = _group(cues)
    names = _deal(len(spans))
    shots: list[Shot] = []
    for i, (t0, t1, text) in enumerate(spans):
        rng = random.Random(1000 + i * 7919)
        shots.append(Shot(
            index=i, t0=t0, t1=t1, text=text,
            motif=names[i],
            zoom=rng.choice([0.86, 0.92, 0.97, 1.0, 1.04, 1.09]),
            rot=rng.choice([0.0, 0.0, 0.13, -0.13, 0.26, -0.26, math.pi / 6]),
            mirror=rng.random() < 0.35,
            pan_x=rng.choice([-0.05, -0.02, 0.0, 0.02, 0.05]),
            pan_y=rng.choice([-0.03, 0.0, 0.03]),
            accent=ACCENTS[i % len(ACCENTS)] if rng.random() < 0.6
            else rng.choice(ACCENTS),
            phase0=rng.uniform(0, 6.283),
            bars=_bars_for(rng),
        ))
    _randomize_bar_geometry(shots)
    return shots
    return shots


_SHOTS: list[Shot] | None = None


def all_shots() -> list[Shot]:
    global _SHOTS
    if _SHOTS is None:
        _SHOTS = build_shots()
    return _SHOTS


def shot_at(t: float) -> Shot | None:
    shots = all_shots()
    if not shots:
        return None
    lo, hi = 0, len(shots) - 1
    if t < shots[0].t0:
        return shots[0]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if shots[mid].t0 <= t:
            lo = mid
        else:
            hi = mid - 1
    return shots[lo]
