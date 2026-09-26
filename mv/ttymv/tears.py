#!/usr/bin/env python3
"""When the picture comes apart, decided here and nowhere else.

The song has three stretches where the machine is failing -- the illegal
arguments with the verdict held over sixteen seconds of wreckage, the
executions, and the last one at the end -- plus a handful of places where the
voice is clipped mid-syllable.  All of it is written down in this file.

**Everything the tear answers to lives in this module.**  That is deliberate.
The moments were *found* by reading the song (its lyric, its measured
transients) but they are not *read* from anything at run time: not from the
lyric, not from the analysis cache, not from the audio.  Two reasons:

  * The lyrics are not redistributed with this piece -- the copy in ``lyrics/``
    carries initials instead of words -- so anything that re-derived these
    moments from the lyric at run time would find nothing here and quietly
    ship a different film.  That is exactly what happened once, and it is why
    the stutter table is baked rather than computed.
  * The same film should come out of the same code.  A tear schedule that
    shifts with which files happen to be present is not a schedule, it is a
    side effect.

So the tear level is a pure function of time and the act table: the same
every run, the same with the music as without it.  What the picture *looks
like* while it is slipping is still randomised per frame (see
``player.Show._postfx``) -- the fault is unpredictable, its schedule is not.

``calibrate()`` below checks the properties that make the schedule work, and
``mv/tools/selftest.py`` runs it.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# the failures
# --------------------------------------------------------------------------
#
# The song says three times that something has gone wrong, and each stretch is
# a run of moments rather than one hit: the picture stays unstable for as long
# as the fault lasts.  Each entry is (seconds, intensity).

EVENTS: tuple[tuple[float, float], ...] = (
    # "illegal arguments" -- the verdict, then the wreckage it leaves
    (118.333, 0.55), (120.860, 0.90), (124.890, 1.00),
    (125.708, 0.70), (128.661, 0.90), (131.224, 1.00),
    # the executions
    (147.660, 0.85), (149.520, 0.65), (153.980, 0.70),
    (157.040, 0.65), (161.584, 0.90),
    # and the last one, as the machine stops
    (205.811, 1.00),
)

#: The level each act never falls below, so that a fault reads as a *stretch*
#: rather than as a series of hits with clean picture in between.
FLOOR: dict[str, float] = {"argument": 0.46, "execution": 0.30, "end": 0.52}

#: Seconds for one burst to fade out.
DECAY = 0.85


# --------------------------------------------------------------------------
# the clipped syllables
# --------------------------------------------------------------------------
#
# Where the voice is cut short mid-word.  These are short on purpose: a
# stutter is a clipped syllable, not a failing machine, so the slip lasts
# about a third of a second and does not carry the floor the way a failure
# does.

STUTTERS: tuple[tuple[float, float], ...] = (
    (45.850, 0.58), (49.534, 0.74), (53.225, 0.58), (56.916, 0.74),
    (71.764, 0.58), (90.197, 0.58), (97.739, 0.58), (101.474, 0.74),
    (110.900, 1.00), (147.660, 1.00),
    (179.929, 0.62), (183.646, 0.62), (187.665, 0.62), (191.356, 0.62),
)

#: How long one clipped syllable slips the picture.
STUTTER_SPAN = 0.30

#: How that slip falls away.  Above 1 it drops fast then lingers; the value is
#: the one the tear was tuned with.
STUTTER_SHAPE = 1.4


# --------------------------------------------------------------------------
# the resets
# --------------------------------------------------------------------------
#
# Each long breakdown ends the way a machine does: everything blanks for a
# moment and comes back.  These are the only full-screen flashes in the piece.
# A flash that arrives on a beat is a strobe; a flash that arrives once, when
# a fault clears, is a reset.

RESETS: tuple[tuple[float, float], ...] = (
    (147.660, 1.00),      # the wreckage clears, the blade starts
    (162.632, 0.95),      # the executions end, the last chorus begins
    (207.481, 1.00),      # and the machine stops
)

RESET_ATTACK = 0.09       # how long it stays blank
RESET_DECAY = 0.50


# --------------------------------------------------------------------------
# the curves
# --------------------------------------------------------------------------

def level(t: float, act: str) -> float:
    """How badly the picture is slipping at ``t``, 0..1.

    The floor for the act, plus any burst still decaying.  Never keyed to the
    beat grid: a screen that slips on every downbeat is a metronome, not a
    fault.
    """
    out = FLOOR.get(act, 0.0)
    for at, k in EVENTS:
        dt = t - at
        if 0.0 <= dt < DECAY:
            out = max(out, k * (1.0 - dt / DECAY) ** 1.5)
    return min(1.0, out)


def stutter_level(t: float) -> float:
    """The short slip for a clipped syllable at ``t``, 0..1."""
    out = 0.0
    for at, k in STUTTERS:
        dt = t - at
        if 0.0 <= dt < STUTTER_SPAN:
            u = dt / STUTTER_SPAN
            out = max(out, k * (1.0 - u) ** STUTTER_SHAPE)
    return out


def glitch(t: float, act: str) -> float:
    """Everything that makes the frame unstable at ``t``, 0..1.

    One number, from this file alone -- so the renderer never has to know
    whether a track was measured, synthesised or absent.
    """
    return max(level(t, act), stutter_level(t))


def reset_level(t: float) -> float:
    """How blanked the screen is at ``t``, 0..1."""
    for at, k in RESETS:
        dt = t - at
        if dt < 0.0:
            continue
        if dt < RESET_ATTACK:
            return k
        if dt < RESET_ATTACK + RESET_DECAY:
            u = (dt - RESET_ATTACK) / RESET_DECAY
            return k * (1.0 - u) ** 1.7
    return 0.0


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

#: Two tear moments closer together than this stop reading as separate
#: events.  The tightest pair in the table is 0.85s apart.
MIN_SPACING = 0.70

#: Acts in which the picture is allowed to be unstable at all.
FAILURE_ACTS = ("argument", "execution", "end")


def _act_of(t: float) -> str:
    from . import scenes as S
    return S.act_at(t)


def calibrate(verbose: bool = False) -> list[tuple[str, bool, str]]:
    """Check the schedule against the piece's own structure.

    Not against the lyric and not against the audio: against the act table
    that lives in this repository.  Returns (name, ok, detail) rows.
    """
    from . import scenes as S

    rows: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        rows.append((name, bool(ok), detail))

    events = [t for t, _ in EVENTS]
    stutters = [t for t, _ in STUTTERS]

    # 1. every tear lives inside an act that is supposed to be failing
    outside = [t for t in events if _act_of(t) not in FAILURE_ACTS]
    add("every failure moment is inside a failure act", not outside,
        f"{outside[:4]}")

    # 2. A clipped syllable inside a stretch that is already failing has to
    #    make the picture slip *harder* than that stretch already does, or it
    #    is doing nothing at all.  One of them (the first execution, 147.660)
    #    is deliberately such a moment: a hard clip landing on the burst that
    #    starts the blade.  Requiring the table to have no overlap at all
    #    would have thrown that away; requiring it to earn its place keeps it.
    redundant = [(t, k, round(level(t, _act_of(t)), 3)) for t, k in STUTTERS
                 if _act_of(t) in FAILURE_ACTS and k <= level(t, _act_of(t))]
    add("every clipped syllable inside a failure act still adds to it",
        not redundant, f"{redundant[:3]}")

    # 3. the floor really does hold the act down for its whole length
    spans = {name: [t for t, n in S.ACT_TIMELINE if n == name]
             for name in FAILURE_ACTS}
    thin = []
    for name in FAILURE_ACTS:
        if not spans[name]:
            thin.append(f"{name}: absent")
            continue
        t0 = spans[name][0]
        i = [n for _t, n in S.ACT_TIMELINE].index(name)
        t1 = (S.ACT_TIMELINE[i + 1][0] if i + 1 < len(S.ACT_TIMELINE)
              else S.ACT_TIMELINE[i][0] + 5.0)
        lowest = min(level(t0 + 0.05 * k, name)
                     for k in range(int((t1 - t0) / 0.05)))
        if lowest < FLOOR[name] - 1e-9:
            thin.append(f"{name}: dips to {lowest:.2f}")
    add("the floor holds each failure act for its whole length", not thin,
        "; ".join(thin))

    # 4. away from all of it, the picture is perfectly still
    busy = events + stutters
    calm_bad = []
    t = 0.0
    while t < 210.0:
        act = _act_of(t)
        if (act not in FAILURE_ACTS
                and not any(0.0 <= t - b < max(DECAY, STUTTER_SPAN)
                            for b in busy)):
            if glitch(t, act) > 1e-9:
                calm_bad.append(round(t, 2))
        t += 0.25
    add("the picture is stable away from every failure and stutter",
        not calm_bad, f"{calm_bad[:4]}")

    # 5. close enough together to be one gesture, far enough to be separate
    tight = [(a, b) for a, b in zip(sorted(events), sorted(events)[1:])
             if b - a < MIN_SPACING]
    add(f"no two failure moments are closer than {MIN_SPACING}s", not tight,
        f"{tight[:2]}")

    # 6. it is a fault, not a tempo: the spacing must not be periodic
    gaps = [round(b - a, 3) for a, b in zip(events, events[1:])]
    common = max((gaps.count(g) for g in set(gaps)), default=0)
    add("the failure moments do not fall on a regular grid",
        common <= max(2, len(gaps) // 3),
        f"most repeated gap appears {common}x in {len(gaps)}")
    # ...and it must not be the beat either.  Asked as a question about the
    # *set*, not about each event: with twelve moments and a 40ms window, one
    # of them landing near a beat is what chance alone gives you (about 9% per
    # event).  Demanding zero would be a check that only passes by luck, and
    # the property that actually matters is that they do not line up.
    from . import shots as SH
    import math
    beat = 60.0 / SH.BPM
    tol = 0.020
    offs = [abs((t - SH.BEAT_PHASE) / beat - round((t - SH.BEAT_PHASE) / beat))
            * beat for t in events]
    landed = [round(o, 3) for o in offs if o < tol]
    expected = len(offs) * (2 * tol / beat)
    add("the failure moments do not line up with the beat",
        len(landed) <= math.ceil(expected),
        f"{len(landed)} of {len(offs)} within {tol*1000:.0f}ms of a beat; "
        f"chance alone gives {expected:.1f}")

    # 7. the resets only ever follow a failure
    early = [t for t, _ in RESETS
             if not any(0.0 <= t - e < 30.0 for e in events)]
    add("every reset follows a failure", not early, f"{early}")

    # 8. and nothing fires past the end of the music
    from . import analyze as A
    late = [t for t in events + stutters + [r for r, _ in RESETS]
            if t > A.STAND_IN_AUDIO_END + 0.01]
    add("nothing is scheduled past the end of the music", not late, f"{late}")

    if verbose:
        for name, ok, detail in rows:
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
                  + (f"  -- {detail}" if detail and not ok else ""))
    return rows


def describe() -> str:
    """A one-line summary of the schedule, for --list."""
    return (f"{len(EVENTS)} failure moments, {len(STUTTERS)} clipped syllables, "
            f"{len(RESETS)} resets; all internal")
