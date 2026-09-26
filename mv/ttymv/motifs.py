"""A library of vector motifs, so no two shots have to look alike.

Everything here returns *contours* -- a list of polylines in normalised space
-- which is exactly what ``figures.stroke``/``contours`` consume.  Keeping them
as geometry rather than as drawings means one motif renders identically from a
40-column terminal to a 260-column one.

The acts have signature subjects (the morphing solids, the eggplant, the
heart); these motifs are the other layer, re-staged every couple of lyric
lines so a 30-second act is not one picture held for 30 seconds.
"""

from __future__ import annotations

import math
import random

TAU = 2 * math.pi


def _poly(n: int, f) -> list:
    return [f(TAU * i / n) for i in range(n)]


# --------------------------------------------------------------------------
# motifs
# --------------------------------------------------------------------------

def orbit(spin: float = 0.0) -> list:
    """Three tilted orbits, like an atom diagram."""
    out = []
    for k in range(3):
        tilt = spin + k * math.pi / 3
        c, s = math.cos(tilt), math.sin(tilt)
        pts = []
        for i in range(48):
            a = TAU * i / 48
            x, y = math.cos(a) * 0.92, math.sin(a) * 0.34
            pts.append((x * c - y * s, x * s + y * c))
        out.append(pts)
    return out


def rose(k: int = 5, spin: float = 0.0) -> list:
    pts = []
    for i in range(130):
        a = TAU * i / 130 + spin
        rr = abs(math.cos(k * a / 2)) * 0.95
        pts.append((math.cos(a) * rr, math.sin(a) * rr))
    return [pts]


def lissajous(a: int = 3, b: int = 4, spin: float = 0.0) -> list:
    return [_poly(140, lambda t: (math.sin(a * t + spin) * 0.9,
                                  math.sin(b * t) * 0.9))]


def torus(spin: float = 0.0) -> list:
    out = []
    R, r = 0.72, 0.28
    for k in range(10):
        ph = TAU * k / 10 + spin
        out.append(_poly(36, lambda t, ph=ph: (
            (R + r * math.cos(t)) * math.cos(ph) * 0.98,
            (R + r * math.cos(t)) * math.sin(ph) * 0.98 * 0.42
            + r * math.sin(t) * 0.72)))
    return out


def tunnel(spin: float = 0.0) -> list:
    """Concentric squares receding -- a corridor."""
    out = []
    depth = spin % 1.0
    for k in range(7):
        sc = (k + 1 + depth) / 7.0
        out.append([(-sc, -sc), (sc, -sc), (sc, sc), (-sc, sc)])
    return out


def lattice(spin: float = 0.0) -> list:
    """A 3x3x3 lattice, projected."""
    out = []
    c, s = math.cos(spin), math.sin(spin)
    for plane in range(3):
        for k in range(3):
            line = []
            for j in range(3):
                x = (k - 1) * 0.85
                y = (j - 1) * 0.85
                z = (plane - 1) * 0.85
                x, z = x * c - z * s, x * s + z * c
                f = 1.0 / (1.0 + 0.35 * z)
                line.append((x * f, y * f))
            out.append(line)
    return out


def moire(spin: float = 0.0) -> list:
    """Two line families at a slight angle: interference."""
    out = []
    for fam in (-1, 1):
        ang = spin * fam
        c, s = math.cos(ang), math.sin(ang)
        for k in range(-6, 7):
            off = k / 7.0
            out.append([(off * c - 1.0 * s, off * s + 1.0 * c),
                        (off * c + 1.0 * s, off * s - 1.0 * c)])
    return out


def helix(spin: float = 0.0) -> list:
    out = []
    for phase in (0.0, math.pi):
        out.append(_poly(120, lambda t, p=phase: (
            math.sin(t * 2 + p + spin) * 0.5, (t / TAU) * 1.9 - 0.95)))
    for k in range(9):
        t = TAU * (k + 0.5) / 9
        y = (t / TAU) * 1.9 - 0.95
        out.append([(math.sin(t * 2 + spin) * 0.5, y),
                    (math.sin(t * 2 + math.pi + spin) * 0.5, y)])
    return out


def star(k: int = 7, spin: float = 0.0) -> list:
    pts = []
    for i in range(2 * k):
        a = TAU * i / (2 * k) + spin
        rr = 0.95 if i % 2 == 0 else 0.42
        pts.append((math.cos(a) * rr, math.sin(a) * rr))
    return [pts]


def wavegrid(spin: float = 0.0) -> list:
    """A rack of sine columns, phase-shifted."""
    out = []
    for k in range(13):
        x0 = (k - 6) / 6.5
        out.append(_poly(28, lambda t, x0=x0: (
            x0, math.sin(t * 2 + x0 * 3 + spin) * 0.55 * (1 - abs(x0) * 0.5))))
    return out


def petals(k: int = 8, spin: float = 0.0) -> list:
    out = []
    for j in range(k):
        base = TAU * j / k + spin
        petal = []
        for i in range(18):
            u = i / 17
            a = base + math.sin(u * math.pi) * 0.42 * (1 if j % 2 else -1)
            rr = math.sin(u * math.pi) * 0.92
            petal.append((math.cos(a) * rr, math.sin(a) * rr))
        out.append(petal)
    return out


def knot(spin: float = 0.0) -> list:
    """A trefoil, projected flat."""
    return [_poly(220, lambda t: (
        (math.sin(t + spin) + 2 * math.sin(2 * (t + spin))) / 3.2,
        (math.cos(t + spin) - 2 * math.cos(2 * (t + spin))) / 3.2))]


def spikes(k: int = 20, spin: float = 0.0, lengths=None) -> list:
    out = []
    for i in range(k):
        a = TAU * i / k + spin
        ln = 0.95 if lengths is None else lengths[i % len(lengths)]
        out.append([(math.cos(a) * 0.22, math.sin(a) * 0.22),
                    (math.cos(a) * ln, math.sin(a) * ln)])
    return out


def web(spin: float = 0.0) -> list:
    """Radial spokes with connecting rings."""
    out = []
    spokes = 12
    for i in range(spokes):
        a = TAU * i / spokes + spin
        out.append([(0.0, 0.0), (math.cos(a) * 0.95, math.sin(a) * 0.95)])
    for r in (0.32, 0.62, 0.92):
        out.append(_poly(spokes * 4, lambda t, r=r: (
            math.cos(t) * r, math.sin(t) * r)))
    return out


def galaxy(spin: float = 0.0) -> list:
    """Two spiral arms."""
    out = []
    for arm in (0.0, math.pi):
        out.append(_poly(110, lambda t, arm=arm: (
            math.cos(t * 2.2 + arm + spin) * (t / TAU) * 0.95,
            math.sin(t * 2.2 + arm + spin) * (t / TAU) * 0.95)))
    return out


def constellation(seed: int = 0, spin: float = 0.0) -> list:
    rng = random.Random(seed)
    nodes = [(rng.uniform(-0.95, 0.95), rng.uniform(-0.85, 0.85))
             for _ in range(11)]
    out = [[p] for p in nodes]
    for i in range(len(nodes) - 1):
        out.append([nodes[i], nodes[i + 1]])
    return out


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

# (name, builder) -- the builder takes a phase so the motif can turn with the
# music rather than sitting still.
MOTIFS = [
    ("orbit", orbit),
    ("rose", rose),
    ("lissajous", lissajous),
    ("torus", torus),
    ("tunnel", tunnel),
    ("lattice", lattice),
    ("moire", moire),
    ("helix", helix),
    ("star", star),
    ("wavegrid", wavegrid),
    ("petals", petals),
    ("knot", knot),
    ("spikes", spikes),
    ("web", web),
    ("galaxy", galaxy),
    ("constellation", constellation),
]

NAMES = [n for n, _ in MOTIFS]
_BY_NAME = dict(MOTIFS)


def build(name: str, phase: float, seed: int) -> list:
    """Call a motif builder with whatever arguments it actually accepts.

    The generators are deliberately not forced into one signature, so this
    inspects rather than guessing and catching TypeError -- guessing turned a
    missing parameter into a crash the first time round."""
    import inspect
    fn = _BY_NAME.get(name)
    if fn is None:
        return []
    params = inspect.signature(fn).parameters
    kwargs = {}
    if "spin" in params:
        kwargs["spin"] = phase
    if "seed" in params:
        kwargs["seed"] = seed
    return fn(**kwargs)
