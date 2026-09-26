"""The story: two entities, and the distance between them.

The MV has no narrator and almost no words, so the relationship has to be
readable from the picture alone.  That is what this module is for: *the self*
is the large figure each act draws, and *the other* is a single bright point
with a trail -- a cursor, a firefly, someone else in the room.

Its path is authored, not generated, and every key time is a measured section
cut or a published lyric cue.  Reading only the positions:

    0:00  alone, before anything
    0:04  the other appears at the edge
    0:16  both present, far apart          (title card between them)
    0:29  watching while the self performs (the geometry verse)
    0:44  drifting closer
    0:59  they meet                        (the first chorus)
    1:14  still merged, changing shape     (the menagerie)
    1:20  beginning to pull away
    1:43  leaving
    1:51  gone
    2:57  back -- but outside the heart, free, while the self is trapped inside
    3:12  gone for good

Nothing about that needs a caption.  It is the same trick as the rest of the
MV: put the meaning in the geometry.
"""

from __future__ import annotations

import math
import random

from .figures import blob, phase, ramp

# (t, x, y, alpha) in the same normalised space the figures use.
# +y is down, matching the Braille layer.
PATH: list[tuple[float, float, float, float]] = [
    (0.000, -1.18, -0.10, 0.00),   # not here yet
    (3.200, -1.26, -0.22, 0.00),   # waiting off-stage
    (4.600, -1.26, -0.22, 0.55),   # first contact
    (10.300, -1.20, 0.24, 0.85),
    (16.000, -1.14, 0.00, 0.95),   # title card: both present, far apart
    (29.709, -1.18, 0.46, 1.00),   # watching the performance
    (44.452, -1.16, 0.34, 1.00),
    (52.000, -1.10, 0.16, 1.00),   # closing in, still outside the subject
    (57.600, -0.62, 0.06, 1.00),   # the last step before they meet
    (59.223, 0.00, 0.00, 1.00),    # they meet
    (69.500, 0.00, 0.00, 1.00),
    (71.600, -0.78, -0.48, 1.00),  # and come apart as the chorus ends
    (74.045, -1.02, -0.58, 1.00),  # clear of the frame before the changes
    (80.500, -1.14, -0.38, 1.00),
    # Changing sides, it walks *around* rather than across.  A straight line
    # from here to the far side passes through the middle of the frame, which
    # is exactly where the act's subject is drawn, so the crossing is broken
    # into an arc that stays outside it the whole way.
    (82.000, -0.88, -0.78, 1.00),
    (83.500, -0.45, -0.99, 1.00),
    (85.000, 0.05, -1.03, 1.00),
    (86.500, 0.58, -0.92, 1.00),
    (88.587, 1.14, 0.32, 1.00),
    (95.465, 1.22, -0.22, 1.00),
    (103.489, 1.30, 0.34, 0.95),
    (110.900, 1.62, -0.18, 0.80),  # pulling away
    (113.100, 2.32, 0.22, 0.45),
    (116.400, 3.20, -0.22, 0.00),  # gone
    (177.246, 3.00, 0.00, 0.00),
    (178.600, 1.74, 0.00, 0.00),   # returns, but only outside the heart
    (179.600, 1.70, 0.00, 0.55),
    (191.356, 1.46, 0.10, 0.45),
    (192.100, 1.38, 0.00, 0.00),   # gone for good
    (211.912, 1.32, 0.00, 0.00),
]

# The only stretch where the two of them occupy the same place.  Outside it
# the other one keeps to the frame's edge so it never crosses the act's
# subject; inside it the crossing *is* the story -- this is the chorus where
# they meet, and it is the one moment the walk is allowed through the middle.
MERGE_SPAN = (52.5, 72.5)

TRAIL_STEPS = 34
TRAIL_DT = 1 / 24


def _lerp(a: float, b: float, u: float) -> float:
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    u = u * u * (3 - 2 * u)                 # smoothstep, no corners
    return a + (b - a) * u


def position(t: float) -> tuple[float, float, float]:
    """(x, y, alpha) of the other at time `t`."""
    prev = PATH[0]
    for cur in PATH:
        if t < cur[0]:
            span = cur[0] - prev[0]
            u = (t - prev[0]) / span if span > 1e-6 else 1.0
            return (_lerp(prev[1], cur[1], u),
                    _lerp(prev[2], cur[2], u),
                    _lerp(prev[3], cur[3], u))
        prev = cur
    return prev[1], prev[2], prev[3]


class Companion:
    """The other.  A point, a trail, and where it is standing."""

    def __init__(self):
        self.trail: list[tuple[float, float, float]] = []
        self._last = -1e9

    def update(self, t: float, dt: float) -> tuple[float, float, float]:
        x, y, a = position(t)
        # A slow drift, so it reads as alive rather than as a plotted marker.
        # The wobble goes *around* the scripted point rather than across it,
        # which keeps the walk on the outside of the frame instead of sliding
        # through the middle where the act's subject is drawn.
        if a > 0.01:
            rad = math.hypot(x, y) or 1e-6
            nx, ny = x / rad, y / rad
            wobble = 0.030 * math.sin(t * 0.9)
            drift = 0.018 * math.cos(t * 1.37)
            x += -ny * wobble + nx * drift
            y += nx * wobble + ny * drift
        if t - self._last >= TRAIL_DT:
            self._last = t
            self.trail.append((x, y, a))
            if len(self.trail) > TRAIL_STEPS:
                self.trail.pop(0)
        # the trail thins as the walker fades
        self.trail = [(tx, ty, ta) for (tx, ty, ta) in self.trail
                      if ta > 0.0 or a > 0.0]
        return x, y, a

    def draw(self, bl, cx: float, cy: float, r: float, c, *,
             scale: float = 1.0) -> bool:
        """Draw the trail and the point.  Returns whether it was on screen."""
        x, y, a = position(c.t)
        if a <= 0.01:
            return False
        base = c.pal["accent"]
        hot = c.pal["hot"]
        rng = random.Random(int(c.t * 24))
        n = len(self.trail)
        for i, (tx, ty, ta) in enumerate(self.trail):
            if ta <= 0.01:
                continue
            fade = (i / max(1, n - 1)) ** 2          # older = fainter
            if rng.random() > fade * 0.75 * ta:
                continue
            blob(bl, cx + tx * r * scale, cy + ty * r * scale,
                 _mix(base, c.pal["base"], 0.45 * (1 - fade)), weight=0)
        # Built in three layers so it stays legible on an empty frame and on
        # a busy one: a broken bloom that survives a dark background, a thin
        # reticle so it reads as an instrument rather than a smudge, and a
        # hard core that survives a bright one.
        px, py = cx + x * r * scale, cy + y * r * scale
        core = max(1.0, r * (0.014 + 0.006 * c.rms))
        for k in range(18):
            ang = 2 * math.pi * k / 18 + c.t * 0.35
            d = core * (3.4 + 1.3 * math.sin(c.t * 1.6 + k * 1.7))
            bl.plot(px + math.cos(ang) * d, py + math.sin(ang) * d,
                    _mix(base, c.pal["base"], 0.64 - 0.34 * a))
        ring = core * 2.3
        steps = max(16, int(ring * 7))
        for k in range(steps):
            if (k * 4 // steps) % 2:                 # four gaps in the ring
                continue
            ang = 2 * math.pi * k / steps + c.t * 0.25
            bl.plot(px + math.cos(ang) * ring, py + math.sin(ang) * ring,
                    _mix(base, hot, 0.32 * a))
        blob(bl, px, py, _mix(base, hot, 0.45 + 0.45 * a), weight=int(core))
        bl.plot(px, py, hot)
        return True


def _mix(a: int, b: int, t: float) -> int:
    from .canvas import mix
    return mix(a, b, t)
