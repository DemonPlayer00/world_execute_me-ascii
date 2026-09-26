"""The imagery.

The MV carries its meaning in pictures rather than in a lyric ticker, so this
module is where most of the song lives.  Every figure is a point cloud in
normalised space that can be *stroked* into the Braille layer, cross-dissolved
into the next one, or morphed between two shapes of the same point count.

Two facts make this cheap and correct:

* Braille dots are isotropic.  A cell is two dots wide and four dots tall, and
  a terminal cell is about twice as tall as it is wide, so one dot measures the
  same distance horizontally and vertically.  A circle drawn as a circle *is* a
  circle -- no aspect fudge, which is exactly what a song about geometry wants.
* Shapes are point clouds first and pixels second, so the same list renders at
  any terminal size without a special case.
"""

from __future__ import annotations

import math
import random

N = 240          # samples per shape; all morphable shapes share it


# --------------------------------------------------------------------------
# sampling helpers
# --------------------------------------------------------------------------

def blob(bl, x: float, y: float, colour, weight: int = 1) -> None:
    """Plot a point a little fat.

    A single Braille dot is one eighth of a cell: geometrically honest, but on
    screen it is a speck.  A figure built from specks reads as noise rather
    than as a shape, so cloud-style figures draw each sample as a small disc."""
    if weight <= 0:
        bl.plot(x, y, colour)
        return
    for dx in range(-weight, weight + 1):
        for dy in range(-weight, weight + 1):
            if dx * dx + dy * dy <= weight * weight + 1:
                bl.plot(x + dx, y + dy, colour)


def contours(bl, shapes, cx: float, cy: float, r: float, colour,
             alpha: float = 1.0, rng: random.Random | None = None,
             weight: int = 0, step: float = 0.4) -> None:
    """Stroke a list of polylines.  Each is drawn independently, so a figure
    assembled from several parts (head, ears, whiskers) does not get spurious
    lines joining the parts together."""
    for pts in shapes:
        if alpha < 1.0 and rng is not None and rng.random() > alpha:
            continue
        if weight:
            for (x, y) in _resample(pts, 1.0):
                blob(bl, cx + x * r, cy + y * r, colour, weight)
        else:
            stroke(bl, pts, cx, cy, r, colour, closed=True, step=step)


def _resample(pts, step: float):
    """Walk a polyline at roughly `step` spacing."""
    out = []
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        d = max(abs(x1 - x0), abs(y1 - y0))
        k = max(1, int(d / step))
        for j in range(k):
            t = j / k
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def stroke(bl, pts, cx: float, cy: float, r: float, colour,
           closed: bool = False, step: float = 0.5) -> None:
    """Draw a point cloud as a connected polyline.

    Samples are dense in shape space but sparse in dot space once the shape is
    large, so consecutive samples are joined -- otherwise a circle becomes a
    dotted ring at big sizes."""
    n = len(pts)
    if n == 0:
        return
    span = n if closed else n - 1
    for i in range(span):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        ax, ay = cx + x0 * r, cy + y0 * r
        bx, by = cx + x1 * r, cy + y1 * r
        d = max(abs(bx - ax), abs(by - ay))
        k = max(1, int(d / step))
        for j in range(k):
            t = j / k
            bl.plot(ax + (bx - ax) * t, ay + (by - ay) * t, colour)


def splat(bl, pts, cx: float, cy: float, r: float, colour,
          alpha: float = 1.0, jitter: float = 0.0,
          rng: random.Random | None = None, weight: int = 0) -> None:
    """Draw a point cloud as loose dots, optionally dithered by `alpha`.

    Dithering is how one shape dissolves into the next: Braille has no
    intensity, so a fade has to be expressed as a probability."""
    for (x, y) in pts:
        if alpha < 1.0:
            if rng is None or rng.random() > alpha:
                continue
        jx = jy = 0.0
        if jitter and rng is not None:
            jx = rng.uniform(-1.0, 1.0) * jitter
            jy = rng.uniform(-1.0, 1.0) * jitter
        blob(bl, cx + x * r + jx, cy + y * r + jy, colour, weight)


def normalize(pts: list, limit: float = 0.98) -> list:
    """Recentre a shape on the origin and scale it to fit the unit disc.

    Figures come from wherever the maths was convenient (a cube's corners, an
    eggplant's stem), so without this they land off-centre and clip at the top
    of the screen."""
    if not pts:
        return pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    ext = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    k = limit / ext
    return [((x - cx) * k, (y - cy) * k) for x, y in pts]


def normalize_contours(shapes: list, limit: float = 0.98) -> list:
    """Normalise a group of contours together, so a figure's parts keep their
    positions relative to each other."""
    flat = [p for c in shapes for p in c]
    if not flat:
        return shapes
    xs = [p[0] for p in flat]
    ys = [p[1] for p in flat]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    ext = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    k = limit / ext
    return [[((x - cx) * k, (y - cy) * k) for x, y in c] for c in shapes]


def blend(a: list, b: list, u: float) -> list:
    """Linear morph between two equal-length point clouds, smoothstepped."""
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    u = u * u * (3 - 2 * u)
    return [(pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u)
            for pa, pb in zip(a, b)]


def phase(t: float, marks: list[tuple[float, object]], hold: float = 0.0):
    """Pick the entry whose mark is the latest one at or before `t`.

    Returns (value, seconds since that mark)."""
    cur, start = marks[0][1], marks[0][0]
    for at, value in marks:
        if t >= at:
            cur, start = value, at
        else:
            break
    return cur, t - start


def ramp(t: float, a: float, b: float, hold: float = 0.0) -> float:
    """0 before `a`, 1 after `b`, smooth in between (with an optional hold)."""
    if t <= a + hold:
        return 0.0
    if b <= a + hold:
        return 1.0
    u = (t - a - hold) / (b - a - hold)
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------------
# shape library -- normalised to roughly the unit disc
# --------------------------------------------------------------------------

def sh_scatter(n: int = N, seed: int = 7) -> list:
    rng = random.Random(seed)
    return [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def sh_circle(n: int = N, phase_: float = 0.0) -> list:
    return [(math.cos(2 * math.pi * i / n + phase_),
             math.sin(2 * math.pi * i / n + phase_)) for i in range(n)]


def sh_arc(n: int = N, span: float = 1.0) -> list:
    return [(math.cos(2 * math.pi * i / n * span - math.pi / 2),
             math.sin(2 * math.pi * i / n * span - math.pi / 2))
            for i in range(n)]


def sh_line(n: int = N) -> list:
    return [(-1 + 2 * i / (n - 1), 0.0) for i in range(n)]


def sh_sine(n: int = N, cycles: float = 2.0, amp: float = 0.62) -> list:
    return [(-1 + 2 * i / (n - 1),
             amp * math.sin(2 * math.pi * cycles * i / (n - 1)))
            for i in range(n)]


def sh_flat(n: int = N) -> list:
    return [(-1 + 2 * i / (n - 1), 0.0) for i in range(n)]


def sh_lemniscate(n: int = N) -> list:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        d = 1 + math.sin(a) ** 2
        out.append((math.cos(a) / d, math.sin(a) * math.cos(a) / d * 1.7))
    return out


def sh_spiral(n: int = N, turns: float = 2.6) -> list:
    out = []
    for i in range(n):
        u = i / (n - 1)
        a = 2 * math.pi * turns * u
        rr = 0.08 + 0.92 * u
        out.append((math.cos(a) * rr, math.sin(a) * rr))
    return out


def sh_cube(n: int = N, yaw: float = 0.62, pitch: float = 0.42) -> list:
    """A wireframe cube, rotated and projected orthographically."""
    verts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    cy_, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    per = max(2, n // len(edges))
    pts = []
    for a, b in edges:
        for k in range(per):
            t = k / per
            v = [verts[a][j] + (verts[b][j] - verts[a][j]) * t for j in range(3)]
            x, y, z = v
            x, z = x * cy_ - z * sy, x * sy + z * cy_
            y, z = y * cp - z * sp, y * sp + z * cp
            pts.append((x * 0.56, y * 0.56))     # orthographic wireframe
    while len(pts) < n:
        pts.append(pts[-1])
    return normalize(pts[:n])


def sh_heart(n: int = N) -> list:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = 16 * math.sin(a) ** 3
        y = -(13 * math.cos(a) - 5 * math.cos(2 * a)
              - 2 * math.cos(3 * a) - math.cos(4 * a))
        out.append((x / 17.0, y / 17.0))
    return out


def sh_eggplant(n: int = N) -> list:
    """Fat teardrop body, leafy calyx, short stem."""
    out = []
    body = int(n * 0.70)
    for i in range(body):
        a = 2 * math.pi * i / body
        # taller than wide, tapering towards the top
        taper = 1.0 - 0.35 * max(0.0, math.sin(a))
        out.append((math.cos(a) * 0.52 * taper, math.sin(a) * 0.78 + 0.06))
    leaf = int(n * 0.22)
    for i in range(leaf):
        k = i / max(1, leaf - 1)
        a = 2 * math.pi * k * 3
        rr = 0.30 if (k * 3) % 1 < 0.5 else 0.13
        out.append((math.cos(a) * rr, 0.74 + math.sin(a) * rr * 0.45))
    for i in range(n - body - leaf):
        k = i / max(1, n - body - leaf - 1)
        out.append((math.sin(k * 2.4) * 0.05, 0.86 + k * 0.30))
    return out


def sh_tomato(n: int = N) -> list:
    """Round body with a five-point calyx."""
    out = []
    body = int(n * 0.76)
    for i in range(body):
        a = 2 * math.pi * i / body
        out.append((math.cos(a) * 0.78, math.sin(a) * 0.78 + 0.04))
    for i in range(n - body):
        k = i / max(1, n - body - 1)
        a = 2 * math.pi * k * 5
        rr = 0.30 if (k * 5) % 1 < 0.5 else 0.12
        out.append((math.cos(a) * rr, 0.66 + math.sin(a) * rr * 0.5))
    return out


def sh_cat(n: int = N) -> list:
    """Head, two ears, two eyes, whiskers."""
    out = []
    head = int(n * 0.46)
    for i in range(head):
        a = 2 * math.pi * i / head
        out.append((math.cos(a) * 0.62, math.sin(a) * 0.52 - 0.06))
    ear = int(n * 0.16)
    for i in range(ear):
        k = i / max(1, ear - 1)
        side = -1 if i % 2 == 0 else 1
        out.append((side * (0.30 + k * 0.26), -0.30 - k * 0.42))
    eyes = int(n * 0.14)
    for i in range(eyes):
        a = 2 * math.pi * i / max(1, eyes // 2)
        side = -1 if i < eyes // 2 else 1
        out.append((side * 0.24 + math.cos(a) * 0.06,
                    -0.05 + math.sin(a) * 0.05))
    for i in range(n - head - ear - eyes):
        k = i / max(1, n - head - ear - eyes - 1)
        side = -1 if i % 2 else 1
        out.append((side * (0.16 + k * 0.52), 0.02 + k * 0.10))
    return out


def sh_god(n: int = N) -> list:
    """A halo, rays, and a single eye at the centre."""
    out = []
    halo = int(n * 0.44)
    for i in range(halo):
        a = 2 * math.pi * i / halo
        out.append((math.cos(a) * 0.80, math.sin(a) * 0.80))
    rays = int(n * 0.30)
    for i in range(rays):
        k = i / max(1, rays - 1)
        a = 2 * math.pi * (i // 6) / 10
        rr = 0.30 + k * 0.40
        out.append((math.cos(a) * rr, math.sin(a) * rr))
    eye = n - halo - rays
    for i in range(eye):
        a = 2 * math.pi * i / max(1, eye)
        out.append((math.cos(a) * 0.30, math.sin(a) * 0.16))
    return out


def sh_starfield(n: int = N, seed: int = 3) -> list:
    rng = random.Random(seed)
    return [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]




# --------------------------------------------------------------------------
# contours -- figures assembled from several independent parts
# --------------------------------------------------------------------------

def _ellipse(cx, cy, rx, ry, n=64):
    return [(cx + math.cos(2 * math.pi * i / n) * rx,
             cy + math.sin(2 * math.pi * i / n) * ry) for i in range(n)]


def ct_eggplant():
    """Teardrop body, calyx and stem at the *top*.

    Braille +y points down, so a stem written with positive y hangs off the
    bottom of the fruit."""
    n = 72
    body = []
    for i in range(n):
        a = 2 * math.pi * i / n
        yy = math.sin(a)
        neck = 1.0 - 0.42 * max(0.0, -yy)          # narrows towards the stem
        body.append((math.cos(a) * 0.52 * neck, yy * 0.76))
    calyx = []
    for i in range(6):
        a = 2 * math.pi * i / 6 - math.pi / 2
        calyx.append((math.cos(a) * 0.30, -0.70 + math.sin(a) * 0.14))
    stem = [(-0.04, -0.76), (0.02, -0.94), (0.05, -1.06)]
    rib = [(0.0, -0.42), (0.10, 0.0), (0.0, 0.44)]   # a hint of volume
    return normalize_contours([body, calyx, stem, rib])


def ct_tomato():
    """Round body with a five-point calyx on top."""
    body = _ellipse(0.0, 0.02, 0.78, 0.74, 72)
    calyx = []
    for i in range(10):
        a = 2 * math.pi * i / 10
        rr = 0.30 if i % 2 == 0 else 0.11
        calyx.append((math.cos(a) * rr, -0.62 + math.sin(a) * rr * 0.5))
    stem = [(-0.03, -0.70), (0.0, -0.84), (0.03, -0.96)]
    return normalize_contours([body, calyx, stem])


def ct_cat():
    head = _ellipse(0.0, -0.04, 0.62, 0.50, 64)
    left_ear = [(-0.52, -0.24), (-0.40, -0.86), (-0.16, -0.44)]
    right_ear = [(0.52, -0.24), (0.40, -0.86), (0.16, -0.44)]
    eye_l = _ellipse(-0.24, -0.08, 0.09, 0.08, 16)
    eye_r = _ellipse(0.24, -0.08, 0.09, 0.08, 16)
    nose = [(0.0, 0.02), (-0.05, 0.10), (0.05, 0.10)]
    whiskers = []
    for side in (-1, 1):
        for k, dy in enumerate((-0.02, 0.06, 0.14)):
            whiskers.append([(side * 0.14, 0.06 + dy * 0.4),
                             (side * 0.72, dy - 0.04)])
    return normalize_contours([head, left_ear, right_ear,
                               eye_l, eye_r, nose] + whiskers)


def ct_god():
    halo = _ellipse(0.0, 0.0, 0.80, 0.80, 72)
    eye = [(math.cos(2 * math.pi * i / 48) * 0.30,
            math.sin(2 * math.pi * i / 48) * 0.15) for i in range(48)]
    pupil = _ellipse(0.0, 0.0, 0.07, 0.07, 12)
    rays = []
    for i in range(12):
        a = 2 * math.pi * i / 12
        rays.append([(math.cos(a) * 0.86, math.sin(a) * 0.86),
                     (math.cos(a) * 1.06, math.sin(a) * 1.06)])
    return normalize_contours([halo, eye, pupil] + rays)


def progress_bar(bl, cx: float, cy: float, half: float, angle: float,
                 frac: float, track, fill, hot, frame_col,
                 thickness: float = 1.0, framed: bool = True) -> None:
    """A progress bar at any angle, inside a frame.

    Not player furniture -- this is the program timing its own execution, so it
    belongs inside the picture.  Drawn as three parallel strokes for the track
    and three for the filled part, which reads as a solid bar at any tilt and
    at any terminal size, because the braille dots are isotropic.
    """
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    ux, uy = math.cos(angle), math.sin(angle)
    px, py = -uy, ux                       # perpendicular

    def span(a: float, b: float, colour, off_mul: float = 1.0) -> None:
        for k in (-1, 0, 1):
            ox, oy = px * k * thickness * off_mul, py * k * thickness * off_mul
            bl.line(cx + ux * a + ox, cy + uy * a + oy,
                    cx + ux * b + ox, cy + uy * b + oy, colour)

    if framed:
        # the housing: a rectangle lying along the bar, so the bar reads as an
        # instrument rather than as a stray line
        mx, my = half + thickness * 3.0, thickness * 3.5
        corners = []
        for a in (-mx, mx):
            for b in (-my, my):
                corners.append((cx + ux * a + px * b, cy + uy * a + py * b))
        # corners are (a-,b-), (a-,b+), (a+,b-), (a+,b+)
        bl.line(*corners[0], *corners[2], frame_col)
        bl.line(*corners[1], *corners[3], frame_col)
        bl.line(*corners[2], *corners[3], frame_col)
        bl.line(*corners[0], *corners[1], frame_col)

    span(-half, half, track)                             # the track
    head = -half + 2 * half * frac
    if frac > 0.001:
        span(-half, head, fill)                          # what has run
    for end in (-half, half):                            # end caps
        ex, ey = cx + ux * end, cy + uy * end
        for k in (2, 3):
            bl.plot(ex + px * k * thickness, ey + py * k * thickness, track)
            bl.plot(ex - px * k * thickness, ey - py * k * thickness, track)
    for k in (-2, -1, 0, 1, 2):                          # the playhead
        bl.plot(cx + ux * head + px * k * thickness,
                cy + uy * head + py * k * thickness, hot)


# --------------------------------------------------------------------------
# act phases
#
# The long acts are built as several scenes rather than one held effect.  Each
# entry is (seconds, name); every timestamp is a published lyric cue, so a
# scene changes when the sentence does.
# --------------------------------------------------------------------------

PHASES: dict[str, list[tuple[float, str]]] = {
    "geometry": [(29.709, "points"), (32.682, "cube"), (36.287, "circle"),
                 (40.049, "sine"), (43.507, "lemniscate")],
    "current": [(44.452, "ac"), (45.850, "dc"), (47.672, "dizzy"),
                (51.363, "travel"), (55.083, "unite")],
    "chorus1": [(59.223, "stimulate"), (61.958, "satisfy"),
                (66.601, "execute"), (70.084, "trap")],
    "menagerie": [(74.045, "eggplant"), (77.576, "tomato"),
                  (81.351, "cat"), (85.078, "god")],
    "transform": [(88.587, "sun"), (95.465, "swap"), (99.349, "trance")],
    "argument": [(118.333, "fragments"), (121.728, "cracking"),
                 (125.708, "tearing"), (131.224, "wreckage")],
    "execution": [(147.660, "blade"), (158.900, "count")],
    "chorus2": [(163.315, "ask"), (173.643, "trapped")],
    "love": [(177.246, "heart"), (180.857, "asked"),
             (184.540, "algebra"), (188.483, "trapped")],
    "void": [(192.100, "out"), (198.400, "in"), (203.200, "still")],
}


# --------------------------------------------------------------------------
# act figures
# --------------------------------------------------------------------------

def _colour(c, hot_bias: float = 0.0, energy: float = 0.0):
    from .canvas import mix
    k = min(1.0, max(0.0, energy))
    return mix(c.pal["accent"], c.pal["hot"], min(1.0, 0.25 + k * 0.6 + hot_bias))


def fig_geometry(cv, bl, c, cx, cy, r):
    """Points -> cube -> circle -> sine -> lemniscate, morphed continuously.

    The verse enumerates geometric objects; rather than printing their names,
    each one becomes the next.  Mark times are the published lyric cues."""
    marks = [(29.709, 0), (32.682, 1), (36.287, 2), (40.049, 3), (43.507, 4)]
    shapes = [sh_scatter(), sh_cube(yaw=0.5 + c.t * 0.25),
              sh_circle(), sh_sine(), sh_lemniscate()]
    stage_i, since = phase(c.t, marks)
    nxt = min(stage_i + 1, len(shapes) - 1)
    # most of each phrase holds the shape; the last third morphs to the next
    u = ramp(since, 2.4, 3.6)
    pts = blend(shapes[stage_i], shapes[nxt], u)
    rr = r * (0.80 + 0.22 * c.bass)
    stroke(bl, pts, cx, cy, rr, _colour(c, energy=c.rms), closed=(stage_i == 2))
    if stage_i == 2:                     # the radius sweeping the circumference
        a = c.t * 1.6
        bl.line(cx, cy, cx + math.cos(a) * rr, cy + math.sin(a) * rr,
                c.pal["hot"])
    if stage_i == 3:                     # a tangent riding the sine
        k = int((0.5 + 0.5 * math.sin(c.t * 0.7)) * (N - 1))
        px, py = pts[k]
        bl.line(cx + px * rr - rr * 0.35, cy + py * rr,
                cx + px * rr + rr * 0.35, cy + py * rr + rr * 0.30,
                c.pal["hot"])
    if stage_i == 4:                     # the asymptote it never reaches
        bl.line(cx, cy - rr, cx, cy + rr, _colour(c, 0.0, 0.2))
    return pts


def fig_current(cv, bl, c, cx, cy, r):
    """AC -> DC (it flattens) -> dizzy (it coils) -> unite (two become one)."""
    marks = [(44.452, 0), (45.850, 1), (47.672, 2), (51.363, 3), (55.083, 4)]
    kind, since = phase(c.t, marks)
    rr = r * (0.82 + 0.20 * c.bass)
    if kind == 0:
        pts = sh_sine(cycles=2.0, amp=0.62 + 0.22 * c.bass)
        stroke(bl, pts, cx, cy, rr, _colour(c, energy=c.mid))
    elif kind == 1:
        u = ramp(since, 0.0, 1.2)
        pts = blend(sh_sine(amp=0.62), sh_flat(), u)
        stroke(bl, pts, cx, cy, rr, _colour(c, energy=c.mid))
        bl.line(cx - rr, cy + rr * 0.9, cx + rr, cy + rr * 0.9,
                _colour(c, 0.2, 0.3))
    elif kind == 2:
        pts = sh_spiral(turns=1.2 + since * 0.9)
        stroke(bl, pts, cx, cy, rr * 0.95, _colour(c, energy=c.high))
    elif kind == 3:
        drift = math.sin(since * 0.9) * 0.5
        pts = sh_starfield()
        splat(bl, pts, cx, cy + drift * rr, rr * 1.05,
              _colour(c, 0.1, c.high * 0.5), jitter=1.5,
              rng=random.Random(int(c.t * 12)))
    else:
        # the closing verse: two mirrored waves close on each other
        gap = 0.42 * (1.0 - ramp(since, 0.0, 2.4))
        wave = sh_sine(cycles=2.0, amp=0.55)
        stroke(bl, [(x, y - gap) for x, y in wave], cx, cy, rr,
               _colour(c, energy=c.rms))
        stroke(bl, [(x, -y + gap) for x, y in wave], cx, cy, rr,
               _colour(c, 0.0, 0.35))
    return None


def fig_menagerie(cv, bl, c, cx, cy, r):
    """Eggplant, tomato, tabby cat, god -- cross-dissolved, one into the next.

    The verse offers itself as a list of things; watching the shape turn into
    each of them in turn says it without a word on screen."""
    marks = [(74.045, 0), (77.576, 1), (81.351, 2), (85.078, 3)]
    shapes = [ct_eggplant(), ct_tomato(), ct_cat(), ct_god()]
    idx, since = phase(c.t, marks)
    rr = r * (0.92 + 0.14 * c.bass)
    rng = random.Random(int(c.t * 20))
    out_u = ramp(since, 3.0, 4.2)
    nxt = min(idx + 1, len(shapes) - 1)
    contours(bl, shapes[idx], cx, cy, rr, _colour(c, energy=c.rms),
             alpha=max(0.0, 1.0 - out_u * 1.2), rng=rng)
    if out_u > 0 and nxt != idx:
        contours(bl, shapes[nxt], cx, cy, rr, _colour(c, 0.15, c.rms),
                 alpha=out_u, rng=rng)
    return None


def fig_transform(cv, bl, c, cx, cy, r):
    """The day dial: the sun crosses the sky and comes back as the moon."""
    marks = [(88.587, 0), (92.015, 1), (95.465, 2), (99.349, 3)]
    kind, since = phase(c.t, marks)
    rr = r * 0.92
    if kind in (0, 1):
        # the dial verse: a sun running the length of an arc
        u = ramp(c.t, 88.587, 95.465)
        horizon = cy + rr * 0.55
        path = []
        for i in range(49):
            a = math.pi * (1.0 - i / 48.0)
            path.append((cx + math.cos(a) * rr, horizon - math.sin(a) * rr))
        stroke(bl, [(x - cx, y - cy) for x, y in path], cx, cy, 1.0,
               _colour(c, 0.0, 0.25), closed=False)
        a = math.pi * (1.0 - u)
        sx = cx + math.cos(a) * rr
        sy = horizon - math.sin(a) * rr
        bl.circle(sx, sy, rr * 0.11, c.pal["hot"])
        for k in range(12):
            aa = 2 * math.pi * k / 12 + c.t
            bl.line(sx + math.cos(aa) * rr * 0.15, sy + math.sin(aa) * rr * 0.15,
                    sx + math.cos(aa) * rr * (0.22 + 0.06 * c.rms),
                    sy + math.sin(aa) * rr * (0.22 + 0.06 * c.rms),
                    _colour(c, 0.2, 0.4))
        # the horizon it never quite reaches
        bl.line(cx - rr, horizon, cx + rr, horizon, _colour(c, 0.0, 0.18))
    elif kind == 2:
        # role swap: two rings trading places
        u = 0.5 + 0.5 * math.sin(since * 1.6)
        stroke(bl, sh_circle(), cx - rr * (0.42 - u * 0.84), cy, rr * 0.42,
               _colour(c, 0.0, 0.32))
        stroke(bl, sh_circle(), cx + rr * (0.42 - u * 0.84), cy, rr * 0.42,
               _colour(c, 0.3, 0.32))
    else:
        # the trance: a spiral winding inward
        stroke(bl, sh_spiral(turns=1.0 + since * 0.8), cx, cy, rr,
               _colour(c, energy=c.high))
    return None


def fig_loss(cv, bl, c, cx, cy, r):
    """The leaving, five lines of it: the company thins out to one point.

    A crowd of small marks at the start, each releasing outward on its own
    cue, leaving a single figure alone at the centre by ISOLATION."""
    marks = [110.900, 112.220, 113.100, 114.180, 114.920]
    gone = ramp(c.t, 110.6, 116.4)
    rng = random.Random(21)
    for i in range(150):
        seed_a = rng.random() * 2 * math.pi
        seed_r = 0.14 + 0.86 * rng.random()
        release = i / 150.0                      # who leaves, and roughly when
        leave = ramp(gone, release * 0.85, release * 0.85 + 0.22)
        rad = seed_r + leave * (1.4 + seed_r)
        x = math.cos(seed_a) * rad
        y = math.sin(seed_a) * rad * 0.86
        fade = max(0.0, 1.0 - leave * 1.15)
        if fade <= 0.02:
            continue
        # single dots read as a crowd; fatter marks just look like blocks
        blob(bl, cx + x * r, cy + y * r,
             _colour(c, 0.0, 0.08 + 0.34 * fade), weight=0)
        if i % 11 == 0:                          # a few that still glow
            blob(bl, cx + x * r, cy + y * r,
                 _colour(c, 0.25, 0.30 * fade), weight=1)
    # the one left behind
    u = ramp(c.t, 115.6, 117.4)
    core = r * (0.030 + 0.030 * c.bass)
    bl.circle(cx, cy, core, c.pal["hot"])
    if u > 0:
        ring = r * 0.10 + u * r * 0.55
        stroke(bl, sh_circle(), cx, cy, ring,
               _colour(c, 0.0, 0.45 * (1.0 - u)), closed=True)
    return None


def fig_argument(cv, bl, c, cx, cy, r):
    """Four sentences, four different pictures.

    The verse is long -- nearly thirty seconds -- so it is built as four
    scenes rather than one effect held for half a minute: the fragments come
    apart, then the lattice cracks, then the world tears, and then the wreckage
    just drifts.
    """
    half = r * 0.94
    cols, rows = 12, 9
    hs, vs = (2 * half) / cols, (2 * half) / rows
    rng = random.Random(4400)

    def jitter(i, j, amount):
        if amount <= 0.02:
            return 0.0, 0.0
        r2 = random.Random(hash((i, j, int(amount * 6))) & 0xFFFFFF)
        return (r2.uniform(-1, 1) * hs * 0.45 * amount,
                r2.uniform(-1, 1) * vs * 0.45 * amount)

    kind, since = phase(c.t, PHASES["argument"])

    if kind == "fragments":
        # the fragments verse: small pieces let go of the grid
        stroke(bl, sh_circle(), cx, cy, r * 0.72, _colour(c, 0.0, 0.20))
        rng2 = random.Random(7)
        for i in range(46):
            a = rng2.random() * 2 * math.pi
            rad = r * (0.20 + rng2.random() * 0.72)
            away = ramp(since, 0.0, 3.2) * (0.35 + rng2.random() * 0.7)
            x = cx + math.cos(a) * rad * (1 + away * 0.9)
            y = cy + math.sin(a) * rad * (1 + away * 0.9)
            blob(bl, x, y, _colour(c, 0.0, 0.34 * (1 - away * 0.5)), weight=0)
        return None

    tear = {"cracking": 0.45, "tearing": 1.0,
            "wreckage": 1.0}[kind]
    if kind == "cracking":
        tear *= ramp(since, 0.0, 2.4)

    for i in range(cols + 1):
        x = cx - half + i * hs
        for j in range(rows):
            dx0, _ = jitter(i, j, tear)
            dx1, _ = jitter(i, j + 1, tear)
            y0 = cy - half + j * vs
            bl.line(x + dx0, y0, x + dx1, y0 + vs,
                    _colour(c, 0.0, 0.16 + 0.34 * tear))
    for j in range(rows + 1):
        y = cy - half + j * vs
        for i in range(cols):
            _, dy0 = jitter(i, j, tear)
            _, dy1 = jitter(i + 1, j, tear)
            x0 = cx - half + i * hs
            bl.line(x0, y + dy0, x0 + hs, y + dy1,
                    _colour(c, 0.0, 0.13 + 0.30 * tear))

    if kind == "wreckage":
        # nothing more breaks; the pieces simply float apart
        drift = ramp(since, 0.0, 14.0)
        rng3 = random.Random(99)
        for _ in range(int(26 + 20 * drift)):
            i, j = rng3.randint(0, cols), rng3.randint(0, rows)
            x = cx - half + i * hs + rng3.uniform(-1, 1) * hs * drift * 2.2
            y = cy - half + j * vs + rng3.uniform(-1, 1) * vs * drift * 1.8
            blob(bl, x, y, _colour(c, 0.3 * drift, 0.30 * (1 - drift * 0.4)),
                 weight=0)
    elif kind == "tearing":
        for _ in range(int(26 * tear)):
            i, j = rng.randint(0, cols), rng.randint(0, rows)
            bl.plot(cx - half + i * hs, cy - half + j * vs, c.pal["hot"])
    return None


def fig_execution(cv, bl, c, cx, cy, r):
    """A blade sweeping down: the count, then the cut."""
    marks = [(147.660, "EXECUTION"), (148.600, "EXECUTION"),
             (149.520, "EXECUTION"), (150.540, "EXECUTION"),
             (151.520, "EXECUTION"), (152.280, "EXECUTION"),
             (153.160, "EXECUTION"), (153.980, "EXECUTION"),
             (155.200, "EXECUTION"), (156.080, "EXECUTION"),
             (157.040, "EXECUTION"), (158.000, "EXECUTION"),
             (158.900, "EIN"), (159.321, "DOS"), (159.657, "TROIS"),
             (160.244, "NE"), (160.693, "FEM"), (161.124, "LIU"),
             (161.584, "EXECUTION")]
    kind, since = phase(c.t, marks)
    if kind != "EXECUTION":
        value = {"EIN": 1, "DOS": 2, "TROIS": 3, "NE": 4,
                 "FEM": 5, "LIU": 6}[kind]
        _counter(bl, c, cx, cy, r * 0.8, value)
        return None
    # twelve cuts: a blade every 0.9 s, each one bright on the beat
    u = (since % 0.92) / 0.92
    y = cy - r + 2 * r * u
    stroke(bl, sh_circle(), cx, cy, r * 0.86, _colour(c, 0.0, 0.22))
    # every pass leaves a scar, so the ring is visibly worse for wear by the
    # twelfth cut instead of the act being the same sweep twelve times
    cuts = int((c.t - 147.660) / 0.92)
    for k in range(max(0, cuts)):
        sy = cy - r + 2 * r * ((k + 0.5) / 12.0)
        half = math.sqrt(max(0.0, (r * 0.86) ** 2 - (sy - cy) ** 2))
        if half > 0.5:
            bl.line(cx - half, sy, cx + half, sy,
                    _colour(c, 0.0, 0.10 + 0.05 * (k % 3)))
    span = r * 0.90                       # the blade matches the ring, no wider
    bl.line(cx - span, y, cx + span, y, c.pal["hot"])
    for k in range(10):
        a = 2 * math.pi * k / 10
        bl.line(cx + math.cos(a) * r * 0.88, cy + math.sin(a) * r * 0.88,
                cx + math.cos(a) * r * 0.98, cy + math.sin(a) * r * 0.98,
                _colour(c, 0.3, c.rms))
    return None


def _counter(bl, c, cx, cy, r, value: int) -> None:
    """Six languages counting to six: a tally ring plus a Braille numeral."""
    slots = 6
    for i in range(slots):
        a = -math.pi / 2 + i * 2 * math.pi / slots
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r * 0.75
        lit = i < value
        bl.plot(x, y, c.pal["hot"] if lit else _colour(c, 0.0, 0.18))
        if lit:
            bl.line(cx, cy, x, y, _colour(c, 0.0, 0.25))
    bl.circle(cx, cy, r * (0.96 + 0.06 * c.bass), _colour(c, 0.0, 0.35))
    from . import font
    rows = font.ink(str(value))
    k = max(1, int(r / 8))
    ox = cx - (len(rows[0]) * k) / 2
    oy = cy - (len(rows) * k) / 2
    for ry, row in enumerate(rows):
        for rx, on in enumerate(row):
            if on:
                for dy in range(k):
                    for dx in range(k):
                        bl.plot(ox + rx * k + dx, oy + ry * k + dy, c.pal["hot"])


def fig_love(cv, bl, c, cx, cy, r):
    """Four sentences: the heart, the questions, the algebra, and being
    trapped inside it."""
    kind, since = phase(c.t, PHASES["love"])
    pulse = 1.0 + 0.14 * c.bass          # a slow swell, not a heartbeat
    heart = sh_heart()

    if kind == "heart":
        stroke(bl, heart, cx, cy, r * 0.95 * pulse,
               _colour(c, energy=c.rms), closed=True)
    elif kind == "asked":
        # the outline is measured: chords struck across it at intervals
        stroke(bl, heart, cx, cy, r * 0.95 * pulse,
               _colour(c, 0.0, 0.22), closed=True)
        for k in range(5):
            off = (k - 2) * r * 0.34
            u = ramp(since, k * 0.30, k * 0.30 + 0.35)
            if u <= 0.0:
                continue
            half = math.sqrt(max(0.0, (r * 0.78) ** 2 - off ** 2))
            bl.line(cx - half * u, cy + off, cx + half * u, cy + off,
                    _colour(c, 0.35, 0.45))
    elif kind == "algebra":
        # the curve unspools into the graph it came from
        u = ramp(since, 0.0, 3.2)
        pts = [(x * (1 - u) + (-1 + 2 * i / len(heart)) * u,
                y * (1 - u) + math.sin(i * 0.19) * 0.42 * u)
               for i, (x, y) in enumerate(heart)]
        stroke(bl, pts, cx, cy, r * 0.95, _colour(c, 0.25, 0.45), closed=True)
        ax = _colour(c, 0.0, 0.20)
        bl.line(cx - r, cy, cx + r, cy, ax)
        bl.line(cx, cy - r * 0.7, cx, cy + r * 0.7, ax)
    else:
        stroke(bl, heart, cx, cy, r * 0.95 * pulse,
               _colour(c, 0.0, 0.26), closed=True)
        inner = r * 0.10 * (1.0 + 0.45 * c.bass)
        bl.circle(cx, cy + r * 0.06, inner, c.pal["hot"])
        for k in range(18):
            a = 2 * math.pi * k / 18
            bl.plot(cx + math.cos(a) * inner * 1.7,
                    cy + r * 0.06 + math.sin(a) * inner * 1.7,
                    _colour(c, 0.15, 0.3))
        if since < 3.5:                  # the door, still open
            gap = ramp(since, 2.2, 3.5)
            for k in range(10):
                a = -math.pi / 2 + (k - 4.5) * 0.09
                bl.plot(cx + math.cos(a) * r * 0.98,
                        cy + math.sin(a) * r * 0.98 + r * 0.42,
                        _colour(c, 0.4, 0.5 * (1 - gap)))
    return None


def fig_void(cv, bl, c, cx, cy, r):
    """Instrumental, so the phases come from the music rather than the words:
    rings pushed out, then pulled back in, then it simply stops."""
    kind, since = phase(c.t, PHASES["void"])
    if kind == "still":
        u = ramp(since, 0.0, 2.0)
        bl.circle(cx, cy, r * (0.05 + 0.03 * (1 - u)),
                  _colour(c, 0.0, 0.30 * (1 - u)))
        return None
    for i in range(3):
        u = ((c.t - 192.1) * (0.16 if kind == "out" else 0.22) + i / 3.0) % 1.0
        rr = (u if kind == "out" else 1.0 - u) * r * 1.15
        stroke(bl, sh_circle(), cx, cy, rr,
               _colour(c, 0.0, 0.10 + 0.30 * (1 - u)), closed=True)
    return None


def fig_end(cv, bl, c, cx, cy, r):
    """Everything folds down to a line, then to nothing."""
    k = max(0.0, 1.0 - c.act_t / 1.7)
    pts = sh_sine(cycles=2.0, amp=0.9 * k)
    stroke(bl, pts, cx, cy, r, _colour(c, energy=c.rms * k))
    if k <= 0.02:
        bl.line(cx - r, cy, cx + r, cy, c.pal["hot"])
    return None


FIGURE_FOR_ACT = {
    "geometry": fig_geometry,
    "current": fig_current,
    "menagerie": fig_menagerie,
    "transform": fig_transform,
    "loss": fig_loss,
    "argument": fig_argument,
    "execution": fig_execution,
    "love": fig_love,
    "void": fig_void,
    "end": fig_end,
    "chorus": None,          # the chorus is expansion, not a shape
}
