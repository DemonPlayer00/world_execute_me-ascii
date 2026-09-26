#!/usr/bin/env python3
"""Render a grid of stills from the MV into one PNG.

    python3 mv/tools/contact_sheet.py preview/mv-stills.png --size 120x36

Runs the real frame pipeline (no audio, no terminal) at the given times,
converts each ANSI frame with ansi2png, and tiles the result.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "mv"))

from ttymv import analyze as A                      # noqa: E402
from ttymv.player import Audio, Show, find_track, load_score    # noqa: E402

# The story, beat by beat: alone, first contact, the performance, the
# meeting, the transformations, the leaving, the judgement, the plea to an
# empty room, love, and shutdown.
DEFAULT_TIMES = [
    (2.0, "boot"), (21.0, "title"), (34.5, "geometry"),
    (48.5, "current"), (63.0, "chorus"), (76.5, "menagerie"),
    (83.0, "menagerie"), (93.0, "transform"), (110.0, "loss"),
    (140.0, "argument"), (157.5, "execution"), (170.0, "chorus"),
    (183.0, "love"), (202.0, "void"), (209.0, "terminated"),
]


def load_ansi2png():
    spec = importlib.util.spec_from_file_location("ansi2png", HERE / "ansi2png.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", default="120x36")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--times", default=None,
                    help="comma-separated seconds (labels default to the clock)")
    ap.add_argument("--from-shots", type=int, default=0,
                    help="instead of --times, take N shots, one per act first")
    a = ap.parse_args(argv)

    if a.from_shots:
        from ttymv import shots as SH
        from ttymv.scenes import act_at
        all_shots = SH.all_shots()
        # One frame per act first, then extra frames from the acts that have
        # more of them.  Sampling straight down the shot list gave the title
        # card -- thirteen seconds of one static picture -- two of the tiles,
        # which reads as the MV showing its title twice when it does not.
        by_act: dict = {}
        for sh in all_shots:
            by_act.setdefault(act_at(sh.t0), []).append(sh)
        # One frame per act, always.  An act is a stretch of one subject, so a
        # second frame from it is a second picture of the same thing -- which
        # is how the thirteen-second title card ended up in two tiles and made
        # it look as though the MV showed its title twice.
        order = sorted(by_act, key=lambda k: -sum(x.duration for x in by_act[k]))
        picked = []
        for act in order:
            if len(picked) >= a.from_shots:
                break
            lst = by_act[act]
            picked.append(lst[len(lst) // 2])
        picked.sort(key=lambda sh: sh.t0)
        times = [(sh.t0 + min(1.1, sh.duration * 0.55),
                  f"{act_at(sh.t0)}·{sh.motif}") for sh in picked]
    elif a.times:
        times = [(float(t), None) for t in a.times.split(",")]
    else:
        times = list(DEFAULT_TIMES)
    w, h = (int(x) for x in a.size.lower().split("x"))

    track, data = load_score(fps=50.0)
    audio = Audio(data)
    show = Show(audio, fps=30.0)

    a2p = load_ansi2png()
    font, font_path = a2p.load_font(a.font_size)
    cw = int(round(font.getlength("M")))
    ch = int(round(cw * 2))
    print(f"font {font_path}  cell {cw}x{ch}", file=sys.stderr)

    tiles = []
    t_cursor, frame_no = 0.0, 0
    for target, label in times:
        while t_cursor < target - 1e-9:          # evolve particles/peaks
            step = min(1 / 30, target - t_cursor)
            show.frame(t_cursor, w, h, step, frame_no)
            t_cursor += step
            frame_no += 1
        cv = show.frame(target, w, h, 1 / 30, frame_no)
        frame_no += 1
        rows = a2p.parse_ansi(cv.render("true", reserve_corner=False))
        tiles.append((target, label, a2p.render(rows, font, cw, ch)))

    cols = max(1, a.cols)
    rows_n = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0][2].size
    bar = int(a.font_size * 1.6)
    pad = 6
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad,
                              rows_n * (th + bar) + (rows_n + 1) * pad),
                      (10, 10, 14))
    d = ImageDraw.Draw(sheet)
    try:
        lab = ImageFont.truetype(font_path, int(a.font_size * 0.95))
    except Exception:
        lab = ImageFont.load_default()

    for i, (t, label, img) in enumerate(tiles):
        cx, cy = i % cols, i // cols
        x = pad + cx * (tw + pad)
        y = pad + cy * (th + bar + pad)
        sheet.paste(img, (x, y))
        act = next((n for s, n in reversed(__import__(
            "ttymv.scenes", fromlist=["x"]).ACT_TIMELINE) if t >= s), "?")
        d.text((x + 4, y + th + 3),
               f"{int(t // 60):02d}:{t % 60:04.1f}  "
               + (label if label else act), font=lab,
               fill=(210, 230, 235))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"{out}  {sheet.width}x{sheet.height}  {len(tiles)} stills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
