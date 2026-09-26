#!/usr/bin/env python3
"""Which characters can this MV actually put on screen?

Rasterising needs one font per character, and the terminal font does not
cover Braille or the shade blocks, so the renderer has to know the exact
repertoire before it can pick fallbacks.  Guessing from source literals is
unreliable (escape sequences, unreachable branches); the honest answer comes
from running the show.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "mv"))

from ttymv import analyze as A                          # noqa: E402
from ttymv.canvas import WIDE                           # noqa: E402
from ttymv.player import Audio, Show, find_track, load_score         # noqa: E402


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="192x54")
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args(argv)

    cols, rows = (int(x) for x in a.size.lower().split("x"))
    track, data = load_score(fps=50.0)
    clock = Audio(data)
    show = Show(clock, color="true", fps=30.0)
    end = clock.audio_end + 3.2

    seen: collections.Counter = collections.Counter()
    i = 0
    t = 0.0
    while t < end:
        cv = show.frame(t, cols, rows, 1 / 30.0, i)
        for c in cv.ch:
            if c and c != " " and c != WIDE:
                seen[c] += 1
        i += 1
        t = i * a.every / 30.0
        if i % 200 == 0:
            print(f"  {t:6.1f}s  {len(seen)} distinct", flush=True)

    nonascii = sorted(c for c in seen if ord(c) > 126)
    print(f"\nframes rendered: {i}")
    print(f"distinct characters: {len(seen)}  "
          f"(non-ASCII {len(nonascii)})")
    for c in nonascii:
        print(f"  U+{ord(c):04X}\t{c!r}\t{seen[c]}")
    print("\nASCII:", "".join(sorted(c for c in seen if 32 <= ord(c) < 127)))

    import json
    out = ROOT / "mv" / "cache" / "repertoire.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "grid": [cols, rows], "frames": i, "end": end,
        "counts": {f"U+{ord(c):04X}": n for c, n in sorted(seen.items())},
        "chars": "".join(sorted(seen)),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
