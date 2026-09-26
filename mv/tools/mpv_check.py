#!/usr/bin/env python3
"""Exercise the mpv-backed clock: connection, drift, seek, pause."""
import json, os, subprocess, sys, time

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ttymv.player import find_track, make_clock         # noqa: E402

track = find_track(None)
clk = make_clock(track, 211.9, 202.0, True, "null")
print("backend", clk.backend, "duration", round(clk.duration, 3), flush=True)
t0 = time.monotonic()
for _ in range(8):
    time.sleep(0.5)
    wall = time.monotonic() - t0
    now = clk.now()
    print(f"  wall+{wall:5.2f}s  clock={now:7.3f}s  drift={now - 202.0 - wall:+.3f}s"
          f"  paused={clk.paused}", flush=True)
clk.seek(60.0)
time.sleep(0.8)
print(f"  after seek(60): clock={clk.now():.3f}", flush=True)
clk.toggle_pause()
time.sleep(0.6)
print(f"  paused={clk.paused} clock={clk.now():.3f}", flush=True)
clk.toggle_pause()
time.sleep(0.8)
print(f"  resumed clock={clk.now():.3f}", flush=True)
clk.close()
print("closed cleanly", flush=True)
