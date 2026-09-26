#!/usr/bin/env bash
# world.execute(me); -- TTY music video
#
#   ./play.sh                  play it, sized to whatever terminal you are in
#   ./play.sh --list           print the measured structure and act map
#   ./play.sh --glyphs block   if your terminal font has no Braille Patterns
#
# Everything after the script name is passed straight to the player.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v python3 >/dev/null; then
    echo "play.sh: python3 is required" >&2
    exit 1
fi

# numpy is the only hard dependency beyond ffmpeg/mpv
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "play.sh: numpy is missing (try: python3 -m pip install --user numpy)" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null; then
    echo "play.sh: ffmpeg is required to measure the track" >&2
    exit 1
fi

export PYTHONPATH="$HERE/mv${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m ttymv "$@"
