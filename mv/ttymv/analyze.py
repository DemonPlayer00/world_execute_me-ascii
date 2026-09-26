"""
Offline audio analysis for the TTY music video.

The player never touches the raw waveform at render time: everything the
director needs is measured once, here, and cached as a compact .npz next to
the source track.  Measured, not assumed -- every visual event in the MV is
driven by a number that came out of this file.

Run standalone:
    python3 -m ttymv.analyze <track.ogg> -o cache/analysis.npz
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 22050          # analysis sample rate
WIN = 2048          # STFT window (46 ms)
NBANDS = 56         # log-spaced display bands
FMIN, FMAX = 32.0, 12000.0
WAVE_RATE = 300     # waveform min/max buckets per second (oscilloscope layer)
DB_FLOOR, DB_CEIL = -90.0, 0.0


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def decode(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any ffmpeg-readable track to float32 mono, shape (n,)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    return float(out)


# --------------------------------------------------------------------------
# spectral analysis
# --------------------------------------------------------------------------

def _band_edges(sr: int, nbands: int) -> list[np.ndarray]:
    """FFT bin indices per log-spaced band.

    At the bottom of the range the bands are narrower than the bin spacing, so
    a naive range() leaves them empty and they read as permanent silence.
    Every band therefore falls back to its nearest bin."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)
    edges = np.geomspace(FMIN, FMAX, nbands + 1)
    out = []
    for i in range(nbands):
        a = int(np.searchsorted(freqs, edges[i], "left"))
        b = int(np.searchsorted(freqs, edges[i + 1], "left"))
        if b <= a:
            centre = 0.5 * (edges[i] + edges[i + 1])
            j = int(np.argmin(np.abs(freqs - centre)))
            a, b = j, j + 1
        out.append(np.arange(a, b))
    return out


def stft_features(x: np.ndarray, sr: int, fps: float, nbands: int = NBANDS):
    """Return (bands_db, flux, centroid, low, mid, high) at `fps` frames/sec."""
    hop = int(round(sr / fps))
    n = 1 + max(0, (len(x) - WIN)) // hop
    window = np.hanning(WIN).astype(np.float32)
    band_idx = _band_edges(sr, nbands)
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)
    # reference the transform to full scale: a full-scale sine then reads
    # 0 dBFS, so DB_FLOOR..DB_CEIL actually means something
    gain = np.float32(2.0 / window.sum())

    bands = np.empty((n, nbands), dtype=np.float32)
    flux = np.zeros(n, dtype=np.float32)
    centroid = np.zeros(n, dtype=np.float32)
    lo = np.zeros(n, dtype=np.float32)
    mi = np.zeros(n, dtype=np.float32)
    hi = np.zeros(n, dtype=np.float32)
    low_mask = freqs < 180.0
    mid_mask = (freqs >= 180.0) & (freqs < 3000.0)
    high_mask = freqs >= 3000.0

    prev = None
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        idx = start + np.arange(stop - start)
        frames = np.stack([x[i * hop: i * hop + WIN] * window for i in idx])
        S = np.abs(np.fft.rfft(frames, axis=1)) * gain
        for b, bi in enumerate(band_idx):
            bands[start:stop, b] = (np.sqrt((S[:, bi] ** 2).mean(axis=1))
                                    if bi.size else 0.0)
        if prev is not None:
            d = S[0] - prev
            flux[start] = np.maximum(d, 0).sum()
        if stop - start > 1:
            d = np.diff(S, axis=0)
            flux[start + 1:stop] = np.maximum(d, 0).sum(axis=1)
        prev = S[-1]
        total = S.sum(axis=1) + 1e-12
        centroid[start:stop] = (S * freqs).sum(axis=1) / total
        lo[start:stop] = S[:, low_mask].mean(axis=1)
        mi[start:stop] = S[:, mid_mask].mean(axis=1)
        hi[start:stop] = S[:, high_mask].mean(axis=1)

    bands_db = 20.0 * np.log10(bands + 1e-9)
    return bands_db, flux, centroid, lo, mi, hi


def waveform_envelope(x: np.ndarray, sr: int, rate: int = WAVE_RATE):
    """Per-bucket (min, max) so the oscilloscope layer can scroll raw shape."""
    bucket = max(1, sr // rate)
    n = len(x) // bucket
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    trimmed = x[: n * bucket].reshape(n, bucket)
    return np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1)


# --------------------------------------------------------------------------
# musical structure
# --------------------------------------------------------------------------

def estimate_tempo(flux: np.ndarray, fps: float,
                   lo_bpm: float = 100.0, hi_bpm: float = 160.0,
                   coarse: float = 0.25, fine: float = 0.01):
    """Comb-filter the onset envelope.  Returns (bpm, beat_phase_seconds)."""
    o = flux.astype(np.float64)
    o -= o.mean()
    np.maximum(o, 0.0, out=o)
    if o.std() > 0:
        o /= o.std()
    t = np.arange(len(o)) / fps

    def score(bpm: float) -> float:
        period = 60.0 / bpm
        return abs(np.dot(o, np.exp(2j * np.pi * t / period))) / len(o)

    best = max(np.arange(lo_bpm, hi_bpm, coarse), key=score)
    best = max(np.arange(best - coarse, best + coarse, fine), key=score)
    period = 60.0 / best
    z = np.dot(o, np.exp(2j * np.pi * t / period))
    phase = (np.angle(z) / (2 * np.pi)) * period % period
    return float(best), float(phase)


def onset_times(flux: np.ndarray, fps: float, min_gap: float = 0.25,
                percentile: float = 92.0):
    """Peak-pick the onset envelope.  Returns [(t, strength), ...]."""
    o = flux.astype(np.float64)
    if not len(o):
        return []
    thresh = np.percentile(o, percentile)
    span = max(1, int(min_gap * fps))
    out = []
    i = 0
    n = len(o)
    while i < n:
        j = min(n, i + span)
        if o[i] >= thresh and o[i] == o[i:j].max():
            # parabolic refinement for sub-frame precision
            t = i / fps
            if 0 < i < n - 1:
                a, b, c = o[i - 1], o[i], o[i + 1]
                denom = (a - 2 * b + c)
                if abs(denom) > 1e-12:
                    t += float((a - c) / (2 * denom)) / fps
            out.append((t, float(o[i])))
            i = j
        else:
            i += 1
    return out


def novelty(spec_db: np.ndarray, fps: float, win: int = 16):
    """Checkerboard novelty: cosine distance between the past and future
    16-frame timbre windows.  Peaks here are real section changes."""
    E = np.log1p(np.maximum(spec_db + 90.0, 0.0) * 10.0)
    E = E - E.mean(axis=0)
    E = E / (E.std(axis=0) + 1e-9)
    n = len(E)
    if n < 2 * win + 1:
        return np.zeros(max(0, n), dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(win, n - win):
        a = E[i - win:i].mean(axis=0)
        b = E[i:i + win].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            out[i] = 1.0 - float(a @ b) / (na * nb)
    k = np.ones(11) / 11
    return np.convolve(out, k, "same")


def segment(nov: np.ndarray, rms_db: np.ndarray, fps: float,
            min_len: float = 4.0, min_strength: float = 0.30,
            onsets: list | None = None, snap_window: float = 0.7,
            quiet_db: float = -30.0):
    """Cut the track at its strongest timbre changes.

    Novelty peaks are smeared by the width of the timbre window, so each raw
    boundary is snapped back onto the nearest detected transient -- a cut
    should land on the drum that caused it, not half a window later."""
    n = len(nov)
    if n == 0:
        return []
    min_gap = max(1, int(min_len * fps))
    order = np.argsort(nov)[::-1]
    picked: list[int] = []
    for i in order:
        if nov[i] < min_strength:
            break
        if all(abs(i - p) >= min_gap for p in picked):
            picked.append(int(i))
    picked.sort()
    if onsets:
        times = np.array([o[0] for o in onsets], dtype=np.float64)
        strength = np.array([o[1] for o in onsets], dtype=np.float64)
        snapped = []
        for p in picked:
            t = p / fps
            near = np.where(np.abs(times - t) <= snap_window)[0]
            if len(near):
                # prefer a strong transient, but stay close to the novelty peak
                w = strength[near] / (strength[near].max() + 1e-9)
                d = np.abs(times[near] - t) / snap_window
                best = near[int(np.argmax(w - 0.8 * d))]
                snapped.append(int(round(times[best] * fps)))
            else:
                snapped.append(p)
        # de-duplicate after snapping
        dedup: list[int] = []
        for p in snapped:
            if not dedup or p - dedup[-1] >= min_gap:
                dedup.append(p)
        picked = dedup
    edges = [0] + picked + [n]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg_rms = rms_db[a:b]
        mean = float(seg_rms.mean()) if len(seg_rms) else -90.0
        out.append({
            "start": a / fps,
            "end": b / fps,
            "frames": [int(a), int(b)],
            "novelty": float(nov[a]) if a else 0.0,
            "rms_db": mean,
            "peak_db": float(seg_rms.max()) if len(seg_rms) else -90.0,
            "silent": bool(mean < quiet_db),
        })
    return out


def detect_gaps(mid: np.ndarray, rms_db: np.ndarray, fps: float,
                drop_db: float = 8.0, window: float = 0.12,
                min_len: float = 0.020, max_len: float = 0.34,
                floor_db: float = -46.0):
    """Short dropouts in the vocal band -- the line being cut.

    The mid band (180 Hz - 3 kHz) carries the voice.  A break is where it
    falls well below what it was a moment ago and comes back quickly, while
    the overall level stays up: that is the vocal being chopped, not the whole
    mix dropping out, which is a different event and already has its own cues.

    Returns [(seconds, strength), ...] with strength in 0..1.
    """
    e = 20.0 * np.log10(mid + 1e-9)
    n = len(e)
    half = max(1, int(window * fps))
    # local reference: the loudest the band has been in the last `window`
    ref = np.empty(n, dtype=np.float64)
    for i in range(n):
        ref[i] = e[max(0, i - half):i + half + 1].max()

    loud_enough = ref > floor_db
    dropped = (ref - e) > drop_db
    # The mix is still playing: this is a cut in the voice, not a full stop.
    # Judged per-frame against the loudest the track gets, so a quiet verse is
    # not mistaken for a break.
    steady = rms_db > (float(rms_db.max()) - 26.0)
    cand = dropped & loud_enough & steady

    lo = max(1, int(min_len * fps))
    hi = max(lo + 1, int(max_len * fps))
    out = []
    i = 0
    while i < n:
        if not cand[i]:
            i += 1
            continue
        j = i
        while j < n and cand[j]:
            j += 1
        if lo <= (j - i) <= hi:
            t = i / fps
            strength = float(np.clip((ref[i] - e[i]) / 17.0, 0.0, 1.0))
            out.append((round(t, 3), round(strength, 3)))
            i = j
        else:
            i = j
    # merge dropouts closer together than a syllable
    merged: list[list] = []
    for t, k in out:
        if merged and t - merged[-1][0] < 0.055:
            merged[-1][0] = t
            merged[-1][1] = max(merged[-1][1], k)
        else:
            merged.append([t, k])
    return [(t, k) for t, k in merged]


def active_span(x: np.ndarray, sr: int, floor_db: float = -55.0):
    """Where the music actually is.  Tracks routinely ship with a silence
    tail; the MV must end with the music, not with the padding."""
    win = sr // 4
    n = len(x) // win
    if n == 0:
        return 0.0, len(x) / sr
    frames = x[: n * win].reshape(n, win)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = np.where(db > floor_db)[0]
    if not len(loud):
        return 0.0, len(x) / sr
    return float(loud[0] * win / sr), float((loud[-1] + 1) * win / sr)


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------

def _u8(db: np.ndarray) -> np.ndarray:
    v = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR) * 255.0
    return np.clip(v, 0, 255).astype(np.uint8)


def _norm_u8(v: np.ndarray, hi_percentile: float = 99.5) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    hi = np.percentile(v, hi_percentile) if len(v) else 1.0
    if hi <= 0:
        hi = 1.0
    return np.clip(v / hi * 255.0, 0, 255).astype(np.uint8)


def analyze(path: str | Path, fps: float = 50.0, nbands: int = NBANDS) -> dict:
    path = Path(path)
    x = decode(path, SR)
    duration = len(x) / SR
    src_duration = probe_duration(path)

    bands_db, flux, centroid, lo, mid, hi = stft_features(x, SR, fps, nbands)
    # frame-aligned RMS derived from the band magnitudes
    frame_rms_db = 20 * np.log10(np.sqrt((10 ** (bands_db / 10.0)).mean(axis=1)) + 1e-9)

    nov = novelty(bands_db, fps)
    onsets = onset_times(flux, fps)
    bpm, beat_phase = estimate_tempo(flux, fps)
    quiet_db = float(np.percentile(frame_rms_db, 5))
    sections = segment(nov, frame_rms_db, fps, onsets=onsets,
                       quiet_db=max(-32.0, quiet_db))

    wave = waveform_envelope(x, SR, WAVE_RATE)
    audio_start, audio_end = active_span(x, SR)
    gaps = detect_gaps(mid, frame_rms_db, fps)

    return {
        "n": int(len(frame_rms_db)),
        "source": str(path.resolve()),
        "source_duration": src_duration,
        "duration": duration,
        "audio_start": audio_start,
        "audio_end": min(audio_end, duration),
        "sr": SR,
        "fps": fps,
        "nbands": nbands,
        "bpm": bpm,
        "beat_phase": beat_phase,
        "quiet_db": quiet_db,
        "bands": _u8(bands_db),                     # (n, nbands) uint8
        "flux": _norm_u8(flux),                     # (n,) uint8
        "centroid": _norm_u8(centroid),             # (n,) uint8
        "low": _norm_u8(lo),
        "mid": _norm_u8(mid),
        "high": _norm_u8(hi),
        "rms": _u8(frame_rms_db).astype(np.uint8),
        "wave": (np.clip(wave, -1, 1) * 127).astype(np.int8),
        "wave_rate": WAVE_RATE,
        "novelty": _norm_u8(nov),
        "sections": sections,
        "onsets": np.array(onsets, dtype=np.float32).reshape(-1, 2),
        "gaps": np.array(gaps, dtype=np.float32).reshape(-1, 2),
    }


def save(analysis: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in analysis.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in analysis.items() if not isinstance(v, np.ndarray)}
    # sections is JSON, not an array -- keep it in meta
    np.savez_compressed(out, **arrays, __meta__=np.array(json.dumps(meta)))
    return out


def load(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        out = dict(meta)
        for k in z.files:
            if k != "__meta__":
                out[k] = z[k]
    # convenience views used all over the renderer
    out["n"] = len(out["rms"])
    return out


PROJECT_DIR = Path(__file__).resolve().parent.parent   # .../mv
CACHE_DIR = PROJECT_DIR / "cache"


def cache_path_for(track: str | Path) -> Path:
    return CACHE_DIR / (Path(track).stem + ".npz")


# --------------------------------------------------------------------------
# the stand-in score
# --------------------------------------------------------------------------
#
# The music is a commercial recording, so it is not in the repository and the
# piece has to run without it.  Everything the renderer wants from the track
# is a table of numbers, one row per frame; this builds that table out of the
# piece's *own* authored timing -- the tempo and beat phase that ``shots.py``
# already hard-codes, the act boundaries in ``scenes.py``, the length the whole
# thing is written against -- rather than out of the recording.
#
# It is not a measurement and is never presented as one.  The meta carries
# ``source="stand-in"``, the player says so before it starts, and ``--list``
# prints the same.  What it buys is that a clone with no music still shows the
# whole piece: every act, shot, title and caption lands where it was written
# to, on the right beat.  What it cannot buy is the part that answers to the
# sound -- the spectrum bars, the transients -- because that is the one thing
# a stand-in has no access to.

# Measured once from the master and then written into the piece: ``shots.py``
# already carries the same tempo and phase, because the shot grid is snapped
# to the beat whether or not a track is present.
STAND_IN_BPM = 130.0
STAND_IN_BEAT_PHASE = 0.12537693748739487
STAND_IN_DURATION = 211.912857
STAND_IN_AUDIO_END = 207.481179138322

# How loud each act is, for the stand-in's envelope.  This is a hand-written
# reading of the piece's own shape -- the choruses and the executions are the
# loud ones, the void and the loss are not -- and not anything measured.
_ACT_LEVEL = {
    "boot": 0.30, "title": 0.45, "geometry": 0.55, "current": 0.60,
    "chorus": 0.95, "menagerie": 0.70, "transform": 0.65, "loss": 0.35,
    "argument": 0.80, "execution": 0.95, "love": 0.75, "void": 0.25,
    "end": 0.55, "terminated": 0.10,
}


def stand_in(duration: float = STAND_IN_DURATION, fps: float = 50.0,
             nbands: int = NBANDS, bpm: float = STAND_IN_BPM,
             beat_phase: float = STAND_IN_BEAT_PHASE,
             audio_end: float = STAND_IN_AUDIO_END,
             fps_wave: int = WAVE_RATE) -> dict:
    """A silent stand-in for a measured track.  See the note above."""
    from . import scenes as S          # local: keeps analyze import-free

    n = int(round(duration * fps))
    t = np.arange(n, dtype=np.float32) / fps
    beat = 60.0 / bpm

    # per-frame level: whichever act the piece is in, smoothed into it
    level = np.zeros(n, dtype=np.float32)
    acts = list(S.ACT_TIMELINE)
    for i, (at, name) in enumerate(acts):
        nxt = acts[i + 1][0] if i + 1 < len(acts) else duration
        i0, i1 = int(at * fps), min(n, int(nxt * fps))
        if i1 > i0:
            level[i0:i1] = _ACT_LEVEL.get(name, 0.5)
    k = max(1, int(0.35 * fps))
    level = np.convolve(level, np.ones(k, dtype=np.float32) / k, mode="same")

    # a pulse on the beat, and a bigger one on the downbeat
    ph = ((t - beat_phase) / beat) % 1.0
    kick = np.exp(-ph * 6.0).astype(np.float32)
    bar = np.exp(-(((t - beat_phase) / (beat * 4)) % 1.0) * 3.0).astype(np.float32)
    shaped = np.clip(level * (0.62 + 0.30 * kick + 0.16 * bar), 0.0, 1.0)

    # spectrum: energy falls off with frequency, and the pulse lives down low
    band = np.arange(nbands, dtype=np.float32) / max(1, nbands - 1)
    tilt = (1.0 - band) ** 1.35 * 0.75 + 0.25
    pulse = (0.55 + 0.45 * kick)[:, None]
    bands = np.clip(shaped[:, None] * tilt[None, :] * pulse * 255.0,
                    0, 255).astype(np.uint8)

    def u8(x) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32) * 255.0,
                       0, 255).astype(np.uint8)

    rms = u8(shaped)
    low = u8(np.clip(level * (0.55 + 0.45 * kick), 0, 1))
    mid = u8(np.clip(level * (0.70 + 0.20 * bar), 0, 1))
    high = u8(np.clip(level * 0.55 + 0.10 * bar, 0, 1))
    flux = u8(np.clip(np.abs(np.diff(kick, prepend=kick[:1])) * 5.0, 0, 1))
    # a slow brightening across the piece, so colour has somewhere to go
    centroid = u8(np.clip(0.30 + 0.45 * (t / max(1e-3, audio_end))
                          + 0.10 * bar, 0, 1))
    novelty = u8(np.clip(bar ** 2 * np.clip(level * 1.3, 0, 1), 0, 1))

    # onsets: every beat, weighted by how loud the act is
    step = max(1, int(round(beat * fps)))
    idx = np.arange(max(1, int(beat_phase * fps)), n, step)
    onsets = np.stack([idx.astype(np.float32) / fps,
                       shaped[np.clip(idx, 0, n - 1)]]).astype(np.float32).T

    # a scope trace at the wave rate: the beat, as min/max pairs
    nw = int(round(duration * fps_wave)) + 1
    tw = np.arange(nw, dtype=np.float32) / fps_wave
    wl = np.interp(tw, t, shaped).astype(np.float32)
    w = np.sin(2 * np.pi * ((tw - beat_phase) / beat)) * wl
    wave = np.stack([np.clip(w * 127, -127, 127),
                     np.clip(w * 127, -127, 127)]).astype(np.int8).T

    # sections, straight from the acts: they are what the piece is cut into
    sections = []
    for i, (at, name) in enumerate(acts):
        nxt = acts[i + 1][0] if i + 1 < len(acts) else duration
        lv = _ACT_LEVEL.get(name, 0.5)
        sections.append({
            "start": round(at, 3), "end": round(nxt, 3),
            "frames": [int(at * fps), min(n, int(nxt * fps))],
            "novelty": 0.0 if i == 0 else 0.25,
            "rms_db": round(20.0 * float(np.log10(max(1e-4, lv * 0.35))), 2),
            "peak_db": round(20.0 * float(np.log10(max(1e-4, lv * 0.9))), 2),
            "silent": lv < 0.15,
        })

    return {
        "n": n,
        "source": "stand-in",
        "source_duration": duration,
        "duration": duration,
        "audio_start": 0.0,
        "audio_end": audio_end,
        "sr": SR,
        "fps": fps,
        "nbands": nbands,
        "bpm": bpm,
        "beat_phase": beat_phase,
        "quiet_db": -60.0,
        "wave_rate": fps_wave,
        "sections": sections,
        "bands": bands,
        "rms": rms,
        "flux": flux,
        "centroid": centroid,
        "low": low,
        "mid": mid,
        "high": high,
        "novelty": novelty,
        "onsets": onsets,
        "gaps": np.zeros((0, 2), dtype=np.float32),
        "wave": wave,
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ttymv.analyze",
                                 description="Measure a track for the TTY MV.")
    ap.add_argument("track")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--bands", type=int, default=NBANDS)
    ap.add_argument("--json", action="store_true", help="print a summary")
    a = ap.parse_args(argv)

    res = analyze(a.track, a.fps, a.bands)
    out = Path(a.out) if a.out else cache_path_for(a.track)
    save(res, out)
    summary = {
        "cache": str(out),
        "duration": round(res["duration"], 3),
        "audio_span": [round(res["audio_start"], 3), round(res["audio_end"], 3)],
        "frames": int(res["n"]),
        "fps": res["fps"],
        "bpm": round(res["bpm"], 3),
        "beat_phase": round(res["beat_phase"], 4),
        "onsets": len(res["onsets"]),
        "sections": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2),
             "rms_db": round(s["rms_db"], 1), "novelty": round(s["novelty"], 3)}
            for s in res["sections"]
        ],
        "size_kb": round(out.stat().st_size / 1024, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False) if a.json
          else f"{out}  {summary['size_kb']} KB  {summary['frames']} frames  "
               f"{summary['bpm']} BPM  {len(summary['sections'])} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
