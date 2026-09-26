#!/usr/bin/env python3
"""Invariants the renderer must never break again.

Each check here exists because it was broken once and the symptom was
confusing (a doubled picture in Konsole, a scrolling console, credits printed
on screen).  Cheaper to assert than to re-diagnose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "mv"))

from ttymv import analyze as A                       # noqa: E402
from ttymv import scenes as S                        # noqa: E402
from ttymv.canvas import Canvas, DEFAULT             # noqa: E402
from ttymv import font as _font                       # noqa: E402
from ttymv import motifs as MO                        # noqa: E402

font = _font
from ttymv.lrc import Lyrics                         # noqa: E402
import re as _re                                     # noqa: E402
_stutter_re = _re.compile(r"[A-Za-z]-[A-Za-z]")

MOTIF_COUNT = len(MO.NAMES)
_RAIN_SET = set("01#$%&*/\\|<>[]{}=+;:~^")
from ttymv.player import (Audio, Show, find_track,     # noqa: E402
                          load_score, make_clock)

CUP = re.compile(r"\x1b\[(\d+);(\d+)H")
SGR = re.compile(r"\x1b\[[0-9;]*m")
CJK = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(name)


skipped: list[str] = []


def skip(name: str, why: str) -> None:
    """Say out loud that a check did not run -- never let it look like a pass.

    Some invariants need the *words* of the lyric, and this repository ships
    the lyric with its words reduced to initials.  Those checks run only when
    a full lyric is present; pretending they passed would be worse than the
    gap they leave.
    """
    skipped.append(name)
    print(f"  SKIP  {name}  -- {why}")


def _mark_vocabulary() -> None:
    """Every mark the renderer can draw must exist in the terminal's font.

    The whole point of rendering in Konsole's own face is that the picture is
    what the terminal shows.  A character that face lacks either falls through
    to a stranger's shapes or arrives as a tofu box -- and the second is what
    happens silently, so it is worth asserting rather than eyeballing.
    """
    sys.path.insert(0, str(Path(ROOT / "mv")))
    from ttymv import ink

    print("mark vocabulary")
    check("no texture ramp emits the character reserved for type",
          all(ink.TYPE_BLOCK not in r for r in ink.RAMPS.values()),
          str([k for k, r in ink.RAMPS.items() if ink.TYPE_BLOCK in r]))
    check("the type-block guard actually rejects one",
          _raises(ink.assert_no_type_block, ink.TYPE_BLOCK))

    paths = {(c, ink.box_char(m, c)) for c in (1, 2, 3) for m in range(1, 16)}
    check("every weight draws a distinct glyph set",
          len({g for _c, g in paths}) == 33, f"{len({g for _c, g in paths})}")
    check("heavy and light differ for the same junction",
          ink.box_char(15, 1) != ink.box_char(15, 2)
          and ink.box_char(15, 2) != ink.box_char(15, 3),
          f"{ink.box_char(15, 1)}{ink.box_char(15, 2)}{ink.box_char(15, 3)}")

    # a crossing must come out as a crossing, without the caller saying so
    lc = ink.LineCanvas(9, 9)
    lc.seg(0, 4, 8, 4)
    lc.seg(4, 0, 4, 8)
    check("two crossing rules produce a crossing glyph",
          ink.box_char(lc.mask[4 * 9 + 4], lc.wt[4 * 9 + 4]) == "\u253c",
          ink.box_char(lc.mask[4 * 9 + 4], lc.wt[4 * 9 + 4]))
    lc2 = ink.LineCanvas(9, 9)
    lc2.seg(0, 4, 4, 4)
    lc2.seg(4, 4, 4, 8)
    check("a corner produces a corner glyph",
          ink.box_char(lc2.mask[4 * 9 + 4]) == "\u2510",
          ink.box_char(lc2.mask[4 * 9 + 4]))
    lc3 = ink.LineCanvas(12, 3)
    lc3.dashed(0, 1, 11, 1, 0xFFFFFF, 1, on=1, off=1)
    lit = sum(1 for m in lc3.mask if m)
    check("a one-cell dash is still drawn", lit == 6, f"{lit} cells lit")

    chars = set()
    for m in range(1, 16):
        for c in (1, 2, 3):
            chars.add(ink.box_char(m, c))
    for r in ink.RAMPS.values():
        chars |= set(r)
    chars.discard(" ")
    import kfont
    st = kfont.FontStack(kfont.read_kde_fixed(), scale=2)
    missing = sorted(c for c in chars
                     if ord(c) not in kfont._cmap(st.primary_path,
                                                  st.primary_index)
                     and st.face_for(c).name != Path(st.primary_path).name)
    check(f"all {len(chars)} drawable marks exist in the terminal's font",
          not missing, f"missing: {missing[:8]}")

    # determinism: the same cell at the same instant must always be the same
    # character, or the video would not be reproducible
    a = [ink.ramp_char(ink.ASH, 0.5, t=i / 30, key=99, boil=0.6)
         for i in range(60)]
    b = [ink.ramp_char(ink.ASH, 0.5, t=i / 30, key=99, boil=0.6)
         for i in range(60)]
    check("a boiling ramp is deterministic", a == b)
    check("but it does change over time", len(set(a)) > 1, f"{len(set(a))}")
    check("neighbouring cells are not in lockstep",
          len({ink.ramp_char(ink.ASH, 0.5, t=1.0, key=k, boil=0.6)
               for k in range(40)}) > 1)


def _stand_in() -> None:
    """The no-music fallback must stay usable and must stay honest.

    It is not a measurement, so the thing worth asserting is not that it
    matches the track -- it cannot -- but that it drives the renderer at all,
    keeps the authored timing, and never claims to be measured.
    """
    from ttymv import shots as SH

    print("stand-in score (no music present)")
    d = A.stand_in()
    check("it carries every field the renderer reads",
          {"bands", "rms", "flux", "centroid", "low", "mid", "high",
           "novelty", "onsets", "gaps", "wave", "sections"} <= set(d))
    check("and every meta key the real analysis has",
          {"n", "duration", "audio_end", "bpm", "beat_phase", "sections",
           "sr", "fps", "nbands", "quiet_db", "wave_rate", "source_duration",
           "audio_start"} <= set(d))
    check("it says it is not a measurement", d["source"] == "stand-in",
          str(d["source"]))
    # The stand-in must agree with the grid the shots are snapped to, or the
    # whole point of it -- everything written down still lands on the beat --
    # would be quietly false.  shots.py rounds its phase to 4dp, hence the
    # looser tolerance there.
    check("the tempo is the one the shot grid already assumes",
          abs(d["bpm"] - SH.BPM) < 1e-6, f"{d['bpm']} vs {SH.BPM}")
    check("and so is the beat phase",
          abs(d["beat_phase"] - SH.BEAT_PHASE) < 1e-3,
          f"{d['beat_phase']} vs {SH.BEAT_PHASE}")
    check("it covers the whole piece",
          abs(d["duration"] - 211.912857) < 0.01
          and len(d["bands"]) == d["n"], f"{d['duration']}")

    a = Audio(d)
    check("Audio builds from it", a.n > 0 and a.audio_end > 200.0)
    check("it has the piece's act count as its sections",
          len(a.sections) == len(S.ACT_TIMELINE),
          f"{len(a.sections)} vs {len(S.ACT_TIMELINE)}")
    check("the bands are not degenerate (the picture still has level to read)",
          float(a.bands.std()) > 0.02, f"std {float(a.bands.std()):.4f}")

    # the authored timing must survive: acts and cues land where they always do
    show = Show(a, color="true", fps=30.0)
    seen = set()
    for t in (2.0, 20.0, 40.0, 70.0, 100.0, 135.0, 160.0, 195.0):
        cv = show.frame(t, 80, 24, 1 / 30, int(t * 30))
        seen.add(S.act_at(t))
        if sum(1 for c in cv.ch if c != " ") < 20:
            check(f"the stand-in actually draws something at t={t}", False,
                  "nearly blank")
            break
    else:
        check("the stand-in draws every act it is asked for", len(seen) >= 6,
              f"{sorted(seen)}")

    # no-music playback must not reach for mpv, which has no track to play
    clk = make_clock(None, a.duration, 0.0, False, None)
    check("the clock falls back to wall time with no track",
          getattr(clk, "backend", "?") == "wall", getattr(clk, "backend", "?"))
    clk.seek(10.0)
    check("and it can seek", abs(clk.now() - 10.0) < 0.5, f"{clk.now():.2f}")


def _music_offset() -> None:
    """The offset has to move something.

    This is not a hypothetical regression: ``--offset`` and the ``[``/``]``
    keys wrote to ``Show.offset`` and nothing anywhere read it, so the
    documented "shift the picture relative to the audio" did nothing at all
    while looking perfectly implemented.  Anything that adjusts the piece
    against the music is worth one assertion that reaches the picture.
    """
    print("music offset")
    track, data = load_score(fps=50.0)
    a0 = Audio(data, offset=0.0)
    a1 = Audio(data, offset=1.0)
    check("the offset moves the analysis lookup", a1.index(0) == a0.index(0) + 50,
          f"{a0.index(0)} -> {a1.index(0)}")
    check("and shifts the end of the music in master time",
          abs((a0.audio_end - a1.audio_end) - 1.0) < 1e-6,
          f"{a0.audio_end:.3f} vs {a1.audio_end:.3f}")

    # it must reach the picture, not just the tables
    def ctx_bass(off: float) -> float:
        sc = Show(Audio(data, offset=off), color="true", fps=30.0)
        return float(sc.context(62.0, 80, 24, 1 / 30, 0).bass)

    b0, b1 = ctx_bass(0.0), ctx_bass(0.5)
    check("and it reaches what is drawn", abs(b0 - b1) > 1e-6,
          f"bass at master t=62 is {b0:.4f} either way")

    # the constant is what the flag defaults to, or editing it would do nothing
    from ttymv import player as P
    args = P.build_parser().parse_args([])
    check("the CLI default is the MUSIC_OFFSET constant",
          abs(args.offset - P.MUSIC_OFFSET) < 1e-9,
          f"{args.offset} vs {P.MUSIC_OFFSET}")


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
    except Exception:
        return True
    return False


def _video_fonts() -> None:
    """The video renders in this terminal's font -- assert that it still does.

    Every number here was measured off Konsole's own configuration; if a
    package upgrade changes the family, the TTC face order, or Qt's rounding,
    the video would quietly stop looking like the terminal.  These catch that.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import kfont
    except Exception as exc:                             # pragma: no cover
        check("the video font stack imports", False, str(exc))
        return

    print("video font (this terminal's own)")
    fixed = kfont.read_kde_fixed()
    check("Konsole's configured family is the WQY Micro Hei face",
          "微米黑" in fixed.family or "WenQuanYi" in fixed.family,
          fixed.family)
    check("Konsole's configured size is 10pt", fixed.point_size == 10.0,
          f"{fixed.point_size}")
    # 5 means AnyStyle in Qt's enum; 3 would be Courier/TypeWriter
    check("Qt style hint 5 decodes as AnyStyle, not TypeWriter",
          fixed.style_hint == "AnyStyle", fixed.style_hint)

    st = kfont.FontStack(fixed, scale=4)
    check("the primary face is the *mono* member of the TTC",
          st.primary_index == 1
          and st.primary_path.endswith("wqy-microhei.ttc"),
          f"{Path(st.primary_path).name}#{st.primary_index}")
    check("Qt's cell at 10pt/96dpi is 8x15, not PIL's 8x17",
          (st.native_cell.w, st.native_cell.h) == (8, 15),
          f"{st.native_cell.w}x{st.native_cell.h} [{st.native_cell.source}]")
    check("scaling keeps the terminal's 1.875:1 cell",
          abs(st.cell.aspect - 1.875) < 1e-9, f"{st.cell.aspect:.4f}")

    cols, rows = 180, 54
    check("the grid is 180x54 at scale 4, and exactly 16:9",
          (cols * st.cw, rows * st.ch) == (5760, 3240)
          and abs(cols * st.cw / (rows * st.ch) - 16 / 9) < 1e-4,
          f"{cols * st.cw}x{rows * st.ch}")

    # The WQY face has no Braille, no U+254E and no light shade; each of those
    # must land on a face that is actually usable in a one-character cell.
    for ch, why in (("\u28ff", "Braille"), ("\u254e", "box drawing"),
                    ("\u2591", "light shade"), ("\u2588", "full block")):
        face = st.face_for(ch)
        covered = ord(ch) in kfont._cmap(face.path, face.index)
        ink = face.ink_w
        check(f"{why} U+{ord(ch):04X} resolves to that face's own glyph",
              covered and ink > 0, f"{face.name} ink={ink}")

    check("Braille does not come from FreeMono (it rings the unset dots)",
          st.face_for("\u28ff").name != "FreeMono.otf")
    check("the main face draws its own Latin and blocks",
          st.face_for("A").name == Path(st.primary_path).name
          and st.face_for("\u2588").name == Path(st.primary_path).name)

    for ch in ("\u28ff", "\u2591", "\u254e", "A", "\u2588"):
        (x0, x1, _, _) = st._ink(st.face_for(ch).font, ch)
        check(f"U+{ord(ch):04X} is not squashed: drawn at natural size",
              x1 - x0 + 1 <= st.cw + 2 * st.BLEED,
              f"{x1 - x0 + 1}px ink in a {st.cw}px cell")


def _lrc_times(path):
    import re as _re
    out = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        m = _re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", raw.strip())
        if m:
            out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3)))
    return out


def main() -> int:
    # Ask the player where its track is rather than re-deriving the layout
    # here: this used to glob "world_execute(me)/*.ogg", which broke the
    # moment the project moved into that directory and became its own root.
    try:
        track = find_track(None)
    except SystemExit:
        print("no track found", file=sys.stderr)
        return 2
    cache = A.cache_path_for(track)
    if not cache.exists():
        print("run ./play.sh --list once to build the analysis cache", file=sys.stderr)
        return 2

    audio = Audio(A.load(cache))
    lrc = next(iter(sorted((ROOT / "lyrics").glob("*.lrc"))), None)
    lyrics = Lyrics.load(lrc) if lrc else Lyrics([])
    show = Show(audio, color="true", fps=30.0)
    from ttymv.player import Show as _Show          # used by the effect blocks
    import ttymv.scenes as _SS

    print("render stream")
    sizes = [(40, 12), (80, 24), (120, 36), (200, 56)]
    for (w, h) in sizes:
        t, frame_no = 8.5, 0
        while t < 8.5 + 0.4:
            cv = show.frame(t, w, h, 1 / 30, frame_no)
            t += 1 / 30
            frame_no += 1
        out = cv.render("true", reserve_corner=False)

        check(f"{w}x{h}: no line feed anywhere",
              "\n" not in out and "\r" not in out)
        rows = CUP.findall(out)
        check(f"{w}x{h}: one cursor address per row",
              [int(r) for r, _ in rows] == list(range(1, h + 1)),
              f"got {[int(r) for r, _ in rows][:6]}...")
        check(f"{w}x{h}: every address is column 1",
              all(int(c) == 1 for _, c in rows))

        # split the stream back into rows and measure what each one emits
        parts = CUP.split(out)[1:]
        widths = []
        for i in range(0, len(parts), 2):
            body = SGR.sub("", parts[i + 1])
            widths.append(len(body))
        check(f"{w}x{h}: no row overruns the width",
              all(n <= w for n in widths), f"max={max(widths)} w={w}")
        # the reservation is a virtual-console safety measure, so assert the
        # policy rather than always reserving
        out_console = cv.render("true", reserve_corner=True)
        cparts = CUP.split(out_console)[1:]
        cwidth = len(SGR.sub("", cparts[-1]))
        check(f"{w}x{h}: console reserves the bottom-right cell",
              cwidth <= w - 1, f"last row wrote {cwidth} of {w}")
        check(f"{w}x{h}: graphical terminals use the full width",
              widths[-1] == w, f"last row wrote {widths[-1]} of {w}")

    print("wide/edge geometry")
    for (w, h) in [(20, 4), (24, 6), (30, 8), (200, 3), (46, 11), (300, 80)]:
        t, frame_no = 62.6, 0
        cv = show.frame(t, w, h, 1 / 30, frame_no)
        out = cv.render("16")
        n_rows = len(CUP.findall(out))
        check(f"{w}x{h}: renders {h} addressed rows",
              n_rows == h, f"got {n_rows}")
        check(f"{w}x{h}: no CJK in the frame",
              not CJK.search(cv.render_plain()),
              CJK.search(cv.render_plain()).group() if CJK.search(cv.render_plain()) else "")

    print("direction")
    # The MV shows a curated subset of the lyric rather than a ticker, so the
    # subset has to stay locked to the vocal: every on-screen word must land
    # exactly on a published cue for this master.
    published = {round(t, 3) for t, _ in _lrc_times(lrc)}
    off_cue = [(at, txt) for at, txt, _ in S.TEXT_CUES
               if round(at, 3) not in published]
    check("every on-screen word lands on a published cue", not off_cue,
          str(off_cue[:3]))
    check("on-screen words are a small subset, not a ticker",
          len(S.TEXT_CUES) <= 20, f"{len(S.TEXT_CUES)} cues")
    total_words = sum(len(t.split()) for _, t, _ in S.TEXT_CUES)
    check("total words on screen stays low", total_words <= 30,
          f"{total_words} words")
    shown_times = [x / 10 for x in range(0, 2120)]
    covered = sum(1 for t in shown_times if S.text_cue_at(t)[0])
    check("words are on screen for well under half the runtime",
          covered / len(shown_times) < 0.45,
          f"{100 * covered / len(shown_times):.0f}% of the time")
    metas = [c for t in shown_times for c in [lyrics.at(t)]
             if c is not None and c.kind == "meta"]
    check("no metadata cue is ever returned for display", not metas,
          str(metas[:2]))
    text_shown = "".join((lyrics.at(t).text if lyrics.at(t) else "")
                         for t in shown_times)
    check("no CJK in the lyric stream",
          not CJK.search(text_shown),
          CJK.search(text_shown).group() if CJK.search(text_shown) else "")

    print("shots")
    from ttymv import shots as SH
    shots = SH.all_shots()
    check("there is a shot roughly every two lyric lines",
          len(shots) >= 55, f"{len(shots)} shots")
    durs = [sh.duration for sh in shots]
    check("no shot is held long enough to go stale",
          max(durs) <= 5.5, f"longest {max(durs):.2f}s")
    check("no shot is too short to register",
          min(durs) >= 1.0, f"shortest {min(durs):.2f}s")
    check("shots cover the song", sum(durs) > 200, f"{sum(durs):.1f}s")
    check("the same motif never lands twice in a row",
          all(a.motif != b.motif for a, b in zip(shots, shots[1:])))
    used = {sh.motif for sh in shots}
    check("the motif library is actually used", len(used) >= 12,
          f"{len(used)} of {MOTIF_COUNT}")
    counts = {}
    for sh in shots:
        counts[sh.motif] = counts.get(sh.motif, 0) + 1
    check("no motif dominates",
          max(counts.values()) <= len(shots) / 8 + 2,
          f"most used {max(counts.values())}x")
    check("the camera varies between shots",
          len({sh.zoom for sh in shots}) >= 4 and len({sh.rot for sh in shots}) >= 4
          and any(sh.mirror for sh in shots))
    # Every boundary is either a published lyric cue or, inside an
    # instrumental stretch, a beat of the measured tempo.  Nothing is placed
    # at an arbitrary fraction of a gap.
    published = {round(t, 3) for t, _ in _lrc_times(lrc)}
    beat = 60.0 / 130.0
    phase = 0.1254

    def on_beat(t: float) -> bool:
        return abs((t - phase) / beat - round((t - phase) / beat)) < 0.02

    bad = [round(sh.t0, 3) for sh in shots[1:]
           if round(sh.t0, 3) not in published and not on_beat(sh.t0)]
    check("every shot boundary is a lyric cue or a beat", not bad,
          str(bad[:4]))

    from ttymv import shots as SH

    print("screen tearing")
    pub = {round(t, 3) for t, _ in _lrc_times(lrc)}
    bad = [round(at, 3) for at, _k in S.TEAR_EVENTS if round(at, 3) not in pub]
    check("every failure moment is a published lyric cue", not bad, str(bad[:4]))
    check("the tear has real force when it fires",
          max(k for _a, k in S.TEAR_EVENTS) >= 0.9)

    # It must be silent everywhere the machine is not failing and the voice
    # is not stuttering.
    stut = [t for t, _k in SH.stutter_cues()]
    calm = [k / 2 for k in range(0, 424)
            if S.act_at(k / 2) not in S.TEAR_FLOOR
            and not any(abs(k / 2 - t) < 0.5 for t in stut)]
    loud = [S.tear_level(t, S.act_at(t)) for t in calm]
    check("the picture is stable away from the failures and the stutters",
          all(v == 0.0 for v in loud), f"max {max(loud):.2f}")

    print("exactly one title card")
    # Scanned rather than assumed: a second draw of the title would show up as
    # a second window with a wide band of display type across it.
    def wide_band(cv) -> int:
        best = 0
        for y in range(cv.h):
            cols = [x for x in range(cv.w) if cv.ch[y * cv.w + x] == "█"]
            if len(cols) >= 12:
                best = max(best, cols[-1] - cols[0] + 1)
        return best

    sc = _Show(audio, color="true", fps=30)
    windows = []
    t = 0.0
    while t < 208.0:
        if wide_band(sc.frame(t, 120, 36, 1 / 30, int(t * 30))) >= 60:
            if windows and t - windows[-1][1] <= 0.25:
                windows[-1][1] = t
            else:
                windows.append([t, t])
        t += 0.2
    real = [w for w in windows if w[1] - w[0] > 1.5]
    check("the title card is shown once, not twice", len(real) == 1,
          f"{len(real)} long title windows: {[(round(a,1), round(b,1)) for a, b in real]}")
    check("and it is the title act", real and S.act_at(real[0][0]) == "title",
          str(real[:1]))

    print("reset after a breakdown")
    check("there are three resets", len(S.RESET_EVENTS) == 3,
          f"{len(S.RESET_EVENTS)}")
    check("each sits at the end of a breakdown",
          all(S.act_at(t - 0.05) in S.TEAR_FLOOR or t > 200
              for t, _k in S.RESET_EVENTS), "a reset with no breakdown")
    check("and at the end of a breakdown, not the start",
          all(S.act_at(t + 0.05) != S.act_at(t - 0.05)
              or t > 200 for t, _k in S.RESET_EVENTS), "reset mid-breakdown")
    # how long each blank window lasts, not how far apart they are
    runs: list[list[float]] = []
    for t in [k / 200 for k in range(0, 42400)]:
        if S.reset_level(t) > 0.02:
            if runs and t - runs[-1][-1] <= 0.01:
                runs[-1].append(t)
            else:
                runs.append([t])
    longest = max((r[-1] - r[0] for r in runs), default=0.0)
    check("each reset blanks the screen for a fraction of a second",
          runs and longest < 0.75, f"longest blank {longest:.2f}s")
    total = sum((r[-1] - r[0]) for r in runs)
    check("and blanked for well under a second in total", total < 1.8,
          f"{total:.2f}s across the song")
    check("nothing is blanked anywhere else",
          all(S.reset_level(t) == 0.0
              for t in (10, 30, 60, 100, 130, 155, 180, 200, 205)),
          "a stray flash")

    # and it must actually wash the frame out
    def mean_bg(t, mute=False):
        sh = _Show(audio, color="true", fps=30)
        orig = _SS.reset_level
        if mute:
            _SS.reset_level = lambda *_a, **_k: 0.0
        w = t - 0.8
        while w < t:
            sh.frame(w, 100, 30, 1 / 30, int(w * 30))
            w += 1 / 30
        cv = sh.frame(t, 100, 30, 1 / 30, int(t * 30))
        _SS.reset_level = orig
        vals = [((b >> 16) & 255) + ((b >> 8) & 255) + (b & 255)
                for b in cv.bg if b != -1]
        return sum(vals) / max(1, len(vals))

    lit, dark = mean_bg(147.70), mean_bg(147.70, mute=True)
    check("the reset visibly washes the frame out", lit > dark * 1.8,
          f"mean background {lit:.0f} lit vs {dark:.0f} not")

    print("code stream")
    from ttymv.scenes import backdrop_mode as _mode

    class _A:
        def __init__(self, act, t):
            self.act, self.t = act, t

    check("the code stream covers the boot", _mode(_A("boot", 8.0)) == "rain")
    check("and runs on through the whole title", _mode(_A("title", 25.0)) == "rain")
    sh = _Show(audio, color="true", fps=30)
    w = 22.0
    while w < 25.0:
        sh.frame(w, 100, 30, 1 / 30, int(w * 30))
        w += 1 / 30
    late = sh.frame(25.0, 100, 30, 1 / 30, 750)
    sh2 = _Show(audio, color="true", fps=30)
    w = 2.0
    while w < 5.0:
        sh2.frame(w, 100, 30, 1 / 30, int(w * 30))
        w += 1 / 30
    early = sh2.frame(5.0, 100, 30, 1 / 30, 150)
    def rain_dots(cv):
        return sum(1 for ch in cv.render_plain().replace("\n", "")
                   if ch in _RAIN_SET)
    check("and it is still falling there", rain_dots(late) > 20,
          f"{rain_dots(late)} glyphs at 25s")

    print("vocal stutter")
    st = SH.stutter_cues()
    check("stutters are found at all", len(st) >= 10, f"{len(st)}")
    check("every stutter lands on a published lyric cue",
          not [t for t, _k in st if round(t, 3) not in pub], "")
    check("stutters are not so dense they stop reading as events",
          len(st) <= 25, f"{len(st)}")
    check("no stutter is at zero strength",
          all(0.0 < k <= 1.0 for _t, k in st))

    # The table is baked in, because the lyrics that ship here have their
    # words reduced to initials and a detector reading them would find
    # nothing -- silently shipping a film without the tears.  The lyric is
    # still the oracle, so when the full text is to hand, re-derive the table
    # from it and require the two to agree exactly.
    full = SH.lyric_path().with_suffix(".lrc.full") if SH.lyric_path() else None
    if full is not None and full.exists() and SH.lyric_has_words(full):
        derived = SH.stutter_cues_from_lyric(full)
        check("the baked stutter table still matches the lyric",
              derived == SH.stutter_cues(),
              f"derived {len(derived)} vs baked {len(SH.stutter_cues())}")
    else:
        skip("the baked stutter table still matches the lyric",
             "no full lyric available (the bundled copy has initials only)")

    # The shapes the detector must be able to see.  Asked structurally rather
    # than by quoting the words, so that checking the detector does not mean
    # republishing what it detects.
    if SH.lyric_has_words():
        rows = {round(t, 2): txt for t, txt in SH.read_lrc(SH.lyric_path())}
        hit = [rows.get(round(t, 2), "") for t, _k in st]
        dup = any(len(ws) > len(set(ws))
                  for ws in ([w.strip(" ,.!?;:()[]\"") for w in v.lower().split()]
                             for v in hit))
        check("a repeated-word line is caught", dup)
        runs = sum(1 for a, b in zip(hit, hit[1:])
                   if a and b and SH._similar(a, b))
        check("a repeated-line run is caught", runs >= 2, f"{runs} adjacent pairs")
        hy = sum(1 for v in hit if _stutter_re.search(v))
        check("a hyphenated letter run is caught", hy >= 3, f"{hy}")
    else:
        for nm in ("a repeated-word line is caught",
                   "a repeated-line run is caught",
                   "a hyphenated letter run is caught"):
            skip(nm, "bundled lyric carries initials only")

    # short: this is a clipped syllable, not a failing machine
    from ttymv.analyze import load as _load
    g = audio.gap_tear
    nz = [i for i, v in enumerate(g) if v > 0]
    runs = []
    for i in nz:
        if runs and i - runs[-1][-1] <= 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    longest = max((len(r) for r in runs), default=0) / audio.fps
    check("each stutter tear is short", longest <= 0.5, f"longest {longest:.2f}s")
    check("and they cover only a small part of the song",
          len(nz) / max(1, len(g)) < 0.06,
          f"{100 * len(nz) / len(g):.1f}% of frames")
    # sampled only where the act really is one of the failing stretches --
    # the argument section starts at 118.333, so a window beginning at 118.0
    # is still the previous act
    inside = [S.tear_level(t / 4, S.act_at(t / 4))
              for t in range(0, 848)
              if S.act_at(t / 4) in S.TEAR_FLOOR]
    check("and unstable throughout them", inside and min(inside) > 0.2,
          f"min {min(inside) if inside else 'n/a'} over {len(inside)} samples")

    # And it must actually move the frame, not just compute a number.
    def rows_with(t, level):
        sh = _Show(audio, color="true", fps=30)
        orig = _SS.tear_level
        if level is not None:
            _SS.tear_level = lambda *_a, **_k: level
        w = t - 1.6
        while w < t:
            sh.frame(w, 120, 36, 1 / 30, int(w * 30))
            w += 1 / 30
        out = sh.frame(t, 120, 36, 1 / 30, int(t * 30)).render_plain().split("\n")
        _SS.tear_level = orig
        return out, sh.glitch

    for t, label in [(120.0, "the verdict"), (150.4, "an execution")]:
        on, _g = rows_with(t, 1.0)
        off, _g0 = rows_with(t, 0.0)
        moved = sum(1 for x, y in zip(on, off) if x != y)
        check(f"the tear visibly displaces the frame at {label}",
              moved >= max(3, len(on) // 4), f"{moved}/{len(on)} rows")
    # At a calm moment the *natural* level is zero, so the frame it produces
    # must be identical to one with tearing switched off entirely.
    natural, gnat = rows_with(30.0, None)
    forced, _gf = rows_with(30.0, 0.0)
    check("no tear level is reached in a calm stretch", gnat == 0.0, f"{gnat:.3f}")
    check("and the frame is identical to one with tearing switched off",
          natural == forced, "frames differ")

    # Randomised, and with no period: consecutive frames must not repeat a
    # pattern, or the tear reads as a flicker rather than as a fault.
    # The *plan* is deliberately held for a beat or two, so the thing that
    # must not repeat is the rendered frame.
    seq = []
    plans = set()
    sh = _Show(audio, color="true", fps=30)
    t = 119.2
    while t < 120.9:
        seq.append(sh.frame(t, 100, 30, 1 / 30, int(t * 30)).render_plain())
        if sh._tear_plan:
            plans.add((sh._tear_plan["roll"],
                       tuple(sh._tear_plan["offsets"]),
                       tuple(sh._tear_plan["repeat_rows"])))
        t += 1 / 30
    check("no two consecutive frames draw the same tear",
          len(set(seq)) >= len(seq) - 1,
          f"{len(set(seq))} distinct of {len(seq)} frames")
    check("and the tear is redrawn in several different shapes",
          len(plans) >= 5, f"{len(plans)} distinct plans")
    check("each is held for a random beat or two, never a fixed cycle",
          0 <= sh._tear_hold <= 2)

    # --tear-random must actually change the draw
    a1, _ = rows_with(120.0, 1.0)
    sh_rand = _Show(audio, color="true", fps=30)
    sh_rand._tear_salt = 1234567
    t = 118.4
    while t < 120.0:
        sh_rand.frame(t, 120, 36, 1 / 30, int(t * 30))
        t += 1 / 30
    a2 = sh_rand.frame(120.0, 120, 36, 1 / 30, 3600).render_plain().split("\n")
    check("a different tear seed draws a different tear", a1 != a2)

    print("scene subdivision")
    durs = [sh.duration for sh in shots]
    check("scenes are cut to sentences, so they are short",
          sorted(durs)[len(durs) // 2] <= 3.8,
          f"median {sorted(durs)[len(durs) // 2]:.2f}s")
    check("there are many more scenes than acts", len(shots) >= 60,
          f"{len(shots)} shots")
    # a long act must not be one picture held: every act over 12 s is built
    # from at least three phases, and every phase lands on a lyric cue
    from ttymv.figures import PHASES
    long_acts = [a for a in PHASES]
    thin = [a for a in long_acts if len(PHASES[a]) < 3
            and a not in ("execution", "chorus2")]
    check("every long act is built from several scenes", not thin, str(thin))
    published2 = {round(t, 3) for t, _ in _lrc_times(lrc)}
    # The instrumental has no lyric to hang scenes on, so its phases come from
    # the measured section cuts instead.  Everywhere else a scene must change
    # when the sentence does.
    LYRICLESS = {"void"}
    stray = [(a, t) for a, v in PHASES.items() if a not in LYRICLESS
             for t, _n in v if round(t, 3) not in published2]
    check("every phase in a sung act begins on a published lyric cue",
          not stray, str(stray[:4]))
    check("the instrumental is the only act without lyric cues",
          all(not any(round(t, 3) in published2 for t, _ in PHASES[a])
              for a in LYRICLESS))
    check("the chorus alone is four scenes, not one",
          len(PHASES["chorus1"]) >= 4, str(len(PHASES["chorus1"])))

    print("overscan")
    # Elements that leave the frame: the picture should read as a window onto
    # something larger, not as a diagram that all fits inside the border.
    edge_hits = 0
    tried = 0
    for sh in shots[:26]:
        t = sh.t0 + min(1.0, sh.duration * 0.5)
        cv = show.frame(t, 120, 36, 1 / 30, 0)
        rows = cv.render_plain().split("\n")
        tried += 1
        # dots sitting in the outermost columns or rows: something ran off
        edge = 0
        for r in rows:
            if r and r[0] not in " ":
                edge += 1
            if r and r[-1] not in " ":
                edge += 1
        if edge >= 2:
            edge_hits += 1
    check("most scenes have something running past the edge",
          edge_hits >= tried * 0.6, f"{edge_hits}/{tried}")

    print("progress bars")
    all_bars = [b for sh in shots for b in sh.bars]
    with_bars = [sh for sh in shots if sh.bars]
    check("only some shots carry bars, not all",
          0 < len(with_bars) < len(shots), f"{len(with_bars)}/{len(shots)}")
    check("bars are a recurring motif, not a one-off",
          len(with_bars) >= 12, f"{len(with_bars)} shots")
    check("some shots carry several bars at once",
          any(len(sh.bars) >= 2 for sh in with_bars),
          f"max {max(len(sh.bars) for sh in shots)} per shot")
    check("bars differ in length", len({b.length for b in all_bars}) >= 4)
    check("bars differ in weight", len({b.thickness for b in all_bars}) >= 4)
    angles = {round(b.angle, 3) for b in all_bars}
    check("bars appear level", 0.0 in angles)
    check("and tilted", any(a != 0.0 for a in angles), f"{len(angles)} angles")
    check("bars sit at different heights",
          len({b.y for b in all_bars}) >= 4)
    tilts = [round(b.angle, 3) for b in all_bars if b.angle]
    level = [b for b in all_bars if b.angle == 0.0]
    check("level bars stay the minority",
          len(level) <= len(all_bars) * 0.22,
          f"{len(level)}/{len(all_bars)} level")
    check("no bar sits at a tilt too shallow to read as a tilt",
          not [b for b in all_bars if 0.0 < abs(b.angle) < SH.MIN_TILT],
          f"shallowest tilt {min((abs(b.angle) for b in all_bars if b.angle), default=0):.3f}")
    check("tilted bars take continuous angles, not a short list",
          len(set(tilts)) >= len(tilts) - 2 and len(set(tilts)) >= 20,
          f"{len(set(tilts))} distinct angles over {len(tilts)} bars")
    check("bars avoid the middle of the frame, where the subject is",
          all(abs(b.y) >= 0.40 for b in all_bars),
          str(sorted({round(abs(b.y), 2) for b in all_bars})[:4]))
    check("bars are framed", sum(1 for b in all_bars if b.framed)
          >= len(all_bars) * 0.6, f"{sum(1 for b in all_bars if b.framed)}")
    # The whole point: a bar reads something musical or narrative, never the
    # global playhead.  If someone adds "duration" back, this fails.
    allowed = set(SH.SEMANTICS)
    stray = sorted({b.semantic for b in all_bars} - allowed)
    check("no bar tracks global playback position", not stray, str(stray))
    check("no bar is beat-locked (they are instruments, not VU meters)",
          not ({"beat", "energy"} & {b.semantic for b in all_bars}),
          str(sorted({b.semantic for b in all_bars})))

    print("no whole-screen strobing")
    # The frame background is the one thing that can change every cell at once.
    # It is smoothed hard so it cannot flash, and switched off entirely on a
    # virtual console, where a 16-colour palette turns small changes into
    # discrete jumps.
    jump = 0.0
    where = 0.0
    prev = None
    t = 55.0
    while t < 76.0:
        cv = show.frame(t, 120, 36, 1 / 30, 0)
        cur = cv.bg[0]
        if prev is not None:
            d = sum(abs(((cur >> sh) & 255) - ((prev >> sh) & 255))
                    for sh in (16, 8, 0))
            if d > jump:
                jump, where = d, t
        prev = cur
        t += 1 / 30
    check("the whole frame never jumps in brightness", jump <= 24,
          f"largest step {jump}/765 at t={where:.1f}")

    from ttymv.caps import Caps as _Caps
    console_show = Show(audio, color="16", fps=30, glyphs="block",
                        caps=_Caps(term="linux", colors=8, is_console=True))
    # On a console the background must be the act's colour and nothing else --
    # no per-frame tint riding the bass.  (It still cross-fades between acts,
    # which is a slow cut, not a flash.)
    tinted = []
    jump_c = 0
    prev_c = None
    for k in range(80, 300):
        t = k / 4
        cv = console_show.frame(t, 80, 24, 1 / 30, 0)
        want = S.palette_at(t)["base"]
        if cv.bg[0] != want:
            tinted.append((round(t, 2), cv.bg[0], want))
        if prev_c is not None:
            jump_c = max(jump_c, sum(abs(((cv.bg[0] >> sh) & 255)
                                         - ((prev_c >> sh) & 255))
                                     for sh in (16, 8, 0)))
        prev_c = cv.bg[0]
    check("on a virtual console the background does not ride the music",
          not tinted, f"{len(tinted)} frames tinted, e.g. {tinted[:2]}")
    check("and never jumps between palette entries", jump_c <= 24,
          f"largest step {jump_c}/765")

    print("flashed words")
    fz = S.FLASH_CUES
    published = {round(t, 3) for t, _ in _lrc_times(lrc)}
    bad = [at for at, _t, _h in fz if round(at, 3) not in published]
    check("every flashed word lands on a published cue", not bad, str(bad[:4]))
    check("there are enough of them to be a layer", len(fz) >= 18, f"{len(fz)}")
    check("they do not collide with the big cues",
          not ({round(at, 1) for at, _t, _h in fz}
               & {round(at, 1) for at, _t, _h in S.TEXT_CUES}))
    check("words stay short enough for the micro font",
          max(len(t) for _a, t, _h in fz) <= 16,
          str(max(fz, key=lambda c: len(c[1]))[1]))
    # micro type, placed away from the middle where the subject lives
    def peripheral(x, y):
        return abs(x - 0.5) > 0.20 or abs(y - 0.5) > 0.28
    inner = [(x, y) for x, y in S.FLASH_ZONES if not peripheral(x, y)]
    check("every flash anchor is out at the edge, never centre",
          not inner, str(inner))
    check("the micro font really is smaller than the display font",
          font.measure_tiny("SATISFACTION")[0]
          < font.measure("SATISFACTION", 1)[0])
    check("micro glyphs are 3x5", (font.TINY_W, font.TINY_H) == (3, 5))

    print("boot console")
    kinds = {k for k, _t, _b in S.BOOT_LINES}
    # Three names are buried in the log rather than announced: the machine,
    # the runtime it runs under, and whoever wrote the thing being loaded.
    log = " ".join(t for _k, t, _b in S.BOOT_LINES)
    for name, needles in (("Aperture", ("aperture", "c0re", "enrichment")),
                          ("Mili", ("mili", "momocashew", "kasai", "Miracle Milk")),
                          ("dsh", ("dsh", "DeepSeek"))):
        hits = sum(1 for n in needles if n.lower() in log.lower())
        check(f"the boot log carries {name}, and not just once", hits >= 2,
              f"{hits} distinct spellings")
    # and they should be spread over the log's three registers, not clustered
    for name, needle in (("Aperture", "aperture"), ("Mili", "mili"),
                         ("dsh", "dsh")):
        kinds = {k for k, t, _b in S.BOOT_LINES if needle in t.lower()}
        check(f"{name} appears in more than one kind of line",
              len(kinds) >= 2, str(sorted(kinds)))

    check("the boot log uses several real console formats",
          {"kernel", "ok", "pkg"} <= kinds, str(sorted(kinds)))
    check("and is dense enough to read as a boot",
          len(S.BOOT_LINES) >= 30, f"{len(S.BOOT_LINES)} lines")
    check("kernel lines carry a dmesg timestamp",
          any(k == "kernel" for k, _t, _b in S.BOOT_LINES))
    barred = [(k, t, b) for k, t, b in S.BOOT_LINES if b]
    check("boot lines carry inline command-line bars",
          len(barred) >= 10, f"{len(barred)} lines")
    check("the inline bars come in more than one style",
          len({b for _k, _t, b in barred}) >= 2,
          str(sorted({b for _k, _t, b in barred})))
    for style in ("hash", "pkg", "dot"):
        if any(b == style for _k, _t, b in barred):
            rendered = S._inline_bar(style, 0.5)
            check(f"the {style} bar renders as a bar",
                  "[" in rendered and "]" in rendered, rendered)
    # Line by line, not character by character: shortly after a line's slot
    # begins, the whole of it must already be on screen.
    per = max(0.16, 15.0 / len(S.BOOT_LINES))
    idx = 20
    kind, text, bar = S.BOOT_LINES[idx]
    at = 0.55 + idx * per + 0.25
    cv = show.frame(at, 160, 40, 1 / 30, 0)
    # Strip decoration before searching: the log sits over a dot field, so a
    # stray mark can land in the gap between two words.  What is being tested
    # here is that the line arrives complete, not that nothing overlaps it.
    # Substituted with a space, not deleted: a mark that landed in the gap
    # between two words must not close the gap up.
    plain = "".join(
        ch if not (0x2800 <= ord(ch) <= 0x28FF or 0x2500 <= ord(ch) <= 0x259F)
        else " " for ch in cv.render_plain().replace("\n", ""))
    head = text.split("  ")[0][:28]
    check("boot lines appear whole, not typed out", head in plain,
          f"missing {head!r} at t={at:.2f}")

    print("the floating point")
    from ttymv.story import position as _other
    import math as _m
    from ttymv.story import MERGE_SPAN
    clear = []
    for k in range(0, 424):
        tt = k / 2
        if MERGE_SPAN[0] <= tt <= MERGE_SPAN[1]:
            continue                      # the one moment they share a place
        x, y, a = _other(tt)
        if a > 0.05 and _m.hypot(x, y) < 0.85:
            clear.append((tt, round(_m.hypot(x, y), 2)))
    check("the other's walk keeps clear of the subject", not clear,
          f"{len(clear)} samples inside, e.g. {clear[:3]}")
    check("and is allowed through the middle exactly once, for the meeting",
          MERGE_SPAN[1] - MERGE_SPAN[0] < 25)
    # While it is at full opacity the walk must stay inside the frame.  Once
    # it starts to fade it is on its way out, and crossing the edge on the way
    # out is the point.
    lost = []
    for k in range(0, 424):
        tt = k / 2
        x, y, a = _other(tt)
        if a >= 0.80 and (abs(x) > 1.70 or abs(y) > 1.05):
            lost.append((tt, round(x, 2), round(y, 2)))
    check("at full opacity the walk stays inside the frame", not lost,
          str(lost[:3]))

    print("held caption")
    # ILLEGAL ARGUMENTS is asked to stay up until the next sung line.
    held = [(at, txt, hold) for at, txt, hold in S.TEXT_CUES
            if hold == S.HOLD_TO_NEXT_VOCAL]
    check("the verdict is held to the next vocal", len(held) == 1, str(held))
    if held:
        at, txt, _ = held[0]
        nxt = SH.next_cue_after(at)
        still = S.text_cue_at(nxt - 0.5)[0]
        gone = S.text_cue_at(nxt + 0.6)[0]
        check(f"{txt!r} is still up just before the next line", still == txt)
        check("and is replaced once the singing resumes", gone != txt,
              f"still {gone!r}")

    print("display type")
    # The words used to run the full frame and bury the imagery.  They are now
    # capped as a fraction of the width; assert the cap actually holds, since
    # it is the sort of thing that silently regresses.
    def type_span(cv, hot) -> int:
        """Horizontal extent of the display type.

        Selected by colour: the words are drawn near the palette's hot colour,
        while the code stream behind them is dim and its drop heads are also
        block glyphs.  Counting every block character measured the rainfall.
        """
        hr, hg, hb = (hot >> 16) & 255, (hot >> 8) & 255, hot & 255
        span = 0
        for y in range(cv.h):
            cols = []
            for x in range(cv.w):
                i = y * cv.w + x
                if cv.ch[i] != "█":
                    continue
                c = cv.fg[i]
                if c == -1:
                    continue
                d = (abs(((c >> 16) & 255) - hr) + abs(((c >> 8) & 255) - hg)
                     + abs((c & 255) - hb))
                if d < 90:
                    cols.append(x)
            if cols:
                span = max(span, cols[-1] - cols[0] + 1)
        return span

    wide_worst = 0.0
    narrow_worst = 0.0
    for (w, h) in [(80, 24), (120, 36), (200, 56), (260, 70)]:
        for at, text, _hold in S.TEXT_CUES:
            cv = show.frame(at + 0.6, w, h, 1 / 30, 0)
            frac = type_span(cv, S.palette_at(at + 0.6)["hot"]) / w
            if w >= 120:
                wide_worst = max(wide_worst, frac)
            narrow_worst = max(narrow_worst, frac)
    # On a narrow terminal a nine-letter word at the smallest raster simply is
    # two thirds of the screen; that is a floor, not a regression.  What must
    # hold is that on a roomy terminal the words no longer own the frame.
    check("type takes well under half the width when there is room",
          wide_worst <= 0.55, f"{100 * wide_worst:.0f}% at 120 cols and up")
    check("type never dominates even at 80 columns",
          narrow_worst <= 0.70, f"{100 * narrow_worst:.0f}%")

    print("story")
    # The relationship is carried entirely by where the second entity is, so
    # the beats are asserted rather than left to drift.
    from ttymv.story import position as other_at
    x, y, al = other_at(2.0)
    check("before the song starts, nobody else is there", al < 0.05, f"a={al:.2f}")
    _, _, al = other_at(16.0)
    check("at the title card, both are present", al > 0.6, f"a={al:.2f}")
    x16, _, _ = other_at(16.0)
    check("and they start far apart", abs(x16) > 0.5, f"x={x16:.2f}")
    x59, _, al59 = other_at(60.0)
    check("the first chorus is a meeting", abs(x59) < 0.1 and al59 > 0.9,
          f"x={x59:.2f} a={al59:.2f}")
    _, _, al = other_at(120.0)
    check("by the argument, the other has gone", al < 0.05, f"a={al:.2f}")
    xl, _, all_ = other_at(183.0)
    check("in the love act it returns", all_ > 0.3, f"a={all_:.2f}")
    check("but stays outside the heart -- free, while the self is trapped",
          abs(xl) > 1.05, f"x={xl:.2f}")
    _, _, al = other_at(196.0)
    check("and is gone for good before the shutdown", al < 0.05, f"a={al:.2f}")
    gaps = [t / 2 for t in range(0, 424)]
    alphas = [other_at(t)[2] for t in gaps]
    check("the other is off screen for a real stretch, not just a frame",
          sum(1 for a in alphas if a <= 0.01) >= 40,
          f"{sum(1 for a in alphas if a <= 0.01)} of {len(alphas)} samples")

    print("colour encoding")
    for mode, pat, bad in (("16", r"\x1b\[(3|4|9|10)\d+m", r"\x1b\[(38|48);2;"),
                           ("256", r"\x1b\[(38|48);5;\d+m", None),
                           ("true", r"\x1b\[(38|48);2;\d+;\d+;\d+m", None)):
        cv = Canvas(20, 3)
        cv.clear(" ", DEFAULT, 0x112233)
        for x in range(20):
            cv.put(x, 1, "x", 0xFFAA00)
        out = cv.render(mode)
        ok = bool(re.search(pat, out))
        check(f"colour mode {mode} emits its own SGR form", ok)
        if mode == "16" and bad:
            check("colour mode 16 never leaks true-colour params",
                  not re.search(bad, out))

    _video_fonts()
    _mark_vocabulary()
    _stand_in()
    _music_offset()

    print()
    if skipped:
        print(f"{len(skipped)} check(s) SKIPPED (not passed): "
              f"{', '.join(skipped)}")
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
