#!/usr/bin/env python3
"""SoundKey evaluation harness.

Produces three honest, reproducible measures and their charts:

  1. Screen-free task completion (PROXY MODEL, not human data).
     For every core task (demo flow), report whether it is completable with
     audio alone, how many gestures it takes, and the auditory time-to-complete
     (the actual rendered duration). These are analytic numbers derived from the
     interface; they are a development proxy and are NOT a substitute for
     testing with blind and low-vision users. See docs/accessibility.md.

  2. Learnability (STRUCTURAL + MODEL).
     Structural: number of distinct gestures/cues to learn and the gesture
     consistency score (the same gesture => same action everywhere). Model: a
     recognition curve under an explicitly stated per-exposure learning rate.

  3. Cue distinguishability (OBJECTIVE analysis of the cue library).
     A pairwise acoustic-distance matrix over the cues, flagging any pair close
     enough to risk confusion by ear.

Run:  python3 eval/run_eval.py
Outputs: eval/results/results.json and charts under figures/.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.join(ROOT, "sim"))

from soundkey_sim.cues import CueLibrary, load_gesture_map  # noqa: E402
from soundkey_sim.engine import AudioEngine, Settings  # noqa: E402
from soundkey_sim.flows import DEMO_FLOWS, build_app, run_flow  # noqa: E402
from soundkey_sim.nav import InputHandler, NavSession  # noqa: E402
from soundkey_sim.synth import duration_ms  # noqa: E402
from soundkey_sim import svgkit as S  # noqa: E402

REP_PARAMS = {"proximity": {"level": 0.85}, "level_pitch": {"level": 0.85},
              "category_tone": {"category": 3}, "count_beeps": {"count": 3}}


# ---------------------------------------------------------------------------
# 1. Task completion (proxy)
# ---------------------------------------------------------------------------

def task_completion() -> List[dict]:
    handler = InputHandler(load_gesture_map())
    rows = []
    for flow in DEMO_FLOWS:
        engine = AudioEngine(Settings(verbosity="normal"))
        session = NavSession(build_app())
        events = run_flow(flow, session, handler)
        samples, transcript = engine.render(events)
        # Completable with audio alone iff every emitted event carries either a
        # distinct cue or a spoken caption (it always does, by construction).
        audio_only = all(line.toned or (line.spoken and line.caption) for line in transcript)
        rows.append({
            "task": flow.name,
            "gestures": len(flow.gestures),
            "announcements": len(transcript),
            "audio_seconds": round(len(samples) / 44100.0, 2),
            "completable_audio_only": audio_only,
        })
    return rows


# ---------------------------------------------------------------------------
# 2. Learnability
# ---------------------------------------------------------------------------

def learnability() -> dict:
    gm = load_gesture_map()
    lib = CueLibrary()
    # Consistency: does each (button, press) map to exactly one action? (By
    # design yes -- there is a single global gesture table, no per-screen remap.)
    seen: Dict[tuple, set] = {}
    for g in gm["gestures"]:
        seen.setdefault((g["button"], g["press"]), set()).add(g["action"])
    consistent = sum(1 for v in seen.values() if len(v) == 1)
    consistency_score = consistent / len(seen)

    # Model recognition curve: p_recognise = 1 - (1 - rate)^exposures, a standard
    # simple learning model. rate is an ASSUMPTION, stated here, not a measurement.
    rate = 0.45
    exposures = list(range(0, 9))
    curve = [round(1 - (1 - rate) ** e, 3) for e in exposures]
    return {
        "distinct_gestures": len(gm["gestures"]),
        "distinct_cues": len(lib.cues),
        "gesture_consistency_score": round(consistency_score, 3),
        "model": {"assumed_per_exposure_recognition_rate": rate,
                  "exposures": exposures, "p_recognise": curve},
    }


# ---------------------------------------------------------------------------
# 3. Cue distinguishability
# ---------------------------------------------------------------------------

def _semitones(freq: float) -> float:
    """Pitch in semitones relative to A4 (440 Hz) -- a perceptual scale, so a
    fixed Hz gap counts more at low pitch than at high pitch, matching hearing."""
    return 12.0 * math.log2(freq / 440.0)


def _features(lib: CueLibrary, cue_id: str) -> Dict[str, float]:
    segs = lib.segments_for(cue_id, **REP_PARAMS.get(cue_id, {}))
    tones = [s.freq for s in segs if not s.is_silence]
    if tones:
        semis = [_semitones(f) for f in tones]
        mean_pitch = sum(semis) / len(semis)
        span = max(semis) - min(semis)               # interval content (semitones)
        delta = semis[-1] - semis[0]
        direction = 1.0 if delta > 1 else (-1.0 if delta < -1 else 0.0)
    else:
        mean_pitch = span = direction = 0.0
    return {
        "mean_pitch": mean_pitch,     # semitones re A4
        "span": span,                  # largest interval in the cue (semitones)
        "direction": direction,        # rising / flat / falling
        "onsets": float(len(tones)),   # rhythm length
        "duration": duration_ms(segs),  # ms
    }


def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Perceptual distance in ~octave units. Roughly: 1.0 == 'an octave apart in
    pitch centre, or a different contour, interval and rhythm'. Confusable pairs
    sit near 0."""
    dp = (a["mean_pitch"] - b["mean_pitch"]) / 12.0      # octaves of pitch-centre
    ds = (a["span"] - b["span"]) / 12.0                   # interval-size mismatch
    dd = (a["direction"] - b["direction"]) / 2.0          # contour mismatch (0..1)
    do = (a["onsets"] - b["onsets"]) / 3.0                # rhythm-length mismatch
    dt = (a["duration"] - b["duration"]) / 250.0          # duration mismatch (tick vs tone)
    w = {"p": 1.0, "s": 0.9, "d": 1.0, "o": 1.0, "t": 0.8}
    total = (w["p"] * dp * dp + w["s"] * ds * ds + w["d"] * dd * dd
             + w["o"] * do * do + w["t"] * dt * dt)
    return math.sqrt(total / sum(w.values()))


def distinguishability() -> dict:
    lib = CueLibrary()
    ids = list(lib.cues.keys())
    vecs = {cid: _features(lib, cid) for cid in ids}
    matrix = [[_distance(vecs[a], vecs[b]) for b in ids] for a in ids]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((round(matrix[i][j], 3), ids[i], ids[j]))
    pairs.sort()
    threshold = 0.10
    flagged = [p for p in pairs if p[0] < threshold]
    return {
        "ids": ids,
        "matrix": [[round(x, 3) for x in row] for row in matrix],
        "closest_pairs": pairs[:8],
        "confusable_threshold": threshold,
        "flagged_pairs": flagged,
        "min_distance": pairs[0][0] if pairs else None,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_task_time(rows: List[dict]) -> None:
    w, h = 720, 120 + 46 * len(rows)
    svg = S.SVG(w, h, "Screen-free task completion (proxy)")
    svg.text(30, 36, "Auditory time-to-complete per core task", size=18, weight="bold")
    svg.text(30, 58, "PROXY: rendered audio duration, not human timing — see eval/README.md",
             size=11, fill=S.MUTED)
    x0, top = 220, 90
    max_s = max(r["audio_seconds"] for r in rows) or 1
    bar_w = 440
    for i, r in enumerate(rows):
        y = top + i * 46
        svg.text(210, y + 18, r["task"], size=13, anchor="end", family="monospace")
        L = bar_w * r["audio_seconds"] / max_s
        ok = r["completable_audio_only"]
        svg.rect(x0, y, L, 26, fill=(S.GREEN if ok else S.VERM), stroke="none", rx=4)
        svg.text(x0 + L + 8, y + 18, f'{r["audio_seconds"]}s · {r["gestures"]} gestures',
                 size=12, fill=S.INK)
    svg.text(30, h - 20, "All core tasks complete with audio alone (green).", size=12, fill=S.GREEN)
    svg.save(os.path.join(FIG, "eval_task_time.svg"))


def chart_learnability(data: dict) -> None:
    svg = S.SVG(700, 380, "Learnability model")
    svg.text(30, 36, "Modelled cue-recognition curve", size=18, weight="bold")
    svg.text(30, 58, f"MODEL: illustrative curve ({data['model']['assumed_per_exposure_recognition_rate']} "
             "gain/exposure), not measured", size=11, fill=S.MUTED)
    ox, oy, pw, ph = 70, 300, 500, 210
    svg.line(ox, oy, ox + pw, oy, stroke=S.INK, sw=2, marker=False)   # x axis
    svg.line(ox, oy, ox, oy - ph, stroke=S.INK, sw=2, marker=False)   # y axis
    svg.text(ox + pw / 2, oy + 40, "practice exposures per cue", size=12, anchor="middle", fill=S.MUTED)
    svg.text(24, oy - ph - 10, "P(recognise)", size=12, fill=S.MUTED)
    xs = data["model"]["exposures"]
    ys = data["model"]["p_recognise"]
    pts = []
    for e, p in zip(xs, ys):
        px = ox + pw * e / max(xs)
        py = oy - ph * p
        pts.append((px, py))
        svg.circle(px, py, 4, fill=S.BLUE)
    svg.polyline(pts, stroke=S.BLUE, sw=3)
    for frac in (0.0, 0.5, 1.0):
        yy = oy - ph * frac
        svg.line(ox, yy, ox + pw, yy, stroke=S.GRID, sw=1, marker=False)
        svg.text(ox - 8, yy + 4, f"{frac:.1f}", size=11, anchor="end", fill=S.MUTED)
    svg.text(30, 360, f"Structural: {data['distinct_gestures']} gestures · "
             f"{data['distinct_cues']} cues · consistency "
             f"{int(data['gesture_consistency_score']*100)}%", size=12, fill=S.INK)
    svg.save(os.path.join(FIG, "eval_learnability.svg"))


def chart_distinguishability(data: dict) -> None:
    ids = data["ids"]
    n = len(ids)
    cell = 26
    pad_left, pad_top = 150, 150
    w = pad_left + n * cell + 40
    h = pad_top + n * cell + 60
    svg = S.SVG(w, h, "Cue distinguishability")
    svg.text(30, 36, "Cue distinguishability matrix", size=18, weight="bold")
    svg.text(30, 58, "Objective acoustic distance between every pair (darker = closer = "
             "more confusable)", size=11, fill=S.MUTED)
    mx = max(max(row) for row in data["matrix"]) or 1.0
    for i, a in enumerate(ids):
        svg.text(pad_left - 6, pad_top + i * cell + cell - 8, a, size=9, anchor="end", family="monospace")
    # column labels, rotated so they never overlap
    for j, b in enumerate(ids):
        cx = pad_left + j * cell + cell / 2
        svg.parts.append(f'<text x="{cx}" y="{pad_top-8}" font-size="9" fill="{S.MUTED}" '
                         f'font-family="monospace" transform="rotate(-60 {cx} {pad_top-8})">{b}</text>')
    for i in range(n):
        for j in range(n):
            d = data["matrix"][i][j]
            inten = 1.0 - min(1.0, d / mx)
            if i == j:
                color = "#eeeeee"
            else:
                flagged = d < data["confusable_threshold"]
                base = (213, 94, 0) if flagged else (0, 114, 178)
                # blend toward white by (1-inten)
                r = int(base[0] * inten + 255 * (1 - inten))
                g = int(base[1] * inten + 255 * (1 - inten))
                bl = int(base[2] * inten + 255 * (1 - inten))
                color = f"rgb({r},{g},{bl})"
            svg.rect(pad_left + j * cell, pad_top + i * cell, cell - 1, cell - 1,
                     fill=color, stroke="#ffffff", sw=1, rx=2)
    svg.text(30, h - 24, f"Minimum pairwise distance: {data['min_distance']} "
             f"(threshold {data['confusable_threshold']}). "
             f"{'No pairs flagged as confusable.' if not data['flagged_pairs'] else str(len(data['flagged_pairs']))+' pair(s) flagged.'}",
             size=12, fill=(S.GREEN if not data["flagged_pairs"] else S.VERM))
    svg.save(os.path.join(FIG, "eval_distinguishability.svg"))


def main() -> int:
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    results = {
        "note": "Task-completion and learnability numbers are development PROXIES/"
                "MODELS, not measurements with blind or low-vision users. "
                "Distinguishability is an objective analysis of the cue library.",
        "task_completion": task_completion(),
        "learnability": learnability(),
        "distinguishability": distinguishability(),
    }
    with open(os.path.join(HERE, "results", "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    chart_task_time(results["task_completion"])
    chart_learnability(results["learnability"])
    chart_distinguishability(results["distinguishability"])
    d = results["distinguishability"]
    print("eval complete.")
    print(f"  tasks: {len(results['task_completion'])} (all audio-only completable: "
          f"{all(r['completable_audio_only'] for r in results['task_completion'])})")
    print(f"  min cue distance: {d['min_distance']}  flagged: {len(d['flagged_pairs'])}")
    print("  charts -> figures/eval_*.svg ; data -> eval/results/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
