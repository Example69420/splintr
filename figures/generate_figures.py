#!/usr/bin/env python3
"""Regenerate every figure and table in the repository.

Outputs (all under figures/):
  - architecture.svg / architecture.mmd  : system architecture
  - flow_<name>.svg                       : interaction-flow diagram (what you hear)
  - legend.md / gesture_map.md            : the hero tables, generated from the JSON

Run:  python3 figures/generate_figures.py

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from soundkey_sim.cues import load_gesture_map  # noqa: E402
from soundkey_sim.engine import AudioEngine, Settings  # noqa: E402
from soundkey_sim.flows import DEMO_FLOWS, build_app, run_flow  # noqa: E402
from soundkey_sim.legend import gesture_markdown, legend_markdown  # noqa: E402
from soundkey_sim.nav import InputHandler, NavSession  # noqa: E402
from soundkey_sim import svgkit as S  # noqa: E402


def architecture() -> None:
    svg = S.SVG(880, 620, "SoundKey architecture")
    svg.text(440, 40, "SoundKey — accessibility-first audio interface", size=20,
             anchor="middle", weight="bold")
    svg.text(440, 62, "every box is operable and perceivable without the screen",
             size=13, anchor="middle", fill=S.MUTED)

    boxes = {
        "input":   (60, 110, 240, 70, "Simplified input handler", S.SKY,
                    "10 fixed gestures · same meaning everywhere"),
        "nav":     (60, 230, 240, 80, "Accessible navigation model", S.BLUE,
                    "announce item · confirm action · signal edges"),
        "engine":  (360, 110, 260, 90, "Audio feedback engine", S.GREEN,
                    "tones + patterns + spoken captions + verbosity"),
        "sonifier":(360, 250, 260, 90, "Result sonifier", S.ORANGE,
                    "proximity=rate · category=pitch · found/level/count"),
        "adapters":(60, 360, 240, 80, "Function adapters", S.PURPLE,
                    "wrap Flipper functions in the accessible model"),
        "cues":    (360, 400, 260, 70, "Sound-cue library", S.VERM,
                    "single source of truth (cues/*.json)"),
        "settings":(660, 110, 160, 360, "Settings", S.MUTED, ""),
    }
    for key, (x, y, w, h, label, color, sub) in boxes.items():
        svg.rect(x, y, w, h, fill=S.PAPER, stroke=color, sw=3)
        svg.text(x + 14, y + 26, label, size=15, weight="bold", fill=color)
        if sub:
            svg.text(x + 14, y + 46, sub, size=11, fill=S.MUTED)
    for i, opt in enumerate(["• speech vs tones", "• haptic channel", "• speed",
                             "• volume", "• verbosity"]):
        svg.text(674, 156 + i * 22, opt, size=11, fill=S.MUTED)

    # data-flow arrows (kept clear of the box interiors)
    svg.text(70, 96, "buttons →", size=12, fill=S.MUTED)
    svg.line(180, 180, 180, 230)                       # input -> nav
    svg.line(300, 250, 360, 165, marker=True)          # nav -> engine
    svg.line(180, 310, 180, 360)                       # nav -> adapters
    svg.line(300, 400, 360, 300, marker=True)          # adapters -> sonifier
    svg.line(490, 400, 490, 340)                       # cues -> sonifier
    svg.line(560, 250, 590, 205, marker=True)          # sonifier -> engine (results feed engine)
    svg.line(660, 290, 620, 200, marker=True)          # settings -> engine (representative)
    # audio output
    svg.line(490, 200, 490, 250, marker=False)         # engine down toward output rail
    svg.line(490, 340, 490, 400, marker=False)
    svg.rect(180, 520, 520, 56, fill=S.PAPER, stroke=S.GREEN, sw=3)
    svg.line(490, 470, 490, 520, stroke=S.GREEN, sw=2, marker=True)  # sonifier/engine -> output
    svg.text(440, 545, "audio out → 🔊 tones · patterns · speech", size=14, anchor="middle", weight="bold", fill=S.GREEN)
    svg.text(440, 566, "+ optional 📳 haptic channel", size=12, anchor="middle", fill=S.MUTED)
    svg.save(os.path.join(HERE, "architecture.svg"))

    mmd = """flowchart TB
    U([User buttons]) --> IH[Simplified input handler<br/>10 fixed gestures]
    IH --> NAV[Accessible navigation model<br/>announce / confirm / signal edges]
    NAV --> ENG[Audio feedback engine<br/>tones + patterns + speech]
    NAV --> AD[Function adapters<br/>wrap Flipper functions]
    AD --> SON[Result sonifier<br/>proximity / category / level / count]
    CUE[(Sound-cue library)] --> SON
    CUE --> ENG
    SON --> ENG
    SET[[Settings: speech / tones / haptics / verbosity]] -.-> ENG
    ENG --> OUT([Audio out: tones / patterns / speech + optional haptics])
"""
    with open(os.path.join(HERE, "architecture.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd)


def flow_diagram(flow_name: str) -> None:
    flow = next(f for f in DEMO_FLOWS if f.name == flow_name)
    engine = AudioEngine(Settings(verbosity="normal"))
    session = NavSession(build_app())
    handler = InputHandler(load_gesture_map())
    events = run_flow(flow, session, handler)
    _, transcript = engine.render(events)

    row_h = 30
    top = 90
    h = top + row_h * len(transcript) + 40
    svg = S.SVG(760, h, f"Interaction flow: {flow_name}")
    svg.text(30, 40, f"What you hear — flow: {flow_name}", size=18, weight="bold")
    svg.text(30, 62, flow.description, size=11, fill=S.MUTED)
    color_by_group = {"nav_move": S.SKY, "select": S.BLUE, "back": S.BLUE,
                      "home": S.BLUE, "nav_boundary": S.MUTED, "proximity": S.ORANGE,
                      "result_found": S.GREEN, "result_not_found": S.VERM,
                      "level_pitch": S.ORANGE, "category_tone": S.ORANGE,
                      "count_beeps": S.ORANGE, "action_start": S.PURPLE,
                      "action_stop": S.PURPLE}
    for i, line in enumerate(transcript):
        y = top + i * row_h
        c = color_by_group.get(line.cue_id, S.MUTED)
        svg.circle(40, y - 4, 6, fill=c)
        if i < len(transcript) - 1:
            svg.line(40, y + 2, 40, y + row_h - 10, stroke=S.GRID, sw=2, marker=False)
        svg.text(60, y, f"{line.t_ms/1000:5.2f}s", size=12, fill=S.MUTED, family="monospace")
        svg.text(130, y, line.cue_id, size=12, fill=c, weight="bold", family="monospace")
        svg.text(300, y, f'"{line.caption}"' if line.caption else "—", size=13, fill=S.INK)
    svg.save(os.path.join(HERE, f"flow_{flow_name}.svg"))


def tables() -> None:
    with open(os.path.join(HERE, "legend.md"), "w", encoding="utf-8") as f:
        f.write("# SoundKey sound-cue legend\n\n")
        f.write("_Generated from `cues/cue_library.json` — do not edit by hand._\n\n")
        f.write(legend_markdown() + "\n")
    with open(os.path.join(HERE, "gesture_map.md"), "w", encoding="utf-8") as f:
        f.write("# SoundKey gesture map\n\n")
        f.write("_Generated from `cues/gesture_map.json` — do not edit by hand._\n\n")
        f.write(gesture_markdown() + "\n")


def main() -> int:
    architecture()
    for f in DEMO_FLOWS:
        flow_diagram(f.name)
    tables()
    print("figures + tables regenerated in", HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
