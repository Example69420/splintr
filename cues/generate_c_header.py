#!/usr/bin/env python3
"""Generate src/soundkey_cues.h from cues/cue_library.json.

The on-device Flipper app embeds the exact same cue definitions the simulator
plays, so the sound a blind user hears on hardware matches what contributors
designed and tested on the desktop. Run this whenever the cue library changes.

Run:  python3 cues/generate_c_header.py

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main() -> int:
    with open(os.path.join(HERE, "cue_library.json"), encoding="utf-8") as f:
        lib = json.load(f)

    lines = []
    lines.append("// Auto-generated from cues/cue_library.json by cues/generate_c_header.py.")
    lines.append("// Do not edit by hand. Single source of truth: cues/cue_library.json.")
    lines.append("// Author: Krishita Sanjay Choksi")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("")
    lines.append("// A tone step: frequency in Hz (0 = silence) and duration in ms.")
    lines.append("typedef struct {")
    lines.append("    uint16_t freq;")
    lines.append("    uint16_t dur_ms;")
    lines.append("} SoundKeyStep;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char* id;")
    lines.append("    const char* speech;      // caption a TTS layer would speak")
    lines.append("    const SoundKeyStep* steps;")
    lines.append("    size_t n_steps;          // 0 for dynamic cues (rendered at runtime)")
    lines.append("    uint8_t dynamic;         // 1 if parameterised (proximity/level/category/count)")
    lines.append("} SoundKeyCue;")
    lines.append("")

    cue_ids = []
    for c in lib["cues"]:
        cid = c["id"]
        cue_ids.append(cid)
        if c["kind"] in ("tone", "pattern"):
            steps = []
            for step in c["sequence"]:
                if "gap" in step:
                    steps.append(f"{{0, {int(step['gap'])}}}")
                else:
                    steps.append(f"{{{int(step['freq'])}, {int(step['dur'])}}}")
            lines.append(f"static const SoundKeyStep soundkey_steps_{cid}[] = {{{', '.join(steps)}}};")
    lines.append("")
    lines.append("static const SoundKeyCue soundkey_cues[] = {")
    for c in lib["cues"]:
        cid = c["id"]
        speech = (c.get("speech_template", "") or "").replace('"', '\\"')
        if c["kind"] in ("tone", "pattern"):
            n = len(c["sequence"])
            lines.append(f'    {{"{cid}", "{speech}", soundkey_steps_{cid}, {n}, 0}},')
        else:
            lines.append(f'    {{"{cid}", "{speech}", NULL, 0, 1}},')
    lines.append("};")
    lines.append("")
    lines.append(f"#define SOUNDKEY_CUE_COUNT {len(cue_ids)}")
    lines.append("")
    lines.append("// Cue index constants for readable lookups in the app.")
    for i, cid in enumerate(cue_ids):
        lines.append(f"#define SOUNDKEY_CUE_{cid.upper()} {i}")
    lines.append("")

    out = os.path.join(ROOT, "src", "soundkey_cues.h")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
