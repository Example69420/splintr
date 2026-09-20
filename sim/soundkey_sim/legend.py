"""Generate the sound-cue legend and gesture-map tables from the JSON sources.

These tables are the project's hero artifacts: the interface, made legible.
They are generated (never hand-written) so they can never drift from the cue
library the code actually plays.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

from typing import List

from .cues import CueLibrary, load_gesture_map


def _describe_sound(cue) -> str:
    if cue.kind in ("tone", "pattern"):
        parts = []
        for step in cue.sequence:
            if "gap" in step:
                parts.append(f"·{int(step['gap'])}ms")
            else:
                parts.append(f"{int(step['freq'])}Hz/{int(step['dur'])}ms")
        return " → ".join(parts)
    d = cue.dynamic or {}
    t = d.get("type")
    if t == "click_rate":
        return (f"click stream, {d['min_rate_hz']}–{d['max_rate_hz']} Hz "
                f"({d['curve']} in level)")
    if t == "pitch_map":
        return f"pitch {d['min_freq']}–{d['max_freq']}Hz over value"
    if t == "category":
        return "note on scale " + "/".join(str(n) for n in d["scale"]) + " Hz"
    if t == "count":
        return f"{int(d['beep']['freq'])}Hz beep ×N (max {d.get('max', 9)})"
    return "(dynamic)"


def legend_markdown() -> str:
    lib = CueLibrary()
    lines: List[str] = []
    lines.append("| Cue ID | State / result | Sound (tones · patterns) | Spoken caption | Group |")
    lines.append("|---|---|---|---|---|")
    for c in lib.cues.values():
        speech = c.speech_template or "—"
        lines.append(f"| `{c.id}` | {c.label} | {_describe_sound(c)} | `{speech}` | {c.group} |")
    return "\n".join(lines)


def gesture_markdown() -> str:
    gm = load_gesture_map()
    lines: List[str] = []
    lines.append(f"_Long-press threshold: {gm['meta']['long_press_ms']} ms. "
                 "The same gesture means the same thing on every screen._\n")
    lines.append("| Button | Press | Action | Meaning | Cue |")
    lines.append("|---|---|---|---|---|")
    for g in gm["gestures"]:
        lines.append(f"| {g['button']} | {g['press']} | `{g['action']}` | {g['meaning']} | `{g['cue']}` |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("## Sound-cue legend\n")
    print(legend_markdown())
    print("\n## Gesture map\n")
    print(gesture_markdown())
