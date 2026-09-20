"""Load the sound-cue library and resolve cues into playable segments.

The cue library (``cues/cue_library.json``) is the single source of truth for
the whole project. This module loads it, and turns each cue -- including the
dynamic, parameterised ones (proximity, pitch-mapped level, category, count) --
into a concrete list of :class:`~soundkey_sim.synth.Segment` objects that the
synth can render.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .synth import Segment

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CUE_LIBRARY_PATH = os.path.join(REPO_ROOT, "cues", "cue_library.json")
GESTURE_MAP_PATH = os.path.join(REPO_ROOT, "cues", "gesture_map.json")


@dataclass
class Cue:
    id: str
    group: str
    label: str
    purpose: str
    kind: str
    sequence: List[dict] = field(default_factory=list)
    dynamic: Optional[dict] = None
    speech_template: str = ""
    verbosity_floor: str = "normal"


class CueLibrary:
    def __init__(self, path: str = CUE_LIBRARY_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.raw = json.load(f)
        self.meta = self.raw["meta"]
        self.tone_palette = self.raw.get("tone_palette", {})
        self.cues: Dict[str, Cue] = {}
        for c in self.raw["cues"]:
            self.cues[c["id"]] = Cue(
                id=c["id"],
                group=c["group"],
                label=c["label"],
                purpose=c["purpose"],
                kind=c["kind"],
                sequence=c.get("sequence", []),
                dynamic=c.get("dynamic"),
                speech_template=c.get("speech_template", ""),
                verbosity_floor=c.get("verbosity_floor", "normal"),
            )

    def __getitem__(self, cue_id: str) -> Cue:
        return self.cues[cue_id]

    def get(self, cue_id: str) -> Optional[Cue]:
        return self.cues.get(cue_id)

    # --- resolution to segments ------------------------------------------

    def segments_for(self, cue_id: str, **params) -> List[Segment]:
        """Resolve a cue to a concrete segment list.

        ``params`` supply the values dynamic cues need:
          - proximity / level_pitch: ``level`` (0.0-1.0), optional ``total_ms``
          - category: ``category`` (1-based int)
          - count: ``count`` (int)
        """
        cue = self.cues[cue_id]
        if cue.kind in ("tone", "pattern"):
            return _static_segments(cue.sequence)
        if cue.kind == "dynamic":
            return _dynamic_segments(cue, params)
        raise ValueError(f"unknown cue kind: {cue.kind}")


def _static_segments(sequence: List[dict]) -> List[Segment]:
    segs: List[Segment] = []
    for step in sequence:
        if "gap" in step:
            segs.append(Segment(None, float(step["gap"])))
        else:
            segs.append(Segment(float(step["freq"]), float(step["dur"])))
    return segs


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dynamic_segments(cue: Cue, params: dict) -> List[Segment]:
    d = cue.dynamic or {}
    dtype = d.get("type")

    if dtype == "click_rate":
        level = _clamp01(params.get("level", 0.5))
        total_ms = float(params.get("total_ms", 1500))
        curve = d.get("curve", "quadratic")
        shaped = level * level if curve == "quadratic" else level
        rate = d["min_rate_hz"] + (d["max_rate_hz"] - d["min_rate_hz"]) * shaped
        period_ms = 1000.0 / rate
        click = d["click"]
        segs: List[Segment] = []
        elapsed = 0.0
        while elapsed + click["dur"] <= total_ms:
            segs.append(Segment(float(click["freq"]), float(click["dur"])))
            gap = max(0.0, period_ms - click["dur"])
            segs.append(Segment(None, gap))
            elapsed += click["dur"] + gap
        return segs

    if dtype == "pitch_map":
        level = _clamp01(params.get("level", 0.5))
        freq = d["min_freq"] + (d["max_freq"] - d["min_freq"]) * level
        return [Segment(float(freq), float(d["dur"]))]

    if dtype == "category":
        scale = d["scale"]
        cat = int(params.get("category", 1))
        idx = cat - 1
        octave = idx // len(scale)
        note = scale[idx % len(scale)] * (2 ** octave)
        return [Segment(float(note), float(d["dur"]))]

    if dtype == "count":
        n = int(params.get("count", 0))
        beep = d["beep"]
        gap = float(d["gap"])
        if n <= 0:
            return [Segment(None, float(beep["dur"]))]
        segs = []
        for i in range(min(n, int(d.get("max", 9)))):
            segs.append(Segment(float(beep["freq"]), float(beep["dur"])))
            if i < n - 1:
                segs.append(Segment(None, gap))
        return segs

    raise ValueError(f"unknown dynamic type: {dtype}")


def load_gesture_map(path: str = GESTURE_MAP_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
