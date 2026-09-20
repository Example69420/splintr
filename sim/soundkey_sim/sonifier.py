"""Result sonifier: turn a function's output into sound + a spoken caption.

Each helper returns a :class:`SoundEvent` (or list of them): a cue id, the
parameters that cue needs, and the spoken caption a text-to-speech layer would
read. The engine renders the audio; the caption is what the simulator prints
and what a real TTS voice would say on capable hardware.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SoundEvent:
    cue_id: str
    params: dict = field(default_factory=dict)
    caption: str = ""


class Sonifier:
    """Maps semantic results to SoundEvents using the cue library's conventions."""

    def proximity(self, level: float) -> SoundEvent:
        """level in 0..1 -> geiger-style click stream (rate encodes closeness)."""
        pct = round(max(0.0, min(1.0, level)) * 100)
        return SoundEvent("proximity", {"level": level}, f"{pct} percent")

    def level(self, value: float, unit: str = "percent") -> SoundEvent:
        """Continuous value 0..1 -> pitch (e.g. battery, volume, meter)."""
        pct = round(max(0.0, min(1.0, value)) * 100)
        return SoundEvent("level_pitch", {"level": value}, f"{pct} {unit}")

    def category(self, index: int, name: Optional[str] = None) -> SoundEvent:
        """Discrete category (1-based) -> distinct tone on a rising scale."""
        caption = f"category {index}" if not name else name
        return SoundEvent("category_tone", {"category": index}, caption)

    def count(self, n: int, noun: str = "") -> SoundEvent:
        """Small integer -> that many beeps; large -> spoken number."""
        caption = f"{n} {noun}".strip()
        return SoundEvent("count_beeps", {"count": n}, caption)

    def found(self, detail: str = "") -> SoundEvent:
        return SoundEvent("result_found", {}, f"found {detail}".strip())

    def not_found(self) -> SoundEvent:
        return SoundEvent("result_not_found", {}, "nothing found")

    def status(self, cue_id: str, caption: Optional[str] = None) -> SoundEvent:
        return SoundEvent(cue_id, {}, caption or cue_id.replace("_", " "))
