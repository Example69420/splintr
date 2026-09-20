"""Function adapters: wrap common Flipper-style functions in the accessible model.

Each adapter is a screen-free "screen". It exposes:
  - ``label``: how it is announced in the menu
  - ``run(sonifier, **inputs)``: produce the SoundEvents for one invocation

Adapters take *scripted inputs* (e.g. a sequence of proximity levels) so the
simulator and the evaluation harness are fully deterministic and reproducible
without hardware. On the device, the same adapter interface is fed by real
sensor readings instead of a script.

The starter set deliberately covers the output shapes that suit audio best:
a proximity/scan stream, a discrete classifier, a continuous readout, a small
count, and a boolean toggle.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

from typing import List

from .sonifier import Sonifier, SoundEvent


class ProximityScanner:
    """Geiger-style locator. Feeds a sequence of 0..1 strength levels and ends
    with found / not-found. Models 'find the strongest signal / tag / source'."""

    label = "Proximity locator"
    id = "proximity_locator"

    def run(self, son: Sonifier, levels: List[float], found: bool = True,
            detail: str = "target") -> List[SoundEvent]:
        events: List[SoundEvent] = [son.status("action_start", "locator started")]
        for lvl in levels:
            events.append(son.proximity(lvl))
        events.append(son.found(detail) if found else son.not_found())
        events.append(son.status("action_stop", "locator stopped"))
        return events


class SignalClassifier:
    """Reads a discrete category (1..6) and announces it as a distinct tone.
    Models 'what kind of signal/tag/protocol is this'."""

    label = "Signal classifier"
    id = "signal_classifier"
    CATEGORIES = {1: "sub-GHz", 2: "NFC", 3: "RFID 125k", 4: "infrared", 5: "iButton", 6: "bluetooth"}

    def run(self, son: Sonifier, category: int) -> List[SoundEvent]:
        name = self.CATEGORIES.get(category, f"category {category}")
        return [
            son.status("action_start", "classifier started"),
            son.category(category, name),
            son.status("action_stop", "classifier stopped"),
        ]


class BatteryReadout:
    """Continuous value as pitch. Models any single-value readout."""

    label = "Battery readout"
    id = "battery_readout"

    def run(self, son: Sonifier, fraction: float) -> List[SoundEvent]:
        return [son.level(fraction, "percent battery")]


class TagCounter:
    """Small count as beeps. Models 'how many tags/records are stored'."""

    label = "Saved tags"
    id = "tag_counter"

    def run(self, son: Sonifier, count: int) -> List[SoundEvent]:
        return [son.count(count, "saved tags")]


class VibrationToggle:
    """A boolean setting rendered with the toggle cues."""

    label = "Vibration"
    id = "vibration_toggle"

    def run(self, son: Sonifier, on: bool) -> List[SoundEvent]:
        return [son.status("toggle_on" if on else "toggle_off",
                           f"vibration {'on' if on else 'off'}")]


STARTER_ADAPTERS = [
    ProximityScanner(),
    SignalClassifier(),
    BatteryReadout(),
    TagCounter(),
    VibrationToggle(),
]
