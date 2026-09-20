"""The SoundKey app menu tree + scripted demo flows.

``build_app()`` wires the starter adapters into the accessible menu the device
would present. The flows are scripted gesture sequences that exercise real
tasks end to end; they are what the audio demos and the evaluation harness
replay, so a listener can experience the interface without hardware.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .adapters import (BatteryReadout, ProximityScanner, SignalClassifier,
                       TagCounter, VibrationToggle)
from .nav import MenuNode, NavSession
from .sonifier import Sonifier, SoundEvent


def _approach_curve(steps: int = 12) -> List[float]:
    """A believable 'walking toward the source' strength curve: rises, dips,
    then locks on -- the shape that makes geiger feedback intuitive."""
    base = [min(1.0, 0.15 + (i / (steps - 1)) ** 1.4) for i in range(steps)]
    if steps > 4:
        base[steps // 3] *= 0.7  # a dip: you turned the wrong way briefly
    return [round(x, 3) for x in base]


def build_app() -> MenuNode:
    prox = ProximityScanner()
    clf = SignalClassifier()
    batt = BatteryReadout()
    tags = TagCounter()
    vib = VibrationToggle()
    son = Sonifier()

    scanners = MenuNode("Scanners", children=[
        MenuNode(prox.label, action=lambda s: prox.run(s.son, _approach_curve(), found=True, detail="tag")),
        MenuNode(clf.label, action=lambda s: clf.run(s.son, category=2)),
    ])
    readouts = MenuNode("Readouts", children=[
        MenuNode(batt.label, action=lambda s: batt.run(s.son, 0.72)),
        MenuNode(tags.label, action=lambda s: tags.run(s.son, 3)),
    ])
    settings = MenuNode("Settings", children=[
        MenuNode(vib.label, action=lambda s: vib.run(s.son, on=True)),
    ])
    root = MenuNode("SoundKey", children=[scanners, readouts, settings])
    return root


@dataclass
class Flow:
    name: str
    description: str
    # A list of (button, press) gestures; the input handler turns them into actions.
    gestures: List[Tuple[str, str]] = field(default_factory=list)


DEMO_FLOWS: List[Flow] = [
    Flow(
        name="locate_tag",
        description="From home, open Scanners, run the Proximity locator, hear the "
                    "geiger stream accelerate as the target gets closer, then the 'found' cue.",
        gestures=[("Down", "short"), ("Up", "short"),   # wander home list, return to Scanners
                  ("OK", "short"),                       # open Scanners (focus Proximity locator)
                  ("OK", "short"),                       # run Proximity locator
                  ("Back", "short"), ("Back", "long")],  # back out, go home
    ),
    Flow(
        name="classify_signal",
        description="Open Scanners, move to the Signal classifier, and hear the category tone.",
        gestures=[("OK", "short"), ("Down", "short"), ("OK", "short"), ("Back", "long")],
    ),
    Flow(
        name="check_battery",
        description="Open Readouts and hear the battery level as pitch.",
        gestures=[("Down", "short"), ("OK", "short"), ("OK", "short"), ("Back", "short")],
    ),
    Flow(
        name="learn_by_help",
        description="Use the Help gesture to hear where you are and what is available, "
                    "then repeat the last announcement.",
        gestures=[("Down", "long"), ("Down", "short"), ("Up", "long")],
    ),
]


def run_flow(flow: Flow, session: NavSession, input_handler) -> List[SoundEvent]:
    events: List[SoundEvent] = []
    for button, press in flow.gestures:
        action = input_handler.resolve(button, press)
        events.extend(session.do(action))
    return events
