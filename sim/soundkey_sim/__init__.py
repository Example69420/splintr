"""SoundKey desktop simulator package.

An accessibility-first audio interface for the Flipper Zero. This package lets
you experience the interface's audio on a desktop, with no hardware and no
third-party dependencies.

Author: Krishita Sanjay Choksi
License: MIT
"""

from .cues import CueLibrary, load_gesture_map
from .engine import AudioEngine, Settings
from .nav import InputHandler, MenuNode, NavSession
from .sonifier import Sonifier, SoundEvent

__version__ = "1.0.0"
__author__ = "Krishita Sanjay Choksi"

__all__ = [
    "CueLibrary",
    "load_gesture_map",
    "AudioEngine",
    "Settings",
    "InputHandler",
    "MenuNode",
    "NavSession",
    "Sonifier",
    "SoundEvent",
]
