"""Audio feedback engine + settings.

The engine is the heart of SoundKey: it renders a stream of SoundEvents into a
single WAV plus a synchronised caption transcript. Settings decide *how* each
event is expressed -- spoken, tonal, or both -- how verbose the output is, the
volume, and whether a haptic channel is active.

Speech in the simulator is represented as printed/transcribed captions rather
than synthesised voice (offline, zero-dependency). On hardware that can drive a
TTS or pre-recorded clips, the same caption strings are what gets spoken; where
speech is unavailable, the tonal cue alone still carries the meaning. This is
the multi-modal contract: nothing depends on speech being present.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .cues import CueLibrary
from .sonifier import SoundEvent
from .synth import Segment, duration_ms, gap_samples, render_segments

VERBOSITY_ORDER = {"silent": 0, "terse": 1, "normal": 2, "verbose": 3}


@dataclass
class Settings:
    speech: bool = True          # emit spoken captions
    tones: bool = True           # emit tonal cues
    haptics: bool = False        # emit an (annotated) haptic channel
    verbosity: str = "normal"    # silent | terse | normal | verbose
    volume: float = 0.6          # 0..1, maps to synth amplitude
    speech_rate: float = 1.0     # relative; recorded in transcript metadata
    inter_event_gap_ms: float = 120

    def amplitude(self) -> float:
        return max(0.0, min(1.0, self.volume))


@dataclass
class TranscriptLine:
    t_ms: float          # start time of this event in the rendered audio
    cue_id: str
    caption: str
    spoken: bool
    toned: bool

    def format(self) -> str:
        mods = []
        if self.spoken and self.caption:
            mods.append(f'say "{self.caption}"')
        if self.toned:
            mods.append(f"cue:{self.cue_id}")
        detail = " + ".join(mods) if mods else "(suppressed)"
        return f"[{self.t_ms/1000:6.2f}s] {detail}"


class AudioEngine:
    def __init__(self, settings: Optional[Settings] = None, library: Optional[CueLibrary] = None):
        self.settings = settings or Settings()
        self.lib = library or CueLibrary()

    def _should_emit(self, cue_id: str) -> bool:
        floor = self.lib[cue_id].verbosity_floor
        return VERBOSITY_ORDER[self.settings.verbosity] >= VERBOSITY_ORDER[floor]

    def render(self, events: List[SoundEvent]) -> Tuple[List[float], List[TranscriptLine]]:
        """Render events to (samples, transcript)."""
        samples: List[float] = []
        transcript: List[TranscriptLine] = []
        amp = self.settings.amplitude()
        for ev in events:
            if not self._should_emit(ev.cue_id):
                continue
            toned = self.settings.tones
            spoken = self.settings.speech and bool(ev.caption)
            t_ms = len(samples) / 44.1  # samples -> ms at 44100 Hz
            if toned:
                segs = self.lib.segments_for(ev.cue_id, **ev.params)
                samples.extend(render_segments(segs, amplitude=amp))
            elif spoken:
                # Speech-only mode: reserve a short silent slot so the
                # transcript timing stays meaningful.
                samples.extend(gap_samples(400))
            transcript.append(
                TranscriptLine(t_ms=t_ms, cue_id=ev.cue_id, caption=ev.caption,
                               spoken=spoken, toned=toned)
            )
            samples.extend(gap_samples(self.settings.inter_event_gap_ms))
        return samples, transcript

    @staticmethod
    def transcript_text(transcript: List[TranscriptLine], title: str = "") -> str:
        head = [f"# {title}"] if title else []
        head.append("# SoundKey audio transcript (captions = what a TTS voice would say)")
        head.append("")
        body = [line.format() for line in transcript]
        return "\n".join(head + body) + "\n"
