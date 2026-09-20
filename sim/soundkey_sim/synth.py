"""Pure-standard-library audio synthesis for SoundKey.

Turns a list of segments -- (frequency, duration) tone bursts and silences --
into 16-bit PCM WAV data. No numpy, no third-party libraries: only ``math``,
``struct`` and ``wave`` from the standard library, so the whole simulator runs
from a clean clone with nothing to install.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

SAMPLE_RATE = 44100
DEFAULT_AMPLITUDE = 0.6
ATTACK_MS = 4
RELEASE_MS = 12


@dataclass
class Segment:
    """A single sound segment: a tone of ``freq`` Hz, or silence when freq is None."""

    freq: Optional[float]
    dur_ms: float

    @property
    def is_silence(self) -> bool:
        return self.freq is None or self.freq <= 0


def _tone_samples(freq: float, dur_ms: float, amplitude: float) -> List[float]:
    """Sine tone with a short attack/release envelope to avoid clicks."""
    n = max(1, int(SAMPLE_RATE * dur_ms / 1000.0))
    attack = min(n // 2, int(SAMPLE_RATE * ATTACK_MS / 1000.0))
    release = min(n - attack, int(SAMPLE_RATE * RELEASE_MS / 1000.0))
    out: List[float] = []
    two_pi_f = 2.0 * math.pi * freq
    for i in range(n):
        env = 1.0
        if i < attack and attack > 0:
            env = i / attack
        elif i > n - release and release > 0:
            env = max(0.0, (n - i) / release)
        out.append(amplitude * env * math.sin(two_pi_f * (i / SAMPLE_RATE)))
    return out


def _silence_samples(dur_ms: float) -> List[float]:
    return [0.0] * max(0, int(SAMPLE_RATE * dur_ms / 1000.0))


def render_segments(segments: Sequence[Segment], amplitude: float = DEFAULT_AMPLITUDE) -> List[float]:
    """Render segments to a flat list of float samples in [-1, 1]."""
    samples: List[float] = []
    for seg in segments:
        if seg.is_silence:
            samples.extend(_silence_samples(seg.dur_ms))
        else:
            samples.extend(_tone_samples(seg.freq, seg.dur_ms, amplitude))
    return samples


def duration_ms(segments: Sequence[Segment]) -> float:
    return sum(seg.dur_ms for seg in segments)


def samples_to_wav_bytes(samples: Iterable[float]) -> bytes:
    """Encode float samples as a 16-bit mono WAV byte string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for s in samples:
            clipped = max(-1.0, min(1.0, s))
            frames += struct.pack("<h", int(clipped * 32767))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def write_wav(path: str, samples: Iterable[float]) -> None:
    with open(path, "wb") as f:
        f.write(samples_to_wav_bytes(samples))


def concat(*sample_lists: Sequence[float]) -> List[float]:
    out: List[float] = []
    for s in sample_lists:
        out.extend(s)
    return out


def gap_samples(dur_ms: float) -> List[float]:
    return _silence_samples(dur_ms)
