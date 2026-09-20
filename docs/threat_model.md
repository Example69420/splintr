# Threat model / scope & ethics

This is the rare project whose threat model is mostly *"there isn't one."*
SoundKey is purely constructive: it makes an existing device usable by more
people. It is worth stating clearly why, and where the small real
responsibilities lie.

## What SoundKey is

An accessibility layer: a simplified, screen-free interaction model plus an
audio-feedback framework that renders menus, states, and results as sound and
optional haptics. Its purpose is **inclusion** — letting blind and low-vision
users operate common Flipper functions.

## What SoundKey is not

- **It adds no new offensive capability.** SoundKey does not create, extend, or
  unlock any transmit/attack functionality. It wraps functions that already
  exist on the device in an accessible interface. Removing SoundKey would not
  remove any capability from the Flipper.
- **It collects no data.** No personal data, no telemetry, no network. The
  simulator and app run entirely locally; the only "data" it ships is the
  synthetic sound-cue library and demo audio it generates itself.
- **It has no attack surface of its own.** It reads button presses and emits
  sound. There is no parsing of untrusted input, no network endpoint, no
  privileged operation.

## Inherited responsibilities

Where SoundKey provides an accessible wrapper around a Flipper function that has
its own responsible-use considerations (for example a scanning/proximity tool),
SoundKey **inherits that function's ethics and notes** — it neither adds to nor
subtracts from them. Making a function accessible does not change what the
function itself is for. Users remain subject to the same laws and the same
responsible-use expectations that apply to the underlying Flipper feature,
including local rules on radio transmission and on interacting with systems and
credentials you do not own.

## Responsible use, briefly

Use SoundKey (and the Flipper functions it makes accessible) only on devices,
signals, and systems you own or are explicitly authorised to test. Accessibility
is about who can operate a tool, not about what the tool is permitted to do.

## Safety of the accessibility design itself

The one genuine "safety" concern is a design-quality one: if two cues sound too
alike, a user could act on the wrong information. That is treated as a
first-class correctness property, not an afterthought — the
[distinguishability check](../eval/run_eval.py) fails if any two cues are
acoustically too close, and it runs in CI.
