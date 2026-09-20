# Building the SoundKey Flipper app

SoundKey builds with [`ufbt`](https://github.com/flipperdevices/flipperzero-ufbt),
the micro Flipper Build Tool for standalone external apps (`.fap`).

> **Status:** the on-device app in [`src/`](../src) is a functional accessibility
> skeleton that targets the official firmware API (`furi`, `gui`, `input`,
> `furi_hal_speaker`, `furi_hal_vibro`, `notification`). It has been written
> against that API but, because it needs the Flipper SDK to compile, the
> continuously-tested, run-from-a-clean-clone part of this project is the
> **desktop simulator** in [`sim/`](../sim). See the honest-limits section of the
> main README.

## Prerequisites

```bash
python3 -m pip install --upgrade ufbt
```

## Build

From the repository root:

```bash
# 1. (Re)generate the embedded cue table from the single source of truth.
python3 cues/generate_c_header.py

# 2. Build the .fap.
ufbt
```

The compiled app lands at `.ufbt/build/soundkey.fap`.

## Install

- **USB (recommended):** with the Flipper connected, run `ufbt launch` to build,
  upload, and start the app in one step.
- **qFlipper drag-to-SD:** copy `soundkey.fap` onto the SD card under
  `apps/Tools/` using [qFlipper](https://flipperzero.one/update), then launch it
  from **Apps → Tools → SoundKey (Accessibility)** on the device.

## Keeping the device and simulator in sync

`src/soundkey_cues.h` is generated from `cues/cue_library.json` by
`cues/generate_c_header.py`. Never edit the header by hand — change the JSON and
regenerate, so the sounds on hardware always match what the simulator plays and
what the sound-cue legend documents.
