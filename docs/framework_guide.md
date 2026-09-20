# Framework guide — making your Flipper app accessible with SoundKey

This guide is for **other Flipper app authors**. SoundKey's audio-feedback model
is small and adoptable: you can make an existing app operable and perceivable
without the screen by following four steps. The goal is ecosystem
infrastructure — if accessibility is a shared, documented pattern, far more apps
will have it.

## The model in one paragraph

An accessible Flipper app announces the focused item on every move, confirms
every action with a distinct cue, signals the edges of a list, and sonifies its
results (a scalar as click-rate or pitch, a category as a note, a boolean as a
rising/falling blip, success/failure as fixed motifs). Input is a small, fixed
gesture set with the same meaning everywhere. That is the whole contract.

## Step 1 — reuse the sound-cue library

The cue definitions live in [`cues/cue_library.json`](../cues/cue_library.json),
the single source of truth. Regenerate the C header and include it:

```bash
python3 cues/generate_c_header.py   # writes src/soundkey_cues.h
```

```c
#include "soundkey_cues.h"
#include "soundkey_audio.h"

static SoundKeyAudioConfig audio = { .volume = 0.6f, .tones = true, .haptics = false };

soundkey_audio_play(&audio, SOUNDKEY_CUE_SELECT);        // confirm a selection
soundkey_audio_proximity(&audio, strength_0_to_1);       // geiger-style locate
soundkey_audio_category(&audio, protocol_index);         // a category as a note
```

These four calls (`_play`, `_proximity`, `_level`, `_category`, `_count`) cover
most needs. Because every app draws from the *same* library, a cue means the
same thing across the whole ecosystem — a user's learning transfers.

## Step 2 — announce, confirm, and signal edges

Wherever your app moves focus, play `SOUNDKEY_CUE_NAV_MOVE` and (optionally)
speak/label the item. On selection play `SOUNDKEY_CUE_SELECT`; on back,
`SOUNDKEY_CUE_BACK`; at the top/bottom of a list, `SOUNDKEY_CUE_NAV_BOUNDARY`.
On success/failure use `SOUNDKEY_CUE_SUCCESS` / `SOUNDKEY_CUE_ERROR`. That is the
navigation model — see [`src/soundkey.c`](../src/soundkey.c) for a worked
example.

## Step 3 — adopt the fixed gesture set

Map input using the [gesture map](../figures/gesture_map.md): Up/Down move,
OK selects, Back goes back, long-Back goes home, long-Up repeats, long-Down is
help. Don't remap these per screen — consistency is the accessibility feature.

## Step 4 — sonify your results

Pick the sonifier that matches your output shape:

| Your output | Use | Cue |
|---|---|---|
| A distance / strength that changes as you move | click-rate | `soundkey_audio_proximity` |
| A single scalar (battery, level, meter) | pitch | `soundkey_audio_level` |
| A discrete class (protocol, tag type) | a note on a scale | `soundkey_audio_category` |
| A small count (0–9) | that many beeps | `soundkey_audio_count` |
| Found / not-found | fixed motif | `SOUNDKEY_CUE_RESULT_FOUND` / `_RESULT_NOT_FOUND` |

## Test without hardware

Prototype the audio in the desktop simulator before flashing:

```bash
cd sim && python3 -m soundkey_sim.cli play category_tone --category 4 --play
```

## Checklist for "accessible enough to ship"

- [ ] Every screen is operable with the fixed ten-gesture set.
- [ ] Every focus move and action produces a cue.
- [ ] Every result is perceivable by sound (and ideally haptics), not only text.
- [ ] List edges are signalled.
- [ ] Nothing requires reading the display to complete a task.
- [ ] You ran the [distinguishability check](../eval/run_eval.py) if you added
      new cues, and no pair is flagged as confusable.
