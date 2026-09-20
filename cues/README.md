# The sound-cue library

This directory holds SoundKey's **single source of truth** — the mapping of
interface states and results to tones, patterns, spoken captions, and haptics.
Everything else reads from here: the simulator, the audio demos, the on-device
Flipper app, and the generated legend/gesture tables.

## Files

| File | What |
|---|---|
| `cue_library.json` | Every cue: id, purpose, sound (tones/patterns or a dynamic rule), speech caption, haptic pattern, verbosity floor |
| `gesture_map.json` | The fixed ten-gesture control scheme (button + press → action) |
| `generate_c_header.py` | Emits `src/soundkey_cues.h` so the device app embeds the exact same cues |

## Edit here, then regenerate

Never edit `src/soundkey_cues.h` or the legend tables by hand. Change the JSON
and regenerate:

```bash
python3 cues/generate_c_header.py       # -> src/soundkey_cues.h
python3 figures/generate_figures.py     # -> figures/legend.md, figures/gesture_map.md
python3 eval/run_eval.py                # re-checks cue distinguishability
```

CI enforces that the committed C header matches the JSON, and the test suite
fails if any two cues become acoustically confusable.

## Cue kinds

- **tone / pattern** — a fixed sequence of `{freq, dur}` steps (a `gap` is
  silence). Used for navigation and status cues.
- **dynamic** — parameterised at render time:
  - `click_rate` (proximity): click stream whose rate encodes a 0–1 level
  - `pitch_map` (level): a value mapped to pitch
  - `category`: a discrete class mapped to a note on a rising scale
  - `count`: a small integer as that many beeps
