# SoundKey desktop simulator

Experience the SoundKey interface's audio on a desktop — no Flipper, no
third-party dependencies (Python standard library only).

Run everything from this `sim/` directory.

## Hear a whole task

```bash
python3 -m soundkey_sim.cli flows                 # list demo flows
python3 -m soundkey_sim.cli flow locate_tag --play  # geiger locate, then "found"
python3 -m soundkey_sim.cli flow check_battery --play
```

Each flow prints a **caption transcript** (what a text-to-speech voice would
say, aligned to the audio) and writes a `.wav`.

## Hear a single cue

```bash
python3 -m soundkey_sim.cli list-cues
python3 -m soundkey_sim.cli play proximity --level 0.9 --play
python3 -m soundkey_sim.cli play category_tone --category 4 --play
python3 -m soundkey_sim.cli play success --play
```

`--play` uses whatever player is on your system (`afplay`, `aplay`, `play`,
`ffplay`); without it, a `.wav` is written to `media/` that you can open.

## Print the hero tables

```bash
python3 -m soundkey_sim.cli legend     # sound-cue legend (markdown)
python3 -m soundkey_sim.cli gestures   # gesture map (markdown)
```

## Regenerate all audio demos

```bash
python3 -m soundkey_sim.cli export-demos ../media
```

## What's inside

| Module | Role |
|---|---|
| `synth.py` | Pure-stdlib WAV synthesis (tones, patterns, envelopes) |
| `cues.py` | Loads the cue library; resolves static + dynamic cues to segments |
| `sonifier.py` | Maps semantic results → cue + spoken caption |
| `engine.py` | Audio-feedback engine + settings (speech/tones/verbosity/volume) |
| `nav.py` | Accessible navigation model + input handler |
| `adapters.py` | Function adapters (proximity, classifier, readouts, toggle) |
| `flows.py` | The app menu tree + scripted demo flows |
| `legend.py` | Generates the sound-cue legend + gesture map tables |
| `svgkit.py` | Tiny dependency-free SVG builder for figures/charts |

### A note on "speech"

The simulator renders **tonal and dynamic cues as real audio**, and represents
**speech as printed captions** (a transcript), because offline, zero-dependency
TTS isn't assumed. On hardware that can speak, the same caption strings are what
a voice would read; where speech is unavailable, the tonal cue alone still
carries the meaning. Nothing in the design depends on speech being present.
