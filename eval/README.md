# Evaluation

```bash
python3 eval/run_eval.py
```

Writes `eval/results/results.json` and three charts to `figures/`. Everything is
regenerable from a clean clone (standard library only).

## What is measured — and how honest each number is

### 1. Screen-free task completion — **PROXY, not human data**
For each core task (a demo flow), the harness reports whether it completes with
audio alone, how many gestures it takes, and the **auditory time-to-complete**
(the actual rendered audio duration). These are analytic numbers derived from
the interface model. They are a **development proxy** that catches obvious
failures early. They are **not** a substitute for testing with blind and
low-vision users, whose strategies, mental models, and muscle memory differ from
a sighted person working eyes-closed. See
[`docs/accessibility.md`](../docs/accessibility.md).

### 2. Learnability — **structural fact + explicit model**
Structural (objective): the number of distinct gestures and cues to learn, and
the **gesture-consistency score** — the fraction of gestures that map to exactly
one action everywhere (100% by design; there is a single global gesture table).
Model (illustrative): a recognition curve under an *explicitly stated* assumed
per-exposure learning rate — labelled as a model, not a measurement.

### 3. Cue distinguishability — **objective analysis**
The most defensible number here. Every pair of cues is compared on a perceptual
acoustic feature vector (mean pitch in semitones, interval span, contour
direction, rhythm length, duration), producing a distance matrix. Any pair below
a confusability threshold is **flagged**. This runs in CI and the test suite
fails if any pair is flagged — so two cues can never silently drift close enough
to be confused. This check already caught and fixed a real collision during
development (`back` and `action_stop` were once acoustically identical).

## Outputs

| File | What |
|---|---|
| `results/results.json` | All metrics, machine-readable |
| `figures/eval_task_time.svg` | Auditory time-to-complete per task (proxy) |
| `figures/eval_learnability.svg` | Modelled recognition curve + structural facts |
| `figures/eval_distinguishability.svg` | Cue distance matrix (darker = closer) |
