# Accessibility design — principles, grounding, and how to test

SoundKey exists to make common Flipper Zero functions usable by blind and
low-vision people **without relying on the screen**. This document explains the
principles the design is built on, the real-world guidance it draws from, and —
importantly — the honest limits of how it has been evaluated so far.

## The four design principles

1. **Never require the screen.** Every function must be fully operable and its
   results fully perceivable by audio (and optionally haptics) alone. The
   display is a bonus for low-vision users and sighted helpers, never a
   dependency. If a task can only be completed by reading the screen, it is not
   done.

2. **Predictable and consistent.** The same gesture and the same cue mean the
   same thing across the whole app. There are exactly ten gestures
   (see the [gesture map](../figures/gesture_map.md)), and they never change
   meaning between screens. Learnability beats cleverness.

3. **Multi-modal.** Speech *and* tones *and* an optional vibration channel,
   because people's needs and environments differ — noise, hearing, deaf-blind
   users, or simply preference. No single channel is load-bearing: where speech
   is unavailable, the tonal cue still carries the meaning; where sound is
   unwelcome, haptics and the large-text display remain.

4. **Designed with, not just for.** Real accessibility needs input from the
   people it serves. Version 1 is a foundation to iterate *with* the community,
   not a finished claim. Feedback from blind and low-vision users is explicitly
   invited — see [Contributing feedback](#contributing-feedback).

## How the design is grounded (real, cited practice)

SoundKey does not invent interaction from scratch; it adapts established
assistive-technology and audio-UI practice to the constraints of a small
hardware device:

- **Screen-reader interaction conventions.** Announce the focused item on every
  move; confirm actions; signal the edges of a list so the user is never left
  guessing why nothing changed. These mirror how VoiceOver, TalkBack, NVDA and
  JAWS structure non-visual navigation.
  See the W3C **WAI-ARIA Authoring Practices** for the underlying model:
  <https://www.w3.org/WAI/ARIA/apg/>
- **WCAG concepts, adapted to hardware.** The Web Content Accessibility
  Guidelines are written for the web, but their *principles* — perceivable,
  operable, understandable — carry over. Two are load-bearing here: never rely
  on a single sensory channel (WCAG 1.3.3 *Sensory Characteristics*), and never
  convey information by one dimension alone without an alternative (WCAG 1.4.1
  *Use of Color*, generalised to "don't rely on pitch alone — pair it with
  rhythm, speech, or haptics"). <https://www.w3.org/WAI/standards-guidelines/wcag/>
- **Auditory display / sonification and earcons.** Structured non-speech audio
  ("earcons") can reliably encode state and hierarchy, and rate-coded streams
  (the classic geiger counter) are an intuitive way to convey a changing scalar.
  SoundKey's cue language uses distinct, mutually distinguishable motifs and a
  rising-scale mapping for ordered categories, following auditory-display
  research such as Brewster's work on earcons and the community practice
  collected by the International Community for Auditory Display (ICAD),
  <https://icad.org/>.
- **The Flipper's real audio/haptic capabilities.** The cue library is tuned to
  the device's actual piezo range and vibro motor — see
  [`docs/prior_art.md`](prior_art.md) and the `audio` block in
  [`cues/cue_library.json`](../cues/cue_library.json).

## How to test it yourself (and the honest caveat)

You can experience the whole interface on a desktop, with no hardware:

```bash
cd sim
python3 -m soundkey_sim.cli flow locate_tag --play   # hear a full task
python3 -m soundkey_sim.cli play proximity --level 0.9 --play
```

### Eyes-closed testing is a proxy — say so

The evaluation in [`eval/`](../eval) includes a **screen-free task-completion**
measure. A sighted developer working eyes-closed (or the analytic model the
harness computes) is a **development proxy** — a cheap way to catch obvious
failures early. **It is not a substitute for testing with blind and low-vision
users.** Sighted people who lose vision temporarily bring visual mental models,
different navigation strategies, and no long-term muscle memory; their success
or speed does not predict a daily screen-reader user's experience.

So the numbers in the eval are labelled as proxies/models throughout, and the
project treats real user testing as required future work, not a box already
ticked. See [`eval/README.md`](../eval/README.md).

## Contributing feedback

If you are a blind or low-vision Flipper user (or work in assistive tech), your
feedback is the most valuable contribution this project can receive. Areas where
input is especially wanted:

- Are the cues distinguishable and pleasant over long sessions?
- Is the ten-gesture set genuinely learnable and consistent in practice?
- Which Flipper functions matter most to make accessible next?
- Does the verbosity/speech/tone balance work for your environment?

Please open an issue on the repository describing your setup and what worked or
didn't. Nothing here is fixed in stone; v1 is a starting point to build on
together.
