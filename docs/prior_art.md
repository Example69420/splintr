# Prior art and honest positioning

Accessibility work earns trust by being honest about what is and isn't new.
This document states plainly what already existed before SoundKey and what
SoundKey actually contributes.

## What already existed (and deserves credit)

- **The Flipper Zero can already make sound and vibrate.** It has a piezo
  speaker driven through `furi_hal_speaker` (tones at a given frequency and
  volume) and a vibration motor via `furi_hal_vibro`. Many stock apps and games
  beep, and the firmware ships a notification/tone system. **SoundKey did not
  invent on-device audio or haptics on the Flipper.**
- **Rate-coded "geiger" audio feedback is a known idea.** Passive-scanner style
  tools and countless assistive devices have used click-rate to convey
  proximity or signal strength. SoundKey generalises that idiom; it does not
  claim to have originated it.
- **Screen readers and auditory UIs are decades old.** VoiceOver, TalkBack,
  NVDA, JAWS, and the auditory-display research community established the
  conventions SoundKey adapts (announce-on-focus, earcons, sonification). See
  [`docs/accessibility.md`](accessibility.md) for citations.
- **Text-to-speech exists generally.** Speech synthesis is a mature field.
  On-device TTS on the Flipper's hardware is constrained, which is why SoundKey's
  v1 leans on a tonal cue language and treats richer speech as a roadmap item.

## What SoundKey actually contributes

SoundKey is, to the author's knowledge, **the first cohesive accessibility
*interface* and reusable audio-feedback *framework* for the Flipper Zero** — not
the inventor of any single primitive it uses. Its original contributions are:

1. **A coherent, learnable, screen-free interaction model** for the device: a
   fixed ten-gesture control scheme and a navigation model that announces,
   confirms, and signals edges — a designed non-visual way to operate the
   Flipper, rather than scattered beeps.
2. **A reusable audio-feedback framework** other Flipper apps can adopt, turning
   states, menus, and results into consistent sound. See
   [`docs/framework_guide.md`](framework_guide.md).
3. **A documented, distinguishability-checked sound-cue library** — the mapping
   of states and results to tones, patterns, speech, and haptics, published as a
   legible artifact and verified by an acoustic-distance analysis (see
   [`eval/`](../eval)).
4. **A desktop simulator** so the interface can be experienced, reviewed, and
   improved without hardware — important for reproducibility and for sighted
   collaborators building accessibility for people they may not be.
5. **A community accessibility guideline** documenting how to make Flipper apps
   accessible, extending the impact beyond this one app.

## The honest one-liner

> SoundKey is the first cohesive accessibility interface and audio-feedback
> framework for the Flipper Zero. It is **not** the inventor of "the Flipper can
> beep." It stands on the device's existing sound/haptic hardware and on decades
> of assistive-technology practice, and turns them into something a blind or
> low-vision person can actually use end to end.
