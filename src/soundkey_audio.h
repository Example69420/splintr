// SoundKey on-device audio engine.
// Plays cues from the generated cue table on the Flipper piezo speaker, and
// renders the dynamic (parameterised) cues -- proximity, level, category, count
// -- with the same logic the desktop simulator uses.
//
// Author: Krishita Sanjay Choksi
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "soundkey_cues.h"

// Playback volume (0..1). Kept modest by default -- accessible does not mean loud.
typedef struct {
    float volume;
    bool tones;    // play tonal cues
    bool haptics;  // pulse the vibro motor alongside cues
} SoundKeyAudioConfig;

// Play a static cue by index (SOUNDKEY_CUE_* constants). Blocks for the cue's
// duration. Safe to call when the speaker cannot be acquired (it simply skips).
void soundkey_audio_play(const SoundKeyAudioConfig* cfg, int cue_index);

// Dynamic cues. `level` is 0..1; `category` and `count` are small integers.
void soundkey_audio_proximity(const SoundKeyAudioConfig* cfg, float level);
void soundkey_audio_level(const SoundKeyAudioConfig* cfg, float level);
void soundkey_audio_category(const SoundKeyAudioConfig* cfg, int category);
void soundkey_audio_count(const SoundKeyAudioConfig* cfg, int count);
