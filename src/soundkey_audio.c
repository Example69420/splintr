// SoundKey on-device audio engine implementation.
// Author: Krishita Sanjay Choksi

#include "soundkey_audio.h"

#include <furi.h>
#include <furi_hal.h>

// Play a single tone for dur_ms (freq==0 -> silence). Assumes the speaker is
// already acquired by the caller so a whole cue plays without gaps.
static void play_step(uint16_t freq, uint16_t dur_ms, float volume, bool haptics) {
    if(freq > 0) {
        furi_hal_speaker_start(freq, volume);
        if(haptics) furi_hal_vibro_on(true);
    }
    furi_delay_ms(dur_ms);
    if(freq > 0) {
        furi_hal_speaker_stop();
        if(haptics) furi_hal_vibro_on(false);
    }
}

static bool acquire(void) {
    return furi_hal_speaker_is_mine() || furi_hal_speaker_acquire(30);
}

void soundkey_audio_play(const SoundKeyAudioConfig* cfg, int cue_index) {
    if(!cfg || !cfg->tones) return;
    if(cue_index < 0 || cue_index >= SOUNDKEY_CUE_COUNT) return;
    const SoundKeyCue* cue = &soundkey_cues[cue_index];
    if(cue->dynamic || cue->n_steps == 0) return; // dynamic cues have their own calls
    if(!acquire()) return;
    for(size_t i = 0; i < cue->n_steps; i++) {
        play_step(cue->steps[i].freq, cue->steps[i].dur_ms, cfg->volume, cfg->haptics);
    }
    furi_hal_speaker_release();
}

// Proximity: geiger-style click stream whose rate encodes closeness.
// Mirrors cues/cue_library.json proximity: rate = 1..25 Hz, quadratic in level.
void soundkey_audio_proximity(const SoundKeyAudioConfig* cfg, float level) {
    if(!cfg || !cfg->tones) return;
    if(level < 0.f) level = 0.f;
    if(level > 1.f) level = 1.f;
    const float min_rate = 1.f, max_rate = 25.f;
    float rate = min_rate + (max_rate - min_rate) * level * level;
    float period_ms = 1000.f / rate;
    const uint16_t click_freq = 1500, click_dur = 8;
    const float total_ms = 1500.f;
    if(!acquire()) return;
    float elapsed = 0.f;
    while(elapsed + click_dur <= total_ms) {
        play_step(click_freq, click_dur, cfg->volume, cfg->haptics);
        uint16_t gap = (uint16_t)(period_ms - click_dur > 0 ? period_ms - click_dur : 0);
        furi_delay_ms(gap);
        elapsed += click_dur + gap;
    }
    furi_hal_speaker_release();
}

void soundkey_audio_level(const SoundKeyAudioConfig* cfg, float level) {
    if(!cfg || !cfg->tones) return;
    if(level < 0.f) level = 0.f;
    if(level > 1.f) level = 1.f;
    uint16_t freq = (uint16_t)(300 + (1200 - 300) * level);
    if(!acquire()) return;
    play_step(freq, 260, cfg->volume, cfg->haptics);
    furi_hal_speaker_release();
}

void soundkey_audio_category(const SoundKeyAudioConfig* cfg, int category) {
    if(!cfg || !cfg->tones) return;
    static const uint16_t scale[] = {523, 587, 659, 784, 880, 988};
    int idx = category - 1;
    if(idx < 0) idx = 0;
    int octave = idx / 6;
    uint16_t note = scale[idx % 6] * (1 << octave);
    if(!acquire()) return;
    play_step(note, 120, cfg->volume, cfg->haptics);
    furi_hal_speaker_release();
}

void soundkey_audio_count(const SoundKeyAudioConfig* cfg, int count) {
    if(!cfg || !cfg->tones) return;
    if(count <= 0) return;
    if(count > 9) count = 9;
    if(!acquire()) return;
    for(int i = 0; i < count; i++) {
        play_step(740, 45, cfg->volume, cfg->haptics);
        if(i < count - 1) furi_delay_ms(90);
    }
    furi_hal_speaker_release();
}
