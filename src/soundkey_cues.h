// Auto-generated from cues/cue_library.json by cues/generate_c_header.py.
// Do not edit by hand. Single source of truth: cues/cue_library.json.
// Author: Krishita Sanjay Choksi
#pragma once
#include <stdint.h>
#include <stddef.h>

// A tone step: frequency in Hz (0 = silence) and duration in ms.
typedef struct {
    uint16_t freq;
    uint16_t dur_ms;
} SoundKeyStep;

typedef struct {
    const char* id;
    const char* speech;      // caption a TTS layer would speak
    const SoundKeyStep* steps;
    size_t n_steps;          // 0 for dynamic cues (rendered at runtime)
    uint8_t dynamic;         // 1 if parameterised (proximity/level/category/count)
} SoundKeyCue;

static const SoundKeyStep soundkey_steps_nav_move[] = {{700, 30}};
static const SoundKeyStep soundkey_steps_nav_boundary[] = {{300, 40}, {0, 30}, {300, 40}};
static const SoundKeyStep soundkey_steps_select[] = {{660, 45}, {0, 20}, {880, 60}};
static const SoundKeyStep soundkey_steps_back[] = {{660, 45}, {0, 20}, {440, 60}};
static const SoundKeyStep soundkey_steps_home[] = {{523, 45}, {0, 15}, {659, 45}, {0, 15}, {784, 70}};
static const SoundKeyStep soundkey_steps_repeat[] = {{500, 25}};
static const SoundKeyStep soundkey_steps_success[] = {{660, 80}, {0, 25}, {990, 140}};
static const SoundKeyStep soundkey_steps_error[] = {{340, 90}, {0, 25}, {200, 180}};
static const SoundKeyStep soundkey_steps_action_start[] = {{392, 55}, {0, 18}, {784, 75}};
static const SoundKeyStep soundkey_steps_action_stop[] = {{784, 55}, {0, 18}, {392, 75}};
static const SoundKeyStep soundkey_steps_toggle_on[] = {{740, 35}, {0, 12}, {740, 55}};
static const SoundKeyStep soundkey_steps_toggle_off[] = {{466, 35}, {0, 12}, {466, 55}};
static const SoundKeyStep soundkey_steps_result_found[] = {{880, 60}, {0, 20}, {1180, 60}, {0, 20}, {1480, 90}};
static const SoundKeyStep soundkey_steps_result_not_found[] = {{260, 220}};

static const SoundKeyCue soundkey_cues[] = {
    {"nav_move", "{item}", soundkey_steps_nav_move, 1, 0},
    {"nav_boundary", "edge", soundkey_steps_nav_boundary, 3, 0},
    {"select", "{item}, opening", soundkey_steps_select, 3, 0},
    {"back", "back", soundkey_steps_back, 3, 0},
    {"home", "home", soundkey_steps_home, 5, 0},
    {"repeat", "{item}", soundkey_steps_repeat, 1, 0},
    {"success", "done", soundkey_steps_success, 3, 0},
    {"error", "error", soundkey_steps_error, 3, 0},
    {"action_start", "started", soundkey_steps_action_start, 3, 0},
    {"action_stop", "stopped", soundkey_steps_action_stop, 3, 0},
    {"toggle_on", "{item} on", soundkey_steps_toggle_on, 3, 0},
    {"toggle_off", "{item} off", soundkey_steps_toggle_off, 3, 0},
    {"result_found", "found {detail}", soundkey_steps_result_found, 5, 0},
    {"result_not_found", "nothing found", soundkey_steps_result_not_found, 1, 0},
    {"proximity", "{percent} percent", NULL, 0, 1},
    {"level_pitch", "{percent} percent", NULL, 0, 1},
    {"category_tone", "category {category}", NULL, 0, 1},
    {"count_beeps", "{count}", NULL, 0, 1},
};

#define SOUNDKEY_CUE_COUNT 18

// Cue index constants for readable lookups in the app.
#define SOUNDKEY_CUE_NAV_MOVE 0
#define SOUNDKEY_CUE_NAV_BOUNDARY 1
#define SOUNDKEY_CUE_SELECT 2
#define SOUNDKEY_CUE_BACK 3
#define SOUNDKEY_CUE_HOME 4
#define SOUNDKEY_CUE_REPEAT 5
#define SOUNDKEY_CUE_SUCCESS 6
#define SOUNDKEY_CUE_ERROR 7
#define SOUNDKEY_CUE_ACTION_START 8
#define SOUNDKEY_CUE_ACTION_STOP 9
#define SOUNDKEY_CUE_TOGGLE_ON 10
#define SOUNDKEY_CUE_TOGGLE_OFF 11
#define SOUNDKEY_CUE_RESULT_FOUND 12
#define SOUNDKEY_CUE_RESULT_NOT_FOUND 13
#define SOUNDKEY_CUE_PROXIMITY 14
#define SOUNDKEY_CUE_LEVEL_PITCH 15
#define SOUNDKEY_CUE_CATEGORY_TONE 16
#define SOUNDKEY_CUE_COUNT_BEEPS 17

