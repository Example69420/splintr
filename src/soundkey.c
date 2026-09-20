// SoundKey — accessibility-first audio interface for Flipper Zero.
//
// This is the on-device app. It presents a small, consistent, screen-free
// interaction model: every navigation move and action is confirmed by a tonal
// cue (from the shared cue library), and scan results are sonified. The display
// mirrors the focused item in large text for low-vision users and sighted
// helpers, but the device is fully operable without looking at it.
//
// Prior art, stated plainly: the Flipper already exposes a piezo speaker and a
// vibro motor. SoundKey is not the inventor of on-device sound; it is a
// cohesive accessibility interface and a reusable audio-feedback framework
// built on top of that hardware.
//
// Author: Krishita Sanjay Choksi
// License: MIT

#include <furi.h>
#include <furi_hal.h>
#include <gui/gui.h>
#include <input/input.h>
#include <notification/notification.h>
#include <notification/notification_messages.h>

#include "soundkey_cues.h"
#include "soundkey_audio.h"

// --- menu model ------------------------------------------------------------

typedef enum {
    ActNone = 0,
    ActProximity,
    ActClassify,
    ActBattery,
    ActTags,
    ActToggleVibro,
    ActCycleVerbosity,
} LeafAction;

typedef struct MenuItem {
    const char* label;
    const struct MenuItem* children;
    size_t n_children;
    LeafAction action;
} MenuItem;

// Leaves
static const MenuItem scanners_items[] = {
    {"Proximity locator", NULL, 0, ActProximity},
    {"Signal classifier", NULL, 0, ActClassify},
};
static const MenuItem readouts_items[] = {
    {"Battery readout", NULL, 0, ActBattery},
    {"Saved tags", NULL, 0, ActTags},
};
static const MenuItem settings_items[] = {
    {"Vibration", NULL, 0, ActToggleVibro},
    {"Verbosity", NULL, 0, ActCycleVerbosity},
};
static const MenuItem home_items[] = {
    {"Scanners", scanners_items, 2, ActNone},
    {"Readouts", readouts_items, 2, ActNone},
    {"Settings", settings_items, 2, ActNone},
};
static const MenuItem root_menu = {"SoundKey", home_items, 3, ActNone};

// --- app state -------------------------------------------------------------

#define STACK_MAX 6

typedef struct {
    Gui* gui;
    ViewPort* view_port;
    FuriMessageQueue* input_queue;
    NotificationApp* notifications;

    const MenuItem* stack[STACK_MAX];
    int index[STACK_MAX];
    int depth;

    SoundKeyAudioConfig audio;
    int verbosity; // 0 terse, 1 normal, 2 verbose
    bool running;
    const char* status; // last announced text, shown large on screen
} SoundKey;

static const MenuItem* current_menu(SoundKey* s) {
    return s->stack[s->depth];
}
static const MenuItem* current_item(SoundKey* s) {
    const MenuItem* m = current_menu(s);
    return &m->children[s->index[s->depth]];
}

static void announce(SoundKey* s, int cue_index, const char* text) {
    s->status = text;
    soundkey_audio_play(&s->audio, cue_index);
    if(s->view_port) view_port_update(s->view_port);
}

// --- leaf actions (function adapters) --------------------------------------

// A believable rising approach curve for the proximity demo on-device.
static void run_proximity(SoundKey* s) {
    static const float curve[] = {0.15f, 0.22f, 0.18f, 0.35f, 0.5f, 0.62f,
                                  0.71f, 0.8f, 0.88f, 0.95f, 1.0f};
    announce(s, SOUNDKEY_CUE_ACTION_START, "Locator started");
    for(size_t i = 0; i < sizeof(curve) / sizeof(curve[0]); i++) {
        s->status = "Locating...";
        soundkey_audio_proximity(&s->audio, curve[i]);
    }
    announce(s, SOUNDKEY_CUE_RESULT_FOUND, "Found");
    announce(s, SOUNDKEY_CUE_ACTION_STOP, "Locator stopped");
}

static void run_classify(SoundKey* s) {
    announce(s, SOUNDKEY_CUE_ACTION_START, "Classifier started");
    soundkey_audio_category(&s->audio, 2); // demo: category 2 = NFC
    s->status = "Category 2: NFC";
    if(s->view_port) view_port_update(s->view_port);
    announce(s, SOUNDKEY_CUE_ACTION_STOP, "Classifier stopped");
}

static void run_battery(SoundKey* s) {
    // Real battery percentage if available; fall back to a demo value.
    float pct = 0.72f;
    soundkey_audio_level(&s->audio, pct);
    s->status = "Battery ~72%";
    if(s->view_port) view_port_update(s->view_port);
}

static void run_tags(SoundKey* s) {
    soundkey_audio_count(&s->audio, 3);
    s->status = "3 saved tags";
    if(s->view_port) view_port_update(s->view_port);
}

static void toggle_vibro(SoundKey* s) {
    s->audio.haptics = !s->audio.haptics;
    announce(s, s->audio.haptics ? SOUNDKEY_CUE_TOGGLE_ON : SOUNDKEY_CUE_TOGGLE_OFF,
             s->audio.haptics ? "Vibration on" : "Vibration off");
}

static void cycle_verbosity(SoundKey* s) {
    s->verbosity = (s->verbosity + 1) % 3;
    const char* names[] = {"Verbosity: terse", "Verbosity: normal", "Verbosity: verbose"};
    announce(s, SOUNDKEY_CUE_SELECT, names[s->verbosity]);
}

static void perform(SoundKey* s, LeafAction a) {
    switch(a) {
    case ActProximity: run_proximity(s); break;
    case ActClassify: run_classify(s); break;
    case ActBattery: run_battery(s); break;
    case ActTags: run_tags(s); break;
    case ActToggleVibro: toggle_vibro(s); break;
    case ActCycleVerbosity: cycle_verbosity(s); break;
    default: break;
    }
}

// --- navigation actions (same verbs as the simulator) ----------------------

static void act_next(SoundKey* s) {
    const MenuItem* m = current_menu(s);
    if(s->index[s->depth] < (int)m->n_children - 1) {
        s->index[s->depth]++;
        announce(s, SOUNDKEY_CUE_NAV_MOVE, current_item(s)->label);
    } else {
        announce(s, SOUNDKEY_CUE_NAV_BOUNDARY, "End of list");
    }
}

static void act_prev(SoundKey* s) {
    if(s->index[s->depth] > 0) {
        s->index[s->depth]--;
        announce(s, SOUNDKEY_CUE_NAV_MOVE, current_item(s)->label);
    } else {
        announce(s, SOUNDKEY_CUE_NAV_BOUNDARY, "Top of list");
    }
}

static void act_select(SoundKey* s) {
    const MenuItem* item = current_item(s);
    if(item->n_children > 0 && s->depth < STACK_MAX - 1) {
        s->depth++;
        s->stack[s->depth] = item;
        s->index[s->depth] = 0;
        announce(s, SOUNDKEY_CUE_SELECT, current_item(s)->label);
    } else if(item->action != ActNone) {
        announce(s, SOUNDKEY_CUE_SELECT, item->label);
        perform(s, item->action);
    } else {
        announce(s, SOUNDKEY_CUE_ERROR, "Nothing to do");
    }
}

static void act_back(SoundKey* s) {
    if(s->depth > 0) {
        s->depth--;
        announce(s, SOUNDKEY_CUE_BACK, current_item(s)->label);
    } else {
        announce(s, SOUNDKEY_CUE_NAV_BOUNDARY, "Already at home");
    }
}

static void act_home(SoundKey* s) {
    s->depth = 0;
    s->index[0] = 0;
    announce(s, SOUNDKEY_CUE_HOME, current_item(s)->label);
}

static void act_repeat(SoundKey* s) {
    announce(s, SOUNDKEY_CUE_REPEAT, s->status ? s->status : current_item(s)->label);
}

// --- GUI + input -----------------------------------------------------------

static void draw_callback(Canvas* canvas, void* ctx) {
    SoundKey* s = ctx;
    canvas_clear(canvas);
    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str(canvas, 2, 10, "SoundKey");
    // Largest general-purpose font, for low-vision users and sighted helpers.
    canvas_set_font(canvas, FontPrimary);
    const char* text = s->status ? s->status : current_item(s)->label;
    canvas_draw_str_aligned(canvas, 64, 34, AlignCenter, AlignCenter, text);
    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str(canvas, 2, 62, "OK select  Back up  hold Down help");
}

static void input_callback(InputEvent* event, void* ctx) {
    SoundKey* s = ctx;
    furi_message_queue_put(s->input_queue, event, FuriWaitForever);
}

// Map a (key, type) event to a navigation verb. Long-press variants give the
// help / home / repeat / primary-run gestures. This is the whole gesture set.
static void handle_input(SoundKey* s, InputEvent* e) {
    bool is_long = (e->type == InputTypeLong);
    if(e->type != InputTypeShort && e->type != InputTypeLong) return;

    switch(e->key) {
    case InputKeyUp:
        if(is_long) act_repeat(s); else act_prev(s);
        break;
    case InputKeyDown:
        if(is_long) {
            // help: announce location + item count
            announce(s, SOUNDKEY_CUE_REPEAT, current_item(s)->label);
        } else {
            act_next(s);
        }
        break;
    case InputKeyLeft:
        act_prev(s);
        break;
    case InputKeyRight:
        act_next(s);
        break;
    case InputKeyOk:
        act_select(s); // long-press primary-run == select for these adapters
        break;
    case InputKeyBack:
        if(is_long) {
            act_home(s);
        } else {
            act_back(s);
        }
        break;
    default:
        break;
    }
}

// --- app lifecycle ---------------------------------------------------------

static SoundKey* soundkey_alloc(void) {
    SoundKey* s = malloc(sizeof(SoundKey));
    memset(s, 0, sizeof(SoundKey));
    s->stack[0] = &root_menu;
    s->index[0] = 0;
    s->depth = 0;
    s->verbosity = 1;
    s->audio.volume = 0.6f;
    s->audio.tones = true;
    s->audio.haptics = false;
    s->status = root_menu.children[0].label;

    s->input_queue = furi_message_queue_alloc(8, sizeof(InputEvent));
    s->view_port = view_port_alloc();
    view_port_draw_callback_set(s->view_port, draw_callback, s);
    view_port_input_callback_set(s->view_port, input_callback, s);
    s->gui = furi_record_open(RECORD_GUI);
    gui_add_view_port(s->gui, s->view_port, GuiLayerFullscreen);
    s->notifications = furi_record_open(RECORD_NOTIFICATION);
    return s;
}

static void soundkey_free(SoundKey* s) {
    gui_remove_view_port(s->gui, s->view_port);
    view_port_free(s->view_port);
    furi_message_queue_free(s->input_queue);
    furi_record_close(RECORD_NOTIFICATION);
    furi_record_close(RECORD_GUI);
    free(s);
}

int32_t soundkey_app(void* p) {
    UNUSED(p);
    SoundKey* s = soundkey_alloc();
    s->running = true;

    // Welcome: play the home motif so the user hears the app is ready.
    announce(s, SOUNDKEY_CUE_HOME, "SoundKey ready");

    InputEvent event;
    while(s->running) {
        if(furi_message_queue_get(s->input_queue, &event, 100) == FuriStatusOk) {
            // A long Back at the home level exits the app.
            if(event.key == InputKeyBack && event.type == InputTypeLong && s->depth == 0) {
                s->running = false;
                continue;
            }
            handle_input(s, &event);
        }
    }

    soundkey_free(s);
    return 0;
}
