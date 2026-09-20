# SoundKey sound-cue legend

_Generated from `cues/cue_library.json` — do not edit by hand._

| Cue ID | State / result | Sound (tones · patterns) | Spoken caption | Group |
|---|---|---|---|---|
| `nav_move` | Move to next/previous item | 700Hz/30ms | `{item}` | navigation |
| `nav_boundary` | Edge of a list (no wrap) | 300Hz/40ms → ·30ms → 300Hz/40ms | `edge` | navigation |
| `select` | Select / activate item | 660Hz/45ms → ·20ms → 880Hz/60ms | `{item}, opening` | navigation |
| `back` | Go back / cancel | 660Hz/45ms → ·20ms → 440Hz/60ms | `back` | navigation |
| `home` | Returned to home / main menu | 523Hz/45ms → ·15ms → 659Hz/45ms → ·15ms → 784Hz/70ms | `home` | navigation |
| `repeat` | Repeat last announcement | 500Hz/25ms | `{item}` | navigation |
| `success` | Action succeeded | 660Hz/80ms → ·25ms → 990Hz/140ms | `done` | status |
| `error` | Action failed / not allowed | 340Hz/90ms → ·25ms → 200Hz/180ms | `error` | status |
| `action_start` | A running action started (e.g. scan begins) | 392Hz/55ms → ·18ms → 784Hz/75ms | `started` | status |
| `action_stop` | A running action stopped | 784Hz/55ms → ·18ms → 392Hz/75ms | `stopped` | status |
| `toggle_on` | Setting turned on | 740Hz/35ms → ·12ms → 740Hz/55ms | `{item} on` | status |
| `toggle_off` | Setting turned off | 466Hz/35ms → ·12ms → 466Hz/55ms | `{item} off` | status |
| `result_found` | Target found | 880Hz/60ms → ·20ms → 1180Hz/60ms → ·20ms → 1480Hz/90ms | `found {detail}` | result |
| `result_not_found` | Nothing found | 260Hz/220ms | `nothing found` | result |
| `proximity` | Proximity / signal strength (geiger style) | click stream, 1.0–25.0 Hz (quadratic in level) | `{percent} percent` | result |
| `level_pitch` | Continuous value as pitch | pitch 300–1200Hz over value | `{percent} percent` | result |
| `category_tone` | Category as distinct tone | note on scale 523/587/659/784/880/988 Hz | `category {category}` | result |
| `count_beeps` | Small count as beeps | 740Hz beep ×N (max 9) | `{count}` | result |
