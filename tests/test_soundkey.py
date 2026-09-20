"""SoundKey test suite (standard-library unittest, no dependencies).

Run from the repo root:  python3 -m unittest discover -s tests -v

Author: Krishita Sanjay Choksi
"""

import io
import os
import sys
import unittest
import wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from soundkey_sim.cues import CueLibrary, load_gesture_map  # noqa: E402
from soundkey_sim.engine import AudioEngine, Settings  # noqa: E402
from soundkey_sim.flows import DEMO_FLOWS, build_app, run_flow  # noqa: E402
from soundkey_sim.nav import InputHandler, NavSession  # noqa: E402
from soundkey_sim.sonifier import Sonifier  # noqa: E402
from soundkey_sim.synth import samples_to_wav_bytes, duration_ms  # noqa: E402


class TestCueLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = CueLibrary()

    def test_loads_and_unique_ids(self):
        self.assertGreaterEqual(len(self.lib.cues), 15)
        ids = list(self.lib.cues.keys())
        self.assertEqual(len(ids), len(set(ids)), "cue ids must be unique")

    def test_static_cue_segments(self):
        segs = self.lib.segments_for("select")
        self.assertTrue(any(not s.is_silence for s in segs))

    def test_proximity_rate_increases_with_level(self):
        near = self.lib.segments_for("proximity", level=0.95)
        far = self.lib.segments_for("proximity", level=0.1)
        near_clicks = sum(1 for s in near if not s.is_silence)
        far_clicks = sum(1 for s in far if not s.is_silence)
        self.assertGreater(near_clicks, far_clicks,
                           "closer target must click faster (more clicks)")

    def test_level_pitch_monotonic(self):
        lo = self.lib.segments_for("level_pitch", level=0.1)[0].freq
        hi = self.lib.segments_for("level_pitch", level=0.9)[0].freq
        self.assertGreater(hi, lo)

    def test_category_scale_and_octave_wrap(self):
        c1 = self.lib.segments_for("category_tone", category=1)[0].freq
        c3 = self.lib.segments_for("category_tone", category=3)[0].freq
        self.assertGreater(c3, c1)
        c7 = self.lib.segments_for("category_tone", category=7)[0].freq  # wraps up an octave
        self.assertAlmostEqual(c7, c1 * 2, delta=1.0)

    def test_count_beeps(self):
        segs = self.lib.segments_for("count_beeps", count=3)
        beeps = sum(1 for s in segs if not s.is_silence)
        self.assertEqual(beeps, 3)


class TestSynth(unittest.TestCase):
    def test_wav_roundtrip_duration(self):
        lib = CueLibrary()
        segs = lib.segments_for("home")
        engine = AudioEngine(Settings(verbosity="verbose"), lib)
        # render just this cue via segments
        from soundkey_sim.synth import render_segments
        samples = render_segments(segs)
        wav = samples_to_wav_bytes(samples)
        w = wave.open(io.BytesIO(wav))
        self.assertEqual(w.getframerate(), 44100)
        self.assertEqual(w.getnchannels(), 1)
        # duration within a frame or two of the segment total
        self.assertAlmostEqual(w.getnframes() / 44100.0, duration_ms(segs) / 1000.0, places=1)


class TestEngineVerbosity(unittest.TestCase):
    def test_silent_suppresses_normal_floor_cue(self):
        son = Sonifier()
        # proximity has verbosity_floor 'normal'; at 'terse' it should be suppressed
        engine = AudioEngine(Settings(verbosity="terse"))
        _, t = engine.render([son.proximity(0.5)])
        self.assertEqual(len(t), 0)
        engine2 = AudioEngine(Settings(verbosity="normal"))
        _, t2 = engine2.render([son.proximity(0.5)])
        self.assertEqual(len(t2), 1)


class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.s = NavSession(build_app())

    def test_next_prev_and_boundary(self):
        ev = self.s.do("prev_item")  # already at top
        self.assertEqual(ev[0].cue_id, "nav_boundary")
        self.s.do("next_item")
        self.assertEqual(self.s.current_item.label, "Readouts")

    def test_select_enter_and_back(self):
        self.s.do("select")  # enter Scanners
        self.assertEqual(self.s.current_menu.label, "Scanners")
        ev = self.s.do("back")
        self.assertEqual(ev[0].cue_id, "back")
        self.assertEqual(self.s.current_menu.label, "SoundKey")

    def test_home_from_depth(self):
        self.s.do("select")
        self.s.do("select")  # run an action leaf (Proximity locator)
        self.s.do("go_home")
        self.assertEqual(self.s.current_menu.label, "SoundKey")


class TestGestures(unittest.TestCase):
    def test_gesture_table_consistent(self):
        gm = load_gesture_map()
        handler = InputHandler(gm)
        self.assertEqual(handler.resolve("OK", "short"), "select")
        self.assertEqual(handler.resolve("Back", "long"), "go_home")
        # every (button, press) maps to exactly one action
        seen = {}
        for g in gm["gestures"]:
            key = (g["button"], g["press"])
            self.assertNotIn(key, seen, "a gesture must have a single meaning")
            seen[key] = g["action"]


class TestFlows(unittest.TestCase):
    def test_all_flows_run(self):
        handler = InputHandler(load_gesture_map())
        for flow in DEMO_FLOWS:
            session = NavSession(build_app())
            events = run_flow(flow, session, handler)
            self.assertTrue(events, f"{flow.name} produced no events")

    def test_locate_tag_has_proximity_and_found(self):
        handler = InputHandler(load_gesture_map())
        flow = next(f for f in DEMO_FLOWS if f.name == "locate_tag")
        session = NavSession(build_app())
        events = run_flow(flow, session, handler)
        cue_ids = [e.cue_id for e in events]
        self.assertIn("proximity", cue_ids)
        self.assertIn("result_found", cue_ids)


class TestDistinguishability(unittest.TestCase):
    def test_no_confusable_pairs(self):
        import run_eval
        d = run_eval.distinguishability()
        self.assertEqual(len(d["flagged_pairs"]), 0,
                         f"cues too similar: {d['flagged_pairs']}")
        self.assertGreater(d["min_distance"], d["confusable_threshold"])


class TestCHeaderInSync(unittest.TestCase):
    def test_every_cue_in_generated_header(self):
        header = os.path.join(ROOT, "src", "soundkey_cues.h")
        with open(header, encoding="utf-8") as f:
            text = f.read()
        for cue_id in CueLibrary().cues:
            self.assertIn(f'"{cue_id}"', text,
                          f"{cue_id} missing from generated C header; run "
                          f"cues/generate_c_header.py")


if __name__ == "__main__":
    unittest.main()
