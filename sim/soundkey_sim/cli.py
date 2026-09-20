"""SoundKey simulator command line.

Run from the ``sim/`` directory:

    python3 -m soundkey_sim.cli list-cues
    python3 -m soundkey_sim.cli play proximity --level 0.9 --play
    python3 -m soundkey_sim.cli flow locate_tag --out /tmp/locate.wav --play
    python3 -m soundkey_sim.cli legend
    python3 -m soundkey_sim.cli export-demos ../media

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from . import __version__
from .cues import CueLibrary, load_gesture_map
from .engine import AudioEngine, Settings
from .flows import DEMO_FLOWS, build_app, run_flow
from .nav import InputHandler, NavSession
from .sonifier import Sonifier, SoundEvent
from .synth import write_wav
from . import legend as legend_mod


def _play(path: str) -> None:
    for player in ("afplay", "aplay", "play", "ffplay"):
        exe = shutil.which(player)
        if exe:
            args = [exe, path]
            if player == "ffplay":
                args = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
            subprocess.run(args, check=False)
            return
    print(f"(no audio player found; wrote {path} -- open it manually)", file=sys.stderr)


def _sample_event(cue_id: str, args) -> SoundEvent:
    son = Sonifier()
    if cue_id == "proximity":
        return son.proximity(args.level)
    if cue_id == "level_pitch":
        return son.level(args.level)
    if cue_id == "category_tone":
        return son.category(args.category)
    if cue_id == "count_beeps":
        return son.count(args.count)
    return SoundEvent(cue_id, {}, CueLibrary()[cue_id].speech_template)


def cmd_list_cues(args) -> int:
    lib = CueLibrary()
    print(f"SoundKey cue library v{lib.meta['version']} ({len(lib.cues)} cues)\n")
    width = max(len(c.id) for c in lib.cues.values())
    for c in lib.cues.values():
        print(f"  {c.id.ljust(width)}  [{c.group}] {c.label}")
    return 0


def cmd_play(args) -> int:
    lib = CueLibrary()
    if args.cue not in lib.cues:
        print(f"unknown cue: {args.cue}", file=sys.stderr)
        return 2
    engine = AudioEngine(Settings(verbosity="verbose"), lib)
    ev = _sample_event(args.cue, args)
    samples, transcript = engine.render([ev])
    out = args.out or os.path.join(_default_outdir(), f"cue_{args.cue}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, samples)
    print(engine.transcript_text(transcript, title=f"cue {args.cue}"))
    print(f"wrote {out}")
    if args.play:
        _play(out)
    return 0


def cmd_flows(args) -> int:
    for f in DEMO_FLOWS:
        print(f"  {f.name:16s} {f.description}")
    return 0


def cmd_flow(args) -> int:
    flow = next((f for f in DEMO_FLOWS if f.name == args.name), None)
    if flow is None:
        print(f"unknown flow: {args.name}", file=sys.stderr)
        return 2
    settings = Settings(verbosity=args.verbosity, speech=not args.no_speech,
                        tones=not args.no_tones)
    engine = AudioEngine(settings)
    handler = InputHandler(load_gesture_map())
    session = NavSession(build_app())
    events = run_flow(flow, session, handler)
    samples, transcript = engine.render(events)
    out = args.out or os.path.join(_default_outdir(), f"flow_{flow.name}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, samples)
    print(engine.transcript_text(transcript, title=flow.name))
    print(f"wrote {out}")
    if args.play:
        _play(out)
    return 0


def cmd_legend(args) -> int:
    print(legend_mod.legend_markdown())
    return 0


def cmd_gestures(args) -> int:
    print(legend_mod.gesture_markdown())
    return 0


def cmd_export_demos(args) -> int:
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    handler = InputHandler(load_gesture_map())
    manifest = []
    # Flows
    for flow in DEMO_FLOWS:
        engine = AudioEngine(Settings(verbosity="normal"))
        session = NavSession(build_app())
        events = run_flow(flow, session, handler)
        samples, transcript = engine.render(events)
        wav = os.path.join(outdir, f"flow_{flow.name}.wav")
        txt = os.path.join(outdir, f"flow_{flow.name}.txt")
        write_wav(wav, samples)
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(engine.transcript_text(transcript, title=flow.name))
        manifest.append((os.path.basename(wav), flow.description))
    # One WAV per cue at representative parameters
    lib = CueLibrary()
    for cue_id in lib.cues:
        engine = AudioEngine(Settings(verbosity="verbose"), lib)

        class _A:  # representative dynamic params
            level = 0.85
            category = 3
            count = 3
        ev = _sample_event(cue_id, _A)
        samples, _ = engine.render([ev])
        write_wav(os.path.join(outdir, f"cue_{cue_id}.wav"), samples)
    with open(os.path.join(outdir, "MANIFEST.md"), "w", encoding="utf-8") as fh:
        fh.write("# SoundKey audio demos\n\n")
        fh.write("Generated by `soundkey_sim.cli export-demos`. Regenerable from a clean clone.\n\n")
        fh.write("## Flows\n\n")
        for name, desc in manifest:
            fh.write(f"- **{name}** — {desc}\n")
        fh.write("\n## Individual cues\n\n")
        fh.write("One `cue_<id>.wav` per cue in the library (see the sound-cue legend).\n")
    print(f"exported {len(DEMO_FLOWS)} flows + {len(lib.cues)} cue samples to {outdir}")
    return 0


def _default_outdir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "media"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soundkey_sim", description="SoundKey desktop simulator")
    p.add_argument("--version", action="version", version=f"soundkey_sim {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-cues").set_defaults(func=cmd_list_cues)

    pp = sub.add_parser("play", help="render a single cue")
    pp.add_argument("cue")
    pp.add_argument("--level", type=float, default=0.8)
    pp.add_argument("--category", type=int, default=3)
    pp.add_argument("--count", type=int, default=3)
    pp.add_argument("--out")
    pp.add_argument("--play", action="store_true")
    pp.set_defaults(func=cmd_play)

    sub.add_parser("flows", help="list demo flows").set_defaults(func=cmd_flows)

    pf = sub.add_parser("flow", help="render a demo flow")
    pf.add_argument("name")
    pf.add_argument("--out")
    pf.add_argument("--play", action="store_true")
    pf.add_argument("--verbosity", default="normal", choices=["silent", "terse", "normal", "verbose"])
    pf.add_argument("--no-speech", action="store_true")
    pf.add_argument("--no-tones", action="store_true")
    pf.set_defaults(func=cmd_flow)

    sub.add_parser("legend", help="print the sound-cue legend (markdown)").set_defaults(func=cmd_legend)
    sub.add_parser("gestures", help="print the gesture map (markdown)").set_defaults(func=cmd_gestures)

    pe = sub.add_parser("export-demos", help="render all flows + cues to a directory")
    pe.add_argument("outdir")
    pe.set_defaults(func=cmd_export_demos)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
