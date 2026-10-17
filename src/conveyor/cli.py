"""Command-line entry point for demo and measured experiments."""

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path

from conveyor.control import run_episode
from conveyor.experiment import run_batch, summarize_records, _save_charts
from conveyor.media import MediaRecorder
from conveyor.scene import Scene, PARTS


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-guided conveyor sorting")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run one filmed RGB-guided episode")
    demo.add_argument("--kind", choices=PARTS, default="cube")
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--output", type=Path, default=ROOT / "outputs" / "gripper_demo_cube")
    batch = commands.add_parser("batch", help="Run seeded measured episodes")
    batch.add_argument("--episodes", type=int, default=90)
    batch.add_argument("--seed", type=int, default=20260923)
    batch.add_argument("--belt-speed", type=float, default=0.12)
    batch.add_argument("--light-scale", type=float, default=1.0)
    batch.add_argument("--truth", action="store_true", help="Ground-truth position baseline")
    batch.add_argument("--output", type=Path, default=ROOT / "outputs" / "gripper_main")
    summary = commands.add_parser("summarize", help="Regenerate metrics/plots from CSV")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "demo":
        scene = Scene()
        recorder = MediaRecorder(arguments.output)
        try:
            result = run_episode(scene, arguments.kind, arguments.seed,
                                 use_vision=True, observer=recorder)
        finally:
            recorder.close()
            scene.close()
        (arguments.output / "result.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"result": asdict(result), "frames": recorder.frame_count},
                         ensure_ascii=False))
        return 0 if result.sorted else 1
    if arguments.command == "batch":
        summary = run_batch(arguments.episodes, arguments.output,
                            seed=arguments.seed, use_vision=not arguments.truth,
                            belt_speed=arguments.belt_speed,
                            light_scale=arguments.light_scale)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    with arguments.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    summary = summarize_records(rows)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_charts(rows, summary, arguments.output)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
