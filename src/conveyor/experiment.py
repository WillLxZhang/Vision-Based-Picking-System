"""Experiment metric definitions and reproducible batch runners."""

from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from conveyor.control import run_episode
from conveyor.scene import Scene, PARTS


def _is_true(value) -> bool:
    return value is True or str(value).lower() in ("true", "1")


def _optional_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def _mean_optional(rows: list[dict], key: str) -> float | None:
    values = _optional_values(rows, key)
    return float(np.mean(values)) if values else None


def summarize_records(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No episode records")
    total = len(rows)
    errors = [float(row["error_m"]) * 1000 for row in rows
              if row.get("error_m") not in (None, "")]
    classes = sorted({row["kind"] for row in rows})
    return {
        "episodes": total,
        "recognition_accuracy": sum(row.get("predicted_kind") == row["kind"]
                                    for row in rows) / total,
        "grasp_success_rate": sum(_is_true(row["grasped"]) for row in rows) / total,
        "sort_success_rate": sum(_is_true(row["sorted"]) for row in rows) / total,
        "position_mae_mm": float(np.mean(errors)) if errors else None,
        "mean_cycle_s": float(np.mean([float(row["elapsed"]) for row in rows])),
        "mean_max_object_angular_speed_rad_s": _mean_optional(
            rows, "max_object_angular_speed_rad_s"),
        "mean_max_relative_orientation_deg": _mean_optional(
            rows, "max_relative_orientation_deg"),
        "max_object_angular_speed_rad_s": max(
            _optional_values(rows, "max_object_angular_speed_rad_s"), default=None),
        "max_relative_orientation_deg": max(
            _optional_values(rows, "max_relative_orientation_deg"), default=None),
        "class_counts": dict(Counter(row["kind"] for row in rows)),
        "failures": dict(Counter(row.get("failure") or "success" for row in rows)),
        "confusion": {truth: {predicted: sum(
            row["kind"] == truth and (row.get("predicted_kind") or "unknown") == predicted
            for row in rows) for predicted in classes + ["unknown"]} for truth in classes},
    }


def _save_charts(rows: list[dict], summary: dict, output: Path):
    classes = list(PARTS)
    confusion = np.array([[summary["confusion"][truth].get(prediction, 0)
                           for prediction in classes + ["unknown"]]
                          for truth in classes])
    figure, axes = plt.subplots(figsize=(7, 4.8))
    visual = axes.imshow(confusion, cmap="Blues")
    axes.set_xticks(range(4), classes + ["unknown"])
    axes.set_yticks(range(3), classes)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    for i in range(3):
        for j in range(4):
            axes.text(j, i, int(confusion[i, j]), ha="center", va="center")
    figure.colorbar(visual, ax=axes)
    figure.tight_layout()
    figure.savefig(output / "confusion.png", dpi=180)
    plt.close(figure)

    errors = [float(row["error_m"]) * 1000 for row in rows
              if row.get("error_m") not in (None, "")]
    figure, axes = plt.subplots(figsize=(7, 4.2))
    axes.hist(errors, bins=min(12, max(3, len(errors) // 4)), color="#2474a5")
    axes.set_xlabel("Planar position error (mm)")
    axes.set_ylabel("Episodes")
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "position_error.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7, 4.2))
    names = ["Recognition", "Grasp", "Sorted"]
    rates = [summary["recognition_accuracy"], summary["grasp_success_rate"],
             summary["sort_success_rate"]]
    bars = axes.bar(names, rates, color=["#2980b9", "#16a085", "#e67e22"])
    axes.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates])
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("Success rate")
    axes.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "success_rates.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7, 4.2))
    times = np.array([float(row["elapsed"]) for row in rows])
    axes.plot(np.arange(1, len(times) + 1), times, ".", color="#8e44ad")
    axes.axhline(float(np.mean(times)), color="#c0392b", linestyle="--",
                 label=f"Mean {np.mean(times):.2f} s")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Cycle time (s)")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "cycle_time.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, key, label, color in (
        (axes[0], "max_object_angular_speed_rad_s", "Peak angular speed (rad/s)", "#c0392b"),
        (axes[1], "max_relative_orientation_deg", "Peak relative rotation (deg)", "#8e44ad"),
    ):
        values = _optional_values(rows, key)
        if values:
            axis.hist(values, bins=min(12, max(3, len(values) // 4)), color=color)
        else:
            axis.text(0.5, 0.5, "No lift telemetry", transform=axis.transAxes,
                      ha="center", va="center")
        axis.set_xlabel(label)
        axis.set_ylabel("Episodes with lift telemetry")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "rotation.png", dpi=180)
    plt.close(figure)


def run_batch(episodes: int, output: Path, seed: int = 20260923,
              use_vision: bool = True, belt_speed: float = 0.12,
              light_scale: float = 1.0) -> dict:
    if episodes < 1 or belt_speed <= 0:
        raise ValueError("episodes and belt_speed must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    failure_image = output / "failure_case.png"
    failure_image.unlink(missing_ok=True)
    failure_saved = False
    scene = Scene()
    light = scene.model.light("work_light").id
    scene.model.light_diffuse[light] = [light_scale] * 3
    rows = []
    try:
        for index in range(episodes):
            kind = PARTS[index % len(PARTS)]
            trial_seed = seed + index
            result = run_episode(scene, kind, trial_seed, use_vision=use_vision,
                                 belt_speed=belt_speed)
            record = asdict(result)
            record["seed"] = trial_seed
            record["belt_speed"] = belt_speed
            record["light_scale"] = light_scale
            record["states"] = json.dumps(record["states"])
            rows.append(record)
            if result.failure and not failure_saved:
                Image.fromarray(scene.rgb("overview", 1280, 720)).save(
                    failure_image)
                failure_saved = True
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize_records(rows)
        summary["condition"] = {"seed": seed, "use_vision": use_vision,
                                "belt_speed": belt_speed, "light_scale": light_scale}
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_charts(rows, summary, output)
        return summary
    finally:
        scene.close()
