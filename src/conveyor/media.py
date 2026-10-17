"""Offscreen video and image evidence for a simulated episode."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


STAGE_IMAGES = {
    "RESET": "reset", "DETECT": "detect", "PREGRASP": "pregrasp",
    "CLOSE_COMPLETE": "close", "LIFT": "lift", "TRANSFER": "transfer",
    "OPEN_COMPLETE": "open", "VERIFY": "verify",
}


class MediaRecorder:
    def __init__(self, output: Path, fps: int = 30):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._next_time = None
        self._seen_stages = set()
        self._writer = imageio.get_writer(
            str(self.output / "demo.mp4"), fps=fps, codec="libx264",
            pixelformat="yuv420p", quality=8, macro_block_size=1)
        self.predicted = "--"
        self.error_mm = None
        self.frame_count = 0

    def set_detection(self, predicted: str, error_m: float | None,
                      rgb: np.ndarray, contour: np.ndarray | None = None):
        self.predicted = predicted
        self.error_mm = None if error_m is None else error_m * 1000
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (0, 240, 240), 2)
            moments = cv2.moments(contour)
            if moments["m00"]:
                center = (round(moments["m10"] / moments["m00"]),
                          round(moments["m01"] / moments["m00"]))
                cv2.drawMarker(annotated, center, (0, 0, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(annotated, f"CLASS: {predicted}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(
            self.output / "detection.png")

    def __call__(self, scene, state: str):
        instant = float(scene.data.time)
        capture_stage = state in STAGE_IMAGES and state not in self._seen_stages
        if self._next_time is None:
            self._next_time = instant
        if not capture_stage and instant + 1e-9 < self._next_time:
            return
        frame = scene.rgb("overview", 1280, 720)
        frame = np.ascontiguousarray(frame)
        cv2.rectangle(frame, (14, 12), (465, 116), (18, 25, 32), -1)
        cv2.putText(frame, f"STATE  {state}", (28, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 245, 250), 2)
        cv2.putText(frame, f"PART {scene.kind.upper()}   VISION {self.predicted.upper()}",
                    (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 225, 235), 2)
        error = "--" if self.error_mm is None else f"{self.error_mm:.1f} mm"
        cv2.putText(frame, f"SIM {instant:.2f} s   POSITION ERROR {error}",
                    (28, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (195, 210, 225), 1)
        if capture_stage:
            Image.fromarray(frame).save(self.output / f"{STAGE_IMAGES[state]}.png")
            if state == "RESET":
                Image.fromarray(frame).save(self.output / "scene.png")
            self._seen_stages.add(state)
        if instant + 1e-9 >= self._next_time:
            self._writer.append_data(frame)
            self.frame_count += 1
            self._next_time += 1.0 / self.fps

    def close(self):
        self._writer.close()
