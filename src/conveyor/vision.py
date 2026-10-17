"""RGB-only part detection and belt-plane calibration."""

from dataclasses import dataclass

import cv2
import numpy as np


HSV_RANGES = {
    "cube": ((0, 90, 50), (12, 255, 255)),
    "cylinder": ((34, 65, 40), (85, 255, 255)),
    "prism": ((90, 80, 45), (135, 255, 255)),
}


@dataclass(frozen=True)
class Detection:
    kind: str
    pixel_xy: tuple[float, float]
    world_xy: tuple[float, float]
    area: float
    confidence: float
    shape: str
    contour: np.ndarray


def calibrate(pixel_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixel_points, dtype=np.float32)
    world = np.asarray(world_points, dtype=np.float32)
    if pixels.shape != (4, 2) or world.shape != (4, 2):
        raise ValueError("Calibration requires exactly four pixel/world pairs")
    matrix = cv2.getPerspectiveTransform(pixels, world)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Degenerate calibration points")
    return matrix


def _mask(hsv: np.ndarray, low: tuple[int, int, int],
          high: tuple[int, int, int]) -> np.ndarray:
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8),
                       np.array(high, dtype=np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _shape(contour: np.ndarray) -> tuple[str, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
    vertices = len(approx)
    if vertices == 3:
        return "triangle", float(circularity)
    if vertices == 4:
        return "square", float(circularity)
    if circularity > 0.7:
        return "circle", float(circularity)
    return "irregular", float(circularity)


def detect(rgb: np.ndarray, homography: np.ndarray) -> Detection | None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected H×W RGB image")
    hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (3, 3), 0), cv2.COLOR_RGB2HSV)
    candidates = []
    for kind, (low, high) in HSV_RANGES.items():
        contours, _ = cv2.findContours(_mask(hsv, low, high),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 65:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            if not (0.02 * rgb.shape[1] < x < 0.98 * rgb.shape[1]
                    and 0.02 * rgb.shape[0] < y < 0.98 * rgb.shape[0]):
                continue
            shape, circularity = _shape(contour)
            expected = {"cube": "square", "cylinder": "circle", "prism": "triangle"}[kind]
            shape_score = 1.0 if shape == expected else 0.5
            uv = np.array([x, y, 1.0], dtype=float)
            mapped = homography @ uv
            if abs(mapped[2]) < 1e-10:
                continue
            world_xy = tuple((mapped[:2] / mapped[2]).tolist())
            confidence = min(1.0, area / 500.0) * shape_score
            candidates.append(Detection(kind, (float(x), float(y)), world_xy,
                                        float(area), float(confidence), shape, contour))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.area * item.confidence)


def fiducial_centers(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    contours, _ = cv2.findContours(_mask(hsv, (19, 80, 80), (36, 255, 255)),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 12:
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            centers.append((moments["m10"] / moments["m00"],
                            moments["m01"] / moments["m00"]))
    if len(centers) != 4:
        raise ValueError(f"Expected four visible calibration marks, found {len(centers)}")
    ordered = sorted(centers, key=lambda point: point[1])
    ordered = sorted(ordered[:2], key=lambda point: point[0]) + \
              sorted(ordered[2:], key=lambda point: point[0])
    return np.asarray(ordered, dtype=np.float32)
