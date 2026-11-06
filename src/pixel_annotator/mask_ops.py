from __future__ import annotations

import cv2
import numpy as np


def polygon_to_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(binary, eroded)


def keep_component_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
        return binary

    count, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    label = labels[y, x]
    if label == 0:
        return binary
    return (labels == label).astype(np.uint8)


def draw_brush_segment(
    mask: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    brush_size: int,
    value: int,
) -> None:
    radius = max(0, int(brush_size) // 2)
    cv2.line(mask, p0, p1, int(value), thickness=max(1, int(brush_size)), lineType=cv2.LINE_8)
    cv2.circle(mask, p1, radius, int(value), thickness=-1, lineType=cv2.LINE_8)
