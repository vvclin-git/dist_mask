from __future__ import annotations

import numpy as np

from .mask_ops import draw_brush_segment


ADD = 1
REMOVE = 0


def candidate_action_at(candidate_mask: np.ndarray, x: int, y: int) -> int:
    """Return REMOVE when the pressed pixel is already candidate, otherwise ADD."""
    h, w = candidate_mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return ADD
    return REMOVE if candidate_mask[y, x] > 0 else ADD


def apply_mask_stroke(
    mask: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    brush_size: int,
    value: int,
) -> None:
    draw_brush_segment(mask, p0, p1, brush_size, int(value))


def accept_candidate(gt_mask: np.ndarray, candidate_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union the candidate into GT and return a cleared candidate layer."""
    if gt_mask.shape != candidate_mask.shape:
        raise ValueError("Candidate and GT masks must have the same shape.")
    gt = np.maximum(gt_mask, candidate_mask).astype(np.uint8)
    candidate = np.zeros_like(candidate_mask, dtype=np.uint8)
    return gt, candidate
