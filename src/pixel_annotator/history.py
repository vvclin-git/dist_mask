from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskSnapshot:
    shape: tuple[int, int]
    candidate_bits: np.ndarray
    gt_bits: np.ndarray

    @classmethod
    def from_masks(cls, candidate_mask: np.ndarray, gt_mask: np.ndarray) -> "MaskSnapshot":
        if candidate_mask.shape != gt_mask.shape:
            raise ValueError("Candidate and GT masks must have the same shape.")
        candidate = np.packbits((candidate_mask > 0).reshape(-1).astype(np.uint8))
        gt = np.packbits((gt_mask > 0).reshape(-1).astype(np.uint8))
        return cls(shape=candidate_mask.shape, candidate_bits=candidate, gt_bits=gt)

    def restore(self) -> tuple[np.ndarray, np.ndarray]:
        size = self.shape[0] * self.shape[1]
        candidate = np.unpackbits(self.candidate_bits, count=size).reshape(self.shape).astype(np.uint8)
        gt = np.unpackbits(self.gt_bits, count=size).reshape(self.shape).astype(np.uint8)
        return candidate, gt


class MaskHistory:
    """Undo/redo history for the two editable mask layers."""

    def __init__(self, max_steps: int = 40) -> None:
        self._undo: deque[MaskSnapshot] = deque(maxlen=max_steps)
        self._redo: deque[MaskSnapshot] = deque(maxlen=max_steps)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, candidate_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        self._undo.append(MaskSnapshot.from_masks(candidate_mask, gt_mask))
        self._redo.clear()

    def undo(self, candidate_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._undo:
            return None
        self._redo.append(MaskSnapshot.from_masks(candidate_mask, gt_mask))
        return self._undo.pop().restore()

    def redo(self, candidate_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._redo:
            return None
        self._undo.append(MaskSnapshot.from_masks(candidate_mask, gt_mask))
        return self._redo.pop().restore()
