from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PaletteEntry:
    cluster_id: int
    rgb: tuple[int, int, int]
    count: int
    fraction: float
    keep: bool = True


@dataclass
class AnnotationState:
    image_path: Optional[Path] = None
    image_rgb: Optional[np.ndarray] = None
    gt_mask: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None
    candidate_mask: Optional[np.ndarray] = None
    cluster_map: Optional[np.ndarray] = None
    polygon_points: list[tuple[int, int]] = field(default_factory=list)
    palette: list[PaletteEntry] = field(default_factory=list)

    def reset_for_image(self, image_path: Path, image_rgb: np.ndarray) -> None:
        h, w = image_rgb.shape[:2]
        self.image_path = image_path
        self.image_rgb = image_rgb
        self.gt_mask = np.zeros((h, w), dtype=np.uint8)
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        self.candidate_mask = np.zeros((h, w), dtype=np.uint8)
        self.cluster_map = np.full((h, w), -1, dtype=np.int16)
        self.polygon_points = []
        self.palette = []

    @property
    def ready(self) -> bool:
        return self.image_rgb is not None and self.gt_mask is not None
