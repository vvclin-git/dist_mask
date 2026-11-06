from __future__ import annotations

import numpy as np

from .model import PaletteEntry


def generate_candidate_from_palette(cluster_map: np.ndarray, palette: list[PaletteEntry]) -> np.ndarray:
    """Generate one candidate mask from the currently selected color clusters."""
    keep_ids = [entry.cluster_id for entry in palette if entry.keep]
    if not keep_ids:
        return np.zeros(cluster_map.shape, dtype=np.uint8)
    return np.isin(cluster_map, np.asarray(keep_ids, dtype=np.int16)).astype(np.uint8)
