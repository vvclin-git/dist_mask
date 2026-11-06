from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .model import PaletteEntry


@dataclass
class ColorAnalysisResult:
    cluster_map: np.ndarray
    palette: list[PaletteEntry]


def _rgb_to_lab(rgb_pixels: np.ndarray) -> np.ndarray:
    sample = rgb_pixels.reshape((-1, 1, 3)).astype(np.uint8)
    return cv2.cvtColor(sample, cv2.COLOR_RGB2LAB).reshape((-1, 3)).astype(np.float32)


def _lab_to_rgb(lab_center: np.ndarray) -> tuple[int, int, int]:
    center = np.clip(np.rint(lab_center), 0, 255).astype(np.uint8).reshape((1, 1, 3))
    rgb = cv2.cvtColor(center, cv2.COLOR_LAB2RGB)[0, 0]
    return tuple(int(v) for v in rgb)


def analyze_roi_colors(
    image_rgb: np.ndarray,
    roi_mask: np.ndarray,
    clusters: int = 8,
    max_fit_pixels: int = 30000,
    seed: int = 12345,
) -> ColorAnalysisResult:
    ys, xs = np.nonzero(roi_mask > 0)
    if len(xs) == 0:
        raise ValueError("ROI contains no pixels.")

    rgb_pixels = image_rgb[ys, xs]
    lab_pixels = _rgb_to_lab(rgb_pixels)
    k = max(1, min(int(clusters), len(lab_pixels)))

    rng = np.random.default_rng(seed)
    if len(lab_pixels) > max_fit_pixels:
        idx = rng.choice(len(lab_pixels), size=max_fit_pixels, replace=False)
        fit_pixels = lab_pixels[idx]
    else:
        fit_pixels = lab_pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.4)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(
        fit_pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    distances = np.sum((lab_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int16)

    cluster_map = np.full(roi_mask.shape, -1, dtype=np.int16)
    cluster_map[ys, xs] = labels

    counts = np.bincount(labels, minlength=k)
    total = int(counts.sum())
    palette = [
        PaletteEntry(
            cluster_id=i,
            rgb=_lab_to_rgb(centers[i]),
            count=int(counts[i]),
            fraction=float(counts[i] / total),
            keep=True,
        )
        for i in range(k)
    ]
    palette.sort(key=lambda p: p.count, reverse=True)
    return ColorAnalysisResult(cluster_map=cluster_map, palette=palette)
