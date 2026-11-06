from __future__ import annotations

import numpy as np

from .mask_ops import mask_boundary


GT_COLOR = np.array([0, 255, 255], dtype=np.uint8)
CANDIDATE_COLOR = np.array([255, 80, 180], dtype=np.uint8)
HIGHLIGHT_COLOR = np.array([255, 220, 0], dtype=np.uint8)


def _blend_pixels(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    idx = mask > 0
    if not np.any(idx):
        return
    src = image[idx].astype(np.float32)
    dst = color.astype(np.float32)
    image[idx] = np.clip(src * (1.0 - alpha) + dst * alpha, 0, 255).astype(np.uint8)


def render_raster_annotation(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    display_mode: str = "outline",
    candidate_visible: bool = True,
    gt_visible: bool = True,
    highlighted_mask: np.ndarray | None = None,
    preserve_mask_pixels: bool = False,
) -> np.ndarray:
    """Render low/medium zoom raster overlays.

    When preserve_mask_pixels is True, candidate and GT are not painted into the
    image. The UI can then draw cell borders separately at high zoom.
    """
    out = image_rgb.copy()

    if highlighted_mask is not None and np.any(highlighted_mask):
        _blend_pixels(out, highlighted_mask, HIGHLIGHT_COLOR, 0.32)

    if preserve_mask_pixels:
        return out

    if candidate_visible and candidate_mask is not None and np.any(candidate_mask):
        cand_boundary = mask_boundary(candidate_mask)
        out[cand_boundary > 0] = CANDIDATE_COLOR

    if not gt_visible or display_mode == "hidden":
        return out

    if display_mode in {"fill", "outline_fill"}:
        alpha = 0.30 if display_mode == "fill" else 0.12
        _blend_pixels(out, gt_mask, GT_COLOR, alpha)

    if display_mode in {"outline", "outline_fill"}:
        boundary = mask_boundary(gt_mask)
        out[boundary > 0] = GT_COLOR

    return out


def visible_mask_pixels(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Return visible foreground coordinates as N x 2 array in image (x, y) order."""
    x0, y0, x1, y1 = bounds
    crop = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(crop > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((xs + x0, ys + y0)).astype(np.int32)
