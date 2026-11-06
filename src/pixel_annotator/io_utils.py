from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .model import AnnotationState


def load_rgb_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _relative_path_or_none(path: Path, base_dir: Path) -> str | None:
    """Return a portable relative path when possible.

    On Windows, os.path.relpath raises ValueError when path and base_dir are on
    different drives. In that case the absolute/original path is still retained
    in metadata and relative_path is stored as null.
    """
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return None


def build_annotation_metadata(
    state: AnnotationState,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    class_id: int = 1,
    class_name: str = "target",
    annotation_status: str = "completed",
) -> dict:
    """Build the native v3 annotation metadata without writing files.

    This function intentionally has no Tkinter dependency so it can be reused
    by video/frame-sampling tools and unit tests.
    """
    if not state.ready or state.image_path is None or state.gt_mask is None or state.image_rgb is None:
        raise ValueError("No image is loaded.")

    source_path = Path(state.image_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    height, width = state.image_rgb.shape[:2]

    return {
        "format_version": 3,
        "source_image": {
            "filename": source_path.name,
            "relative_path": _relative_path_or_none(source_path, output_dir),
            "original_path": str(source_path),
        },
        "image_size": {
            "width": int(width),
            "height": int(height),
        },
        "mask": {
            "file": mask_path.name,
            "dtype": "uint8",
            "background": 0,
            "foreground": 255,
        },
        "class": {
            "id": int(class_id),
            "name": str(class_name),
        },
        "annotation": {
            "status": str(annotation_status),
            "gt_pixel_count": int((state.gt_mask > 0).sum()),
        },
        "assist": {
            "polygon_roi": {
                "points": [[int(x), int(y)] for x, y in state.polygon_points],
            },
            "color_analysis": {
                "color_space": "Lab",
                "method": "kmeans",
                "clusters": int(len(state.palette)),
                "palette": [
                    {
                        "cluster_id": int(entry.cluster_id),
                        "rgb": [int(v) for v in entry.rgb],
                        "count": int(entry.count),
                        "fraction": float(entry.fraction),
                        "keep": bool(entry.keep),
                    }
                    for entry in state.palette
                ],
            },
        },
        "created_with": {
            "tool": "pixel-mask-annotator",
            "version": __version__,
        },
    }


def save_annotation(
    state: AnnotationState,
    output_dir: str | Path,
    *,
    class_id: int = 1,
    class_name: str = "target",
    annotation_status: str = "completed",
) -> tuple[Path, Path]:
    """Save the GT binary mask plus native v3 JSON metadata.

    The master mask remains an 8-bit lossless grayscale PNG using 0 for
    background and 255 for foreground. Metadata is deliberately separate from
    the mask so future exporters can target COCO, YOLO, class-ID PNG, etc.
    """
    if not state.ready or state.image_path is None or state.gt_mask is None:
        raise ValueError("No image is loaded.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = state.image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    meta_path = output_dir / f"{stem}_annotation.json"

    Image.fromarray((state.gt_mask > 0).astype(np.uint8) * 255, mode="L").save(mask_path)

    metadata = build_annotation_metadata(
        state,
        mask_path,
        output_dir,
        class_id=class_id,
        class_name=class_name,
        annotation_status=annotation_status,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return mask_path, meta_path
