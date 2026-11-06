"""Raster annotation persistence and translation proposals."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def source_id(path):
    path = Path(path)
    digest = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open('rb') as stream:
        digest.update(stream.read(65536))
        stream.seek(max(0, path.stat().st_size - 65536))
        digest.update(stream.read(65536))
    return digest.hexdigest()[:24]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def binary(mask, shape):
    if mask.ndim != 2 or mask.shape != tuple(shape):
        raise ValueError('Mask dimensions differ from crop ROI')
    return (mask > 0).astype(np.uint8)


class AnnotationStore:
    def __init__(self, folder, identity, video, field='buoy'):
        self.folder = Path(folder).resolve()
        self.identity = identity
        self.video = Path(video).resolve()
        self.field = field
        self.index_path = self.folder / 'annotation_index.json'
        self.index = json.loads(self.index_path.read_text(encoding='utf-8')) if self.index_path.exists() else {}
        self.legacy = None

    def key(self, frame, roi):
        class_tag = hashlib.sha256(self.field.encode()).hexdigest()[:8]
        return f'{self.identity}_c{class_tag}_f{frame:07d}_' + '_'.join(map(str, roi))

    def load(self, frame, roi):
        key = self.key(frame, roi)
        if key in self.index:
            record_path = self.folder / self.index[key]
            record = json.loads(record_path.read_text(encoding='utf-8'))
            mask_path = record_path.parent / record['mask']['file']
            with Image.open(mask_path) as image:
                mask = binary(np.asarray(image), (roi[3], roi[2]))
            if int(np.count_nonzero(mask)) != record['annotation']['gt_pixel_count']:
                raise ValueError('Mask and metadata pixel counts disagree')
            if record['frame_index'] != frame or record['crop_roi'] != list(roi) or record['source_id'] != self.identity or record['distance_field'] != self.field:
                raise ValueError('Annotation provenance does not match this frame')
            return mask, record
        if self.legacy is None:
            self.legacy = {}
            for path in (self.folder / 'annotations').glob('*.json'):
                data = json.loads(path.read_text(encoding='utf-8'))
                if Path(data['video'].replace('\\', '/')).name == self.video.name and data.get('distance_field', self.field) == self.field:
                    self.legacy[(data['frame_index'], tuple(data['crop_roi']))] = (path, data)
        old = self.legacy.get((frame, tuple(roi)))
        if old is None:
            return np.zeros((roi[3], roi[2]), np.uint8), None
        path, record = old
        png = self.folder / 'masks' / (path.stem + '.png')
        if png.exists():
            with Image.open(png) as image:
                mask = binary(np.asarray(image), (roi[3], roi[2]))
        else:
            mask = np.zeros((roi[3], roi[2]), np.uint8)
            for obj in record['objects']:
                points = np.asarray(obj['polygon_roi'], dtype=np.int32)
                if np.any(points < 0) or np.any(points[:, 0] >= roi[2]) or np.any(points[:, 1] >= roi[3]):
                    raise ValueError('Legacy polygon outside ROI')
                cv2.fillPoly(mask, [points], 1)
        if int(np.count_nonzero(mask)) != record['union_area_pixels']:
            raise ValueError('Legacy mask and metadata pixel counts disagree')
        return mask, record

    def save(self, row, roi, rgb, state, metadata, negative=False):
        mask = binary(state.gt_mask, rgb.shape[:2])
        area = int(np.count_nonzero(mask))
        if negative != (area == 0):
            raise ValueError('Use explicit no-object confirmation for an empty GT')
        key = self.key(row['frame_index'], roi)
        # Commit the index last; a failed write cannot replace the last good record.
        folder = self.folder / 'records' / key / uuid.uuid4().hex
        folder.mkdir(parents=True)
        Image.fromarray(rgb).save(folder / 'image.png')
        Image.fromarray(mask * 255).save(folder / 'mask.png')
        preview = rgb.copy()
        preview[mask > 0] = (preview[mask > 0] * .5 + np.array([0, 255, 255]) * .5).astype(np.uint8)
        Image.fromarray(preview).save(folder / 'annotated.png')
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)] if area else None
        record = {**metadata, 'schema_version': 2, 'source_id': self.identity,
                  'supersedes': self.index.get(key),
                  'video': str(self.video), 'frame_index': row['frame_index'],
                  'timestamp_s': row.get('timestamp_s', ''), 'distance_field': row['roi'],
                  'distance_m': row['distance_m'], 'bin_lower_m': row['bin_lower_m'],
                  'bin_width_m': row['bin_width_m'], 'crop_roi': list(roi),
                  'image_size': {'width': roi[2], 'height': roi[3]},
                  'source_image': {'relative_path': 'image.png'},
                  'mask': {'file': 'mask.png', 'dtype': 'uint8', 'background': 0, 'foreground': 255},
                  'annotation': {'status': 'human_confirmed', 'gt_pixel_count': area},
                  'union_area_pixels': area, 'negative': negative, 'bbox_roi': bbox,
                  'bbox_original': [bbox[0]+roi[0], bbox[1]+roi[1], bbox[2], bbox[3]] if bbox else None}
        atomic_json(folder / 'annotation.json', record)
        updated = {**self.index, key: str((folder / 'annotation.json').relative_to(self.folder))}
        atomic_json(self.index_path, updated)
        self.index = updated
        return record

    def export(self):
        records = [{**json.loads((self.folder / p).read_text(encoding='utf-8')), 'annotation_file': p} for p in self.index.values()]
        modern = {(Path(r['video'].replace('\\', '/')).name, r['frame_index'], tuple(r['crop_roi']), r['distance_field']) for r in records}
        for path in (self.folder / 'annotations').glob('*.json'):
            old = json.loads(path.read_text(encoding='utf-8'))
            if (Path(old['video'].replace('\\', '/')).name, old['frame_index'], tuple(old['crop_roi']), old.get('distance_field', 'buoy')) not in modern:
                records.append({**old, 'annotation_file': str(path.relative_to(self.folder))})
        records.sort(key=lambda r: (r['distance_m'] is None, r['distance_m'] or 0, r['frame_index']))
        atomic_json(self.folder / 'annotations.json', records)
        return len(records)


def translate_proposal(previous_rgb, current_rgb, gt, radius, threshold):
    if previous_rgb.shape != current_rgb.shape:
        raise ValueError('Reference and current ROI dimensions differ')
    gt = binary(gt, current_rgb.shape[:2])
    ys, xs = np.nonzero(gt)
    if not len(xs):
        raise ValueError('Reference GT is empty')
    if radius < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid matching parameters')
    h, w = gt.shape
    x0, y0 = max(0, int(xs.min())-4), max(0, int(ys.min())-4)
    x1, y1 = min(w, int(xs.max())+5), min(h, int(ys.max())+5)
    patch = cv2.cvtColor(previous_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if patch.std() < 1:
        raise ValueError('Reference template has too little texture')
    sx, sy = max(0, x0-radius), max(0, y0-radius)
    ex, ey = min(w, x1+radius), min(h, y1+radius)
    search = cv2.cvtColor(current_rgb[sy:ey, sx:ex], cv2.COLOR_RGB2GRAY)
    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
    if not np.isfinite(score) or score < threshold:
        raise ValueError(f'Matching below threshold: {score:.3f}')
    dx, dy = sx+loc[0]-x0, sy+loc[1]-y0
    if np.any(xs+dx < 0) or np.any(xs+dx >= w) or np.any(ys+dy < 0) or np.any(ys+dy >= h):
        raise ValueError('Translated mask would leave crop ROI; proposal rejected')
    result = np.zeros_like(gt)
    result[ys+dy, xs+dx] = 1
    return result, {'method': 'template_matching', 'score': float(score), 'dx': dx, 'dy': dy}
