import csv
import hashlib
import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path, PureWindowsPath
import cv2
import numpy as np


def load_rows(path, field, video=None):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        rows = list(reader)
    if {'video_filename', 'frame_number', 'distance_m'} <= columns:
        return load_track_lab_rows(rows, field, video)
    if not {'video', 'frame_index', 'distance_m', 'roi', 'status'} <= columns:
        raise ValueError('CSV 格式不符，請選擇 Track Lab 距離–影格表或 OCR 距離表')

    result = []
    for r in rows:
        if r.get('roi') != field or r.get('status') not in ('accepted', 'manual_accepted'):
            continue
        try:
            value = float(r['distance_m'])
            frame = int(r['frame_index'])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0 and frame >= 0:
            result.append({**r, 'distance_m': value, 'frame_index': frame})
    if not result:
        raise ValueError('No accepted distance rows for this field')
    unique = {(r['video'], r['frame_index']): r for r in result}
    return sorted(unique.values(), key=lambda r: (r['video'], r['frame_index']))


def load_track_lab_rows(rows, field, video):
    if not field.strip():
        raise ValueError('請在距離欄位填寫標註目標名稱')
    filename = PureWindowsPath(video).name.casefold() if video else None
    result = {}
    video_ids, targets = set(), set()
    for row in rows:
        name = (row.get('video_filename') or '').strip()
        if not name or (filename and PureWindowsPath(name).name.casefold() != filename):
            continue
        try:
            frame = int(row['frame_number'])
            distance = float(row['distance_m'])
        except (TypeError, ValueError):
            continue
        if frame < 0 or not math.isfinite(distance) or distance < 0:
            continue
        if row.get('video_id'):
            video_ids.add(row['video_id'])
        if row.get('mmsi'):
            targets.add(row['mmsi'])
        key = (name, frame)
        if key in result and result[key]['distance_m'] != distance:
            raise ValueError(f'Track Lab CSV 的影格 {frame} 有不同的距離值')
        result[key] = {**row, 'video': name, 'frame_index': frame,
                       'distance_m': distance, 'roi': field}

    if not result:
        raise ValueError('Track Lab CSV 中找不到所選影片的有效距離與影格配對')
    if len(video_ids) > 1 or len(targets) > 1:
        raise ValueError('請匯入單一影片與單一 AIS 目標的 Track Lab 距離表')
    return sorted(result.values(), key=lambda row: (row['video'], row['frame_index']))


def distance_workspace(rows, minimum='', maximum=''):
    try:
        lower = float(minimum) if minimum.strip() else None
        upper = float(maximum) if maximum.strip() else None
    except ValueError as exc:
        raise ValueError('距離工作區上下限須為數字') from exc
    if any(bound is not None and (not math.isfinite(bound) or bound < 0)
           for bound in (lower, upper)):
        raise ValueError('距離工作區上下限須為非負有限數值')
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('距離工作區下限不得大於上限')
    return [
        row for row in rows
        if (lower is None or row['distance_m'] >= lower)
        and (upper is None or row['distance_m'] <= upper)
    ]


def sample_rows(rows, width, count, gap, seed):
    if not math.isfinite(width) or width <= 0 or count < 1 or gap < 0:
        raise ValueError('Invalid sampling parameters')
    bins = defaultdict(list)
    for r in rows:
        bins[math.floor(r['distance_m']/width)*width].append(r)
    rng = random.Random(seed); chosen = []; by_video = defaultdict(list)
    for lower, candidates in sorted(bins.items()):
        candidates = list(candidates); rng.shuffle(candidates); found = 0
        for r in candidates:
            if all(abs(r['frame_index']-n) >= gap for n in by_video[r['video']]):
                chosen.append({**r, 'bin_lower_m': lower}); by_video[r['video']].append(r['frame_index']); found += 1
                if found >= count: break
    # Sampling is intentionally random within each bin, but the public result
    # is deterministic and easy to review: distance first, then frame index.
    return sorted(chosen, key=lambda r: (r['distance_m'], r['frame_index'], r['video']))


def validate_roi(roi, frame):
    x,y,w,h = roi
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('ROI outside image')
    return frame[y:y+h, x:x+w].copy()


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path)); self.cache = OrderedDict()
        if not self.cap.isOpened(): raise ValueError(f'Cannot open video: {path}')
    def get(self, index):
        if index < 0: raise ValueError('Frame index cannot be negative')
        if index in self.cache:
            self.cache.move_to_end(index); return self.cache[index]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = self.cap.read()
        if not ok: raise ValueError(f'Cannot decode frame {index}')
        actual = round(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        if actual != index: raise ValueError(f'Decoder returned frame {actual}, expected {index}')
        self.cache[index] = frame
        while len(self.cache) > 16: self.cache.popitem(last=False)
        return frame
    def close(self): self.cap.release()


def propagate(previous, current, polygons, radius=40, min_score=.75):
    if not polygons: raise ValueError('No confirmed polygon to propagate')
    all_points = np.array([p for poly in polygons for p in poly], dtype=np.int32)
    x,y,w,h = cv2.boundingRect(all_points); height,width = previous.shape[:2]
    x0,y0 = max(0,x-4),max(0,y-4); x1,y1 = min(width,x+w+4),min(height,y+h+4)
    patch = cv2.cvtColor(previous[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    if patch.std() < 1: raise ValueError('Template has too little texture')
    sx,sy = max(0,x0-radius),max(0,y0-radius); ex,ey = min(width,x1+radius),min(height,y1+radius)
    search = cv2.cvtColor(current[sy:ey,sx:ex],cv2.COLOR_BGR2GRAY)
    response = cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
    _,score,_,loc = cv2.minMaxLoc(response)
    if not math.isfinite(score) or score < min_score: return [],score
    dx,dy = sx+loc[0]-x0,sy+loc[1]-y0
    shifted = [[[px+dx,py+dy] for px,py in poly] for poly in polygons]
    if any(not (0 <= px < width and 0 <= py < height) for poly in shifted for px,py in poly):
        return [],score
    return shifted,score


def annotation_data(row, video, roi, polygons, crop, negative=False):
    mask = np.zeros(crop.shape[:2],np.uint8); objects=[]; x,y,w,h=roi
    if crop.shape[:2] != (h,w): raise ValueError('Crop dimensions differ from ROI')
    for poly in polygons:
        if len(poly)<3: raise ValueError('Polygon requires at least three vertices')
        pts=np.array(poly,dtype=np.int32)
        if np.any(pts<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]>=h): raise ValueError('Polygon outside ROI')
        single=np.zeros_like(mask); cv2.fillPoly(single,[pts],255); mask=np.maximum(mask,single)
        bx,by,bw,bh=cv2.boundingRect(pts)
        objects.append({'polygon_roi':pts.tolist(),'polygon_original':(pts+[x,y]).tolist(),'bbox_roi':[bx,by,bw,bh], 'bbox_original':[bx+x,by+y,bw,bh],'area_pixels':int(np.count_nonzero(single))})
    if not objects and not negative: raise ValueError('Draw a polygon or confirm no object')
    record={'schema_version':1,'video':str(video),'frame_index':int(row['frame_index']), 'timestamp_s':row.get('timestamp_s',''),'distance_field':row['roi'],'distance_m':row['distance_m'],'bin_lower_m':row.get('bin_lower_m'), 'crop_roi':list(roi),'objects':objects,'union_area_pixels':int(np.count_nonzero(mask)), 'negative':not bool(objects),'status':'human_confirmed'}
    return record,mask


def key_for(video, frame, roi):
    digest=hashlib.sha256((str(video)+str(list(roi))).encode()).hexdigest()[:10]
    return f'{Path(video).stem}_{digest}_f{frame:07d}'


def save_annotation(out, record, crop, mask):
    out=Path(out); key=key_for(record['video'],record['frame_index'],record['crop_roi'])
    for folder in ['images','masks','annotated','annotations']: (out/folder).mkdir(parents=True,exist_ok=True)
    overlay=crop.copy(); active=mask>0; overlay[active]=(overlay[active]*.5+np.array([0,255,0])*.5).astype(np.uint8)
    for folder,image in [('images',crop),('masks',mask),('annotated',overlay)]:
        cv2.imencode('.png',image)[1].tofile(str(out/folder/(key+'.png')))
    p=out/'annotations'/(key+'.json'); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
    return key
