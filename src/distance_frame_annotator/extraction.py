"""Extract exact presentation-order frames into lossless RGB PNGs."""
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image


def read_sample(path, roi):
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != tuple(roi[2:]):
            raise ValueError(f'Invalid sample image: {path}')
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def extract_samples(video, roi, paths):
    missing = sorted((frame, path) for frame, path in paths.items() if not path.is_file())
    if not missing:
        return
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('找不到 FFmpeg，請將 ffmpeg.exe 所在資料夾加入 PATH')
    x, y, width, height = roi
    folder = missing[0][1].parent
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='extract-', dir=folder) as directory:
        stage = Path(directory)
        graph = stage / 'frames.txt'
        selection = '+'.join(f'eq(n,{frame})' for frame, _ in missing)
        graph.write_text(
            f"[0:v:0]select='{selection}',format=rgb24,"
            f"crop={width}:{height}:{x}:{y}:exact=1[out]",
            encoding='ascii',
        )
        result = subprocess.run(
            [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
             '-filter_complex_threads', '1', '-i', str(video),
             '-filter_complex_script', str(graph), '-map', '[out]',
             '-fps_mode', 'passthrough', '-frames:v', str(len(missing)),
             '-c:v', 'png', '-pix_fmt', 'rgb24', '-compression_level', '1',
             '-start_number', '0', str(stage / '%08d.png')],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if result.returncode:
            raise ValueError(f'FFmpeg 抽幀失敗：{result.stderr.strip()}')
        extracted = sorted(stage.glob('*.png'))
        if len(extracted) != len(missing):
            raise ValueError(f'FFmpeg 只取得 {len(extracted)}/{len(missing)} 張取樣影格')
        for image in extracted:
            with Image.open(image) as png:
                if png.mode != 'RGB' or png.size != (width, height):
                    raise ValueError('FFmpeg 輸出尺寸或色彩格式不符')
                png.verify()
        for image, (_, path) in zip(extracted, missing):
            image.replace(path)
