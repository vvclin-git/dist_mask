"""Video sampling host for the pixel mask editor."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image
from pixel_annotator.app import AnnotationApp
from pixel_annotator.io_utils import build_annotation_metadata
from .core import load_rows, sample_rows, VideoReader, validate_roi
from .storage import AnnotationStore, atomic_json, source_id, translate_proposal


class Editor(AnnotationApp):
    def __init__(self, owner, host):
        self.owner = owner
        self._candidate_drag = None
        super().__init__(owner.root, host)

    def _bind_events(self):
        self.canvas.bind('<Configure>', lambda e: self.refresh_canvas())
        for event, callback in [('<ButtonPress-1>', self._on_left_press),
                                ('<B1-Motion>', self._on_left_drag),
                                ('<ButtonRelease-1>', self._on_left_release),
                                ('<Motion>', self._on_motion),
                                ('<MouseWheel>', self._on_mousewheel)]:
            self.canvas.bind(event, callback)
        for button in (2, 3):
            self.canvas.bind(f'<ButtonPress-{button}>', self._on_pan_press)
            self.canvas.bind(f'<B{button}-Motion>', self._on_pan_drag)
            self.canvas.bind(f'<ButtonRelease-{button}>', self._on_pan_release)
        self.canvas.bind('<Leave>', lambda e: self._hide_edit_cursor())
        self.canvas.bind('<Button-4>', lambda e: self._zoom_at(e.x, e.y, 1.25))
        self.canvas.bind('<Button-5>', lambda e: self._zoom_at(e.x, e.y, .8))

    def open_image(self):
        self.owner.status.set('請使用上方來源影片與距離 CSV 載入影像。')

    def save_current(self):
        self.owner.guard(self.owner.save)

    def _on_left_press(self, event):
        if self.owner.mode.get() == 'reference' and not (self._space_down or self.tool.get() == 'pan'):
            return
        if self.tool.get() == 'candidate_move' and not self._space_down:
            if not self.state.ready:
                return
            x, y = self.canvas_to_image(event.x, event.y, clamp=False)
            mask = self.state.candidate_mask
            h, w = mask.shape
            if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
                self.status_text.set('請在 Candidate 像素內按住左鍵拖曳；右鍵仍平移檢視區。')
                return
            ys, xs = np.nonzero(mask)
            self.canvas.focus_set()
            self._candidate_drag = {'start': (x, y), 'mask': mask.copy(), 'history': False,
                                    'bounds': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
            return
        super()._on_left_press(event)

    def _on_left_drag(self, event):
        drag = getattr(self, '_candidate_drag', None)
        if drag is not None and not self._pan_active and self.owner.mode.get() != 'reference':
            x, y = self.canvas_to_image(event.x, event.y, clamp=False)
            h, w = drag['mask'].shape
            left, top, right, bottom = drag['bounds']
            dx = min(max(x-drag['start'][0], -left), w-1-right)
            dy = min(max(y-drag['start'][1], -top), h-1-bottom)
            if not drag['history'] and (dx or dy):
                self.history.push(self.state.candidate_mask, self.state.gt_mask)
                drag['history'] = True
            # Always translate the initial mask, avoiding cumulative resampling.
            shifted = np.zeros_like(drag['mask'])
            ys, xs = np.nonzero(drag['mask'])
            shifted[ys+dy, xs+dx] = 1
            self.state.candidate_mask = shifted
            self.status_text.set(f'Candidate 移動 dx={dx}, dy={dy} px；未改變 GT。')
            self.refresh_canvas()
            return
        if self.owner.mode.get() != 'reference' or self._pan_active:
            super()._on_left_drag(event)

    def _on_left_release(self, event):
        self._candidate_drag = None
        super()._on_left_release(event)

    def _on_pan_press(self, event):
        self._candidate_drag = None
        super()._on_pan_press(event)

    def _update_tool_hint(self):
        if self.tool.get() == 'candidate_move' and hasattr(self, 'tool_hint'):
            self.tool_hint.configure(text='Move Candidate: left drag / M')
        else:
            super()._update_tool_hint()

    def _set_tool(self, tool):
        self._candidate_drag = None
        super()._set_tool(tool)

    def refresh_canvas(self):
        if not self.state.ready:
            return super().refresh_canvas()
        ref = self.owner.reference
        mode = self.owner.mode.get()
        original = (self.state.image_rgb, self.state.gt_mask, self.state.candidate_mask,
                    self.state.polygon_points, self.state.roi_mask, self.state.cluster_map)
        try:
            if ref is not None and ref['roi'] == self.owner.active['roi']:
                if mode == 'reference':
                    self.state.image_rgb = ref['rgb']
                    self.state.gt_mask = ref['mask']
                    self.state.candidate_mask = np.zeros_like(ref['mask'])
                    self.state.polygon_points = []
                    self.state.roi_mask = np.zeros_like(ref['mask'])
                    self.state.cluster_map = None
                elif mode == 'overlay':
                    alpha = self.owner.alpha.get()
                    self.state.image_rgb = cv2.addWeighted(original[0], 1-alpha, ref['rgb'], alpha, 0)
            super().refresh_canvas()
        finally:
            (self.state.image_rgb, self.state.gt_mask, self.state.candidate_mask,
             self.state.polygon_points, self.state.roi_mask, self.state.cluster_map) = original


def readonly_guard(method):
    def wrapped(self, *args, **kwargs):
        if self.owner.mode.get() == 'reference':
            self.status_text.set('參考模式僅供檢視；請切至 current 或 overlay 編輯。')
            return
        self._candidate_drag = None
        return method(self, *args, **kwargs)
    return wrapped


for _name in ('close_polygon', 'clear_polygon', 'analyze_colors', 'generate_candidate',
              'clear_candidate', 'accept_candidate_to_gt', 'clear_gt', 'undo', 'redo',
              '_toggle_cluster_selection'):
    setattr(Editor, _name, readonly_guard(getattr(AnnotationApp, _name)))


class App:
    def __init__(self, root):
        self.root = root
        root.title('Distance Frame Annotator v2.1 — Pixel GT')
        root.geometry('1550x1000')
        self.reader = None
        self.current = None
        self.active = None
        self.reference = None
        self.last_saved = None
        self.samples = []
        self.cursor = 0
        self.selection_guard = False
        self.baseline = None
        self.proposal = None
        self.mode = tk.StringVar(value='current')
        self.alpha = tk.DoubleVar(value=.5)
        self.status = tk.StringVar(value='請載入影片與距離 CSV')
        defaults = {'csv': '', 'video': '', 'out': 'annotations_output', 'field': 'buoy',
                    'width': '10', 'count': '3', 'gap': '30', 'seed': '42',
                    'roi': '0,0,960,711', 'step': '1', 'frame': '0', 'radius': '40', 'score': '.75'}
        self.v = {k: tk.StringVar(value=v) for k, v in defaults.items()}
        top = ttk.Frame(root)
        top.pack(fill='x')
        for i, (key, label) in enumerate([('csv', '距離 CSV'), ('video', '影片'), ('out', '輸出資料夾')]):
            ttk.Label(top, text=label).grid(row=i, column=0)
            ttk.Entry(top, textvariable=self.v[key], width=100).grid(row=i, column=1, sticky='ew')
            ttk.Button(top, text='選擇', command=lambda k=key: self.pick(k)).grid(row=i, column=2)
        top.columnconfigure(1, weight=1)
        parameters = ttk.Frame(root)
        parameters.pack(fill='x')
        for key, label in [('field', '距離欄位'), ('width', 'bin(m)'), ('count', '每 bin 數'),
                           ('gap', '最小間隔'), ('seed', 'seed'), ('roi', '裁切 ROI x,y,w,h')]:
            ttk.Label(parameters, text=label).pack(side='left')
            ttk.Entry(parameters, textvariable=self.v[key], width=22 if key == 'roi' else 7).pack(side='left')
        buttons = ttk.Frame(root)
        buttons.pack(fill='x')
        for label, callback in [('載入／套用並取樣', self.sample), ('手動畫裁切 ROI', self.pick_roi),
                                ('保存工作階段', self.save_session), ('開啟工作階段', self.open_session),
                                ('匯出取樣影像', self.export_samples), ('匯出標註索引', self.export_index)]:
            ttk.Button(buttons, text=label, command=lambda c=callback: self.guard(c)).pack(side='left')
        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True)
        left, host = ttk.Frame(body), ttk.Frame(body)
        body.add(left, weight=1)
        body.add(host, weight=6)
        self.list = ttk.Treeview(left, columns=('frame', 'distance', 'bin'), show='headings', selectmode='browse')
        for column in ('frame', 'distance', 'bin'):
            self.list.heading(column, text=column)
            self.list.column(column, width=80)
        self.list.pack(fill='both', expand=True)
        self.list.bind('<<TreeviewSelect>>', self.choose)
        self.editor = Editor(self, host)
        nav = ttk.Frame(root)
        nav.pack(fill='x')
        for label, callback in [('◀ 鄰幀', lambda: self.move(-1)), ('鄰幀 ▶', lambda: self.move(1)),
                                ('前取樣 PgUp', lambda: self.next_sample(-1)), ('後取樣 PgDn', lambda: self.next_sample(1))]:
            ttk.Button(nav, text=label, command=lambda c=callback: self.guard(c)).pack(side='left')
        for key, label in [('step', '步長'), ('frame', 'frame')]:
            ttk.Label(nav, text=label).pack(side='left')
            ttk.Entry(nav, textvariable=self.v[key], width=7).pack(side='left')
        ttk.Button(nav, text='跳轉', command=lambda: self.guard(lambda: self.show(int(self.v['frame'].get())))).pack(side='left')
        ttk.Button(nav, text='設目前確認 GT 為參考', command=lambda: self.guard(self.set_reference)).pack(side='left')
        ttk.Combobox(nav, textvariable=self.mode, values=['current', 'reference', 'overlay'], state='readonly', width=10).pack(side='left')
        ttk.Scale(nav, from_=0, to=1, variable=self.alpha, command=lambda v: self.editor.refresh_canvas()).pack(side='left')
        self.mode.trace_add('write', lambda *args: (setattr(self.editor, '_candidate_drag', None), self.editor.refresh_canvas()))
        actions = ttk.Frame(root)
        actions.pack(fill='x')
        for label, callback in [('保存 GT Ctrl+S', self.save), ('確認無物件', lambda: self.save(True)),
                                ('帶入最近保存 GT', self.import_last_gt),
                                ('移動 Candidate (M)', lambda: self.editor._set_tool('candidate_move')),
                                ('傳播參考 mask 至 Candidate', self.predict)]:
            ttk.Button(actions, text=label, command=lambda c=callback: self.guard(c)).pack(side='left')
        for key, label in [('radius', '搜尋半徑'), ('score', '最低分數')]:
            ttk.Label(actions, text=label).pack(side='left')
            ttk.Entry(actions, textvariable=self.v[key], width=6).pack(side='left')
        ttk.Label(root, textvariable=self.status).pack(fill='x')
        bindings = {'<Left>': lambda: self.move(-1), '<Right>': lambda: self.move(1),
                    '<Prior>': lambda: self.next_sample(-1), '<Next>': lambda: self.next_sample(1),
                    '<Return>': self.editor.close_polygon, '<Control-s>': self.save,
                    '<Control-z>': self.editor.undo, '<Control-y>': self.editor.redo,
                    '<Escape>': self.editor.clear_polygon,
                    'p': lambda: self.editor._set_tool('polygon'),
                    'a': lambda: self.editor._set_tool('candidate_edit'),
                    'm': lambda: self.editor._set_tool('candidate_move'),
                    'c': lambda: self.editor._set_tool('component'),
                    'b': lambda: self.editor._set_tool('gt_brush'),
                    'e': lambda: self.editor._set_tool('gt_eraser')}
        for event, callback in bindings.items():
            root.bind(event, lambda e, c=callback: self.shortcut(c))
        root.bind('<KeyPress-space>', self.editor._on_space_down)
        root.bind('<KeyRelease-space>', self.editor._on_space_up)
        root.protocol('WM_DELETE_WINDOW', lambda: self.guard(self.close))

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            messagebox.showerror('操作失敗', str(exc))
            self.status.set(str(exc))

    def shortcut(self, callback):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox, tk.Text)):
            return
        self.editor._candidate_drag = None
        self.guard(callback)
        return 'break'

    def pick(self, key):
        path = filedialog.askdirectory() if key == 'out' else filedialog.askopenfilename(
            filetypes=[('CSV', '*.csv'), ('All', '*')] if key == 'csv' else [('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*')])
        if path:
            self.v[key].set(path)

    def roi(self):
        values = [int(x.strip()) for x in self.v['roi'].get().split(',')]
        if len(values) != 4:
            raise ValueError('ROI requires x,y,width,height')
        return values

    def fingerprint(self):
        if not self.editor.state.ready:
            return None
        state = self.editor.state
        digest = hashlib.sha256(state.gt_mask.tobytes() + state.candidate_mask.tobytes())
        digest.update(json.dumps([state.polygon_points, [(p.cluster_id, p.keep) for p in state.palette]]).encode())
        return digest.digest()

    def leave(self):
        if self.current is None or self.fingerprint() == self.baseline:
            return True
        result = messagebox.askyesnocancel('未保存修改', '是否保存目前 GT？\n是：保存後切換；否：捨棄；取消：留在目前 frame。\nCandidate 必須先接受或清除。')
        if result is None:
            return False
        if result:
            self.save()
        return True

    def settings(self):
        return {key: value.get() for key, value in self.v.items()}

    def load_source(self, settings, samples=None, frame=None, identity=None, cursor=0):
        if not self.leave():
            return False
        rows = load_rows(settings['csv'], settings['field'], settings['video'])
        if len({r['video'] for r in rows}) != 1:
            raise ValueError('一次僅支援一支影片的 CSV')
        roi = [int(x.strip()) for x in settings['roi'].split(',')]
        if len(roi) != 4:
            raise ValueError('ROI requires four values')
        width = float(settings['width'])
        if not math.isfinite(width) or width <= 0:
            raise ValueError('Invalid bin width')
        identity_now = source_id(settings['video'])
        if identity and identity != identity_now:
            raise ValueError('所選影片與 session 來源不符')
        selected = samples if samples is not None else sample_rows(rows, width, int(settings['count']), int(settings['gap']), int(settings['seed']))
        selected = sorted(selected, key=lambda r: (float(r['distance_m']), int(r['frame_index'])))
        if not selected:
            raise ValueError('沒有可用取樣')
        first = selected[0]['frame_index'] if frame is None else int(frame)
        reader = VideoReader(settings['video'])
        try:
            crop = validate_roi(roi, reader.get(first))
            store = AnnotationStore(settings['out'], identity_now, settings['video'], settings['field'])
            total = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if any(int(r['frame_index']) < 0 or (total > 0 and int(r['frame_index']) >= total) for r in selected):
                raise ValueError('取樣 frame 超過影片範圍；請確認 CSV 對應影片')
            mask, record = store.load(first, roi)
        except Exception:
            reader.close()
            raise
        same = self.active and self.active['identity'] == identity_now and self.active['roi'] == roi and self.active['settings']['field'] == settings['field']
        if not same:
            self.last_saved = None
        if self.reader:
            self.reader.close()
        self.reader, self.store = reader, store
        self.active = {'settings': settings.copy(), 'identity': identity_now, 'roi': roi, 'width': width}
        self.rows, self.lookup = rows, {r['frame_index']: r for r in rows}
        self.samples, self.cursor = selected, max(0, min(cursor, len(selected)-1))
        self.reference = None
        for key, value in settings.items():
            if key in self.v:
                self.v[key].set(value)
        self.list.delete(*self.list.get_children())
        for i, row in enumerate(selected):
            self.list.insert('', 'end', iid=str(i), values=(row['frame_index'], row['distance_m'], row['bin_lower_m']))
        self.install(first, crop, mask, record, fit=not same)
        self.sync_selection()
        return True

    def sample(self):
        if self.load_source(self.settings()):
            self.save_session()

    def ensure_applied(self):
        if self.active is None:
            raise ValueError('請先載入影片')
        for key in ('csv', 'video', 'out', 'field', 'width', 'count', 'gap', 'seed', 'roi'):
            if self.v[key].get() != self.active['settings'][key]:
                raise ValueError('設定已變更，請先「載入／套用並取樣」；還原設定可先保存舊標註。')

    def install(self, frame, crop, mask, record, fit=False):
        editor = self.editor
        editor._image_generation += 1
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        editor.state.reset_for_image(Path(self.active['settings']['video']).with_name(f'frame_{frame:07d}.png'), rgb)
        editor.state.gt_mask = mask
        editor.history.clear()
        editor._clear_palette_ui()
        editor._highlight_cluster = None
        editor._stroke_active = editor._pan_active = editor._space_down = False
        editor._candidate_drag = None
        editor._last_draw_point = editor._stroke_value = editor._pan_last = None
        editor.canvas.configure(cursor='crosshair')
        self.current, self.record, self.proposal = frame, record, None
        self.v['frame'].set(str(frame))
        self.mode.set('current')
        if fit:
            self.root.update_idletasks()
            editor._fit_image_to_canvas()
        self.baseline = self.fingerprint()
        editor.refresh_canvas()
        editor.status_text.set('分析 ROI → 色彩 → Candidate → 加入 GT；Enter 完成 ROI，Ctrl+S 保存。')
        self.update_status()

    def row(self):
        original = self.lookup.get(self.current)
        distance = original['distance_m'] if original else None
        return {'frame_index': self.current, 'roi': self.active['settings']['field'],
                'distance_m': distance, 'timestamp_s': original.get('timestamp_s', '') if original else '',
                'bin_lower_m': math.floor(distance/self.active['width'])*self.active['width'] if distance is not None else None,
                'bin_width_m': self.active['width']}

    def update_status(self):
        state = '未確認' if self.record is None else ('已確認無物件' if self.record.get('negative') else '已確認 GT')
        reference = self.reference['frame'] if self.reference else '無'
        recent = self.last_saved['frame'] if self.last_saved else '無'
        self.status.set(f'取樣 {self.cursor+1}/{len(self.samples)}（定位 {self.samples[self.cursor]["frame_index"]}） | 目前 frame {self.current} | 距離 {self.row()["distance_m"]} m | {state} | 參考 {reference} | 最近保存 {recent}')

    def show(self, frame):
        self.ensure_applied()
        if frame == self.current:
            return True
        total = int(self.reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame < 0 or (total > 0 and frame >= total):
            raise ValueError('Frame outside video range')
        if not self.leave():
            return False
        crop = validate_roi(self.active['roi'], self.reader.get(frame))
        mask, record = self.store.load(frame, self.active['roi'])
        self.install(frame, crop, mask, record)
        return True

    def sync_selection(self):
        self.selection_guard = True
        self.list.selection_set(str(self.cursor))
        self.list.see(str(self.cursor))
        self.root.after_idle(lambda: setattr(self, 'selection_guard', False))

    def choose(self, event=None):
        ids = self.list.selection()
        if self.selection_guard or not ids:
            return
        target = int(ids[0])
        if target == self.cursor and self.samples[target]['frame_index'] == self.current:
            return
        try:
            if self.show(self.samples[target]['frame_index']):
                self.cursor = target
                self.update_status()
        except Exception as exc:
            messagebox.showerror('切換失敗', str(exc))
        self.sync_selection()

    def next_sample(self, direction):
        self.ensure_applied()
        target = max(0, min(len(self.samples)-1, self.cursor+direction))
        if self.show(self.samples[target]['frame_index']):
            self.cursor = target
            self.sync_selection()
            self.update_status()

    def move(self, direction):
        step = int(self.v['step'].get())
        if step < 1:
            raise ValueError('步長須大於零')
        self.show(self.current + direction*step if self.current is not None else 0)

    def pick_roi(self):
        self.ensure_applied()
        if not self.leave():
            return
        roi = cv2.selectROI('Crop ROI - Enter', self.reader.get(self.current), False, False)
        cv2.destroyWindow('Crop ROI - Enter')
        if roi[2] and roi[3]:
            settings = self.settings()
            settings['roi'] = ','.join(map(str, roi))
            if self.load_source(settings, self.samples, self.current, cursor=self.cursor):
                self.save_session()

    def save(self, negative=False):
        if self.active is None:
            raise ValueError('請先載入影片')
        state = self.editor.state
        if np.any(state.candidate_mask):
            raise ValueError('Candidate 尚未接受；請先加入 GT 或清除 Candidate，再保存。')
        metadata = build_annotation_metadata(state, 'mask.png', self.store.folder, class_name=self.active['settings']['field'], annotation_status='human_confirmed')
        metadata['last_propagation_attempt'] = self.proposal
        self.record = self.store.save(self.row(), self.active['roi'], state.image_rgb, state, metadata, negative)
        self.last_saved = {'frame': self.current, 'roi': self.active['roi'].copy(),
                           'rgb': state.image_rgb.copy(), 'mask': state.gt_mask.copy()}
        self.baseline = self.fingerprint()
        self.update_status()
        self.save_session()

    def set_reference(self):
        self.ensure_applied()
        if self.record is None or self.fingerprint() != self.baseline or np.any(self.editor.state.candidate_mask):
            raise ValueError('請先保存並確認目前 GT')
        self.reference = {'frame': self.current, 'roi': self.active['roi'].copy(),
                          'rgb': self.editor.state.image_rgb.copy(), 'mask': self.editor.state.gt_mask.copy()}
        self.update_status()
        self.editor.refresh_canvas()

    def predict(self):
        self.ensure_applied()
        if self.reference is None or self.reference['roi'] != self.active['roi']:
            raise ValueError('請先設定相同裁切 ROI 的已確認參考')
        proposal, metadata = translate_proposal(self.reference['rgb'], self.editor.state.image_rgb,
                                               self.reference['mask'], int(self.v['radius'].get()), float(self.v['score'].get()))
        state = self.editor.state
        if np.any(state.candidate_mask) and not messagebox.askyesno('替換 Candidate', '將取代目前 Candidate，GT 保留。繼續？'):
            return
        self.editor.history.push(state.candidate_mask, state.gt_mask)
        self.editor._candidate_drag = None
        state.candidate_mask = proposal
        self.proposal = {**metadata, 'reference_frame': self.reference['frame']}
        self.mode.set('current')
        self.editor._set_tool('candidate_edit')
        self.editor.status_text.set(f'傳播 Candidate score={metadata["score"]:.3f}；請修正並加入 GT。')
        self.editor.refresh_canvas()

    def import_last_gt(self):
        self.ensure_applied()
        source = self.last_saved
        if source is None or source['roi'] != self.active['roi']:
            raise ValueError('尚無最近保存的 GT；請先保存一個 frame 的標註。')
        if not np.any(source['mask']):
            raise ValueError('最近保存的是無物件標註，沒有可帶入的 mask。')
        state = self.editor.state
        if np.any(state.candidate_mask) and not messagebox.askyesno('替換 Candidate', '將取代目前 Candidate，GT 保留。繼續？'):
            return
        self.editor._candidate_drag = None
        self.editor.history.push(state.candidate_mask, state.gt_mask)
        state.candidate_mask = source['mask'].copy()
        self.reference = {key: value.copy() if hasattr(value, 'copy') else value for key, value in source.items()}
        self.proposal = {'method': 'saved_gt_copy', 'reference_frame': source['frame'], 'dx': 0, 'dy': 0}
        self.mode.set('current')
        self.editor._set_tool('candidate_move')
        self.editor.status_text.set(f'已帶入 frame {source["frame"]} 的 GT；左鍵拖曳 Candidate，A 編修，再加入 GT。')
        self.update_status()
        self.editor.refresh_canvas()

    def save_session(self):
        if self.active is None:
            raise ValueError('請先載入影片')
        settings = self.active['settings'].copy()
        for key in ('step', 'radius', 'score', 'frame'):
            settings[key] = self.v[key].get()
        for key in ('csv', 'video', 'out'):
            settings[key] = str(Path(settings[key]).resolve())
        folder = self.store.folder
        relative = {}
        for key in ('csv', 'video'):
            try:
                import os
                relative[key] = os.path.relpath(settings[key], folder)
            except ValueError:
                relative[key] = None
        atomic_json(folder / 'session.json', {'schema_version': 2, 'settings': settings,
                    'relative_sources': relative, 'source_id': self.active['identity'],
                    'samples': self.samples, 'current_frame': self.current, 'sample_cursor': self.cursor,
                    'last_saved_frame': self.last_saved['frame'] if self.last_saved else None,
                    'viewport': {'zoom': self.editor.zoom, 'view_x': self.editor.view_x, 'view_y': self.editor.view_y}})

    def open_session(self):
        path = filedialog.askopenfilename(filetypes=[('Session JSON', '*.json')])
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        settings = data['settings'].copy()
        for key in ('csv', 'video'):
            relative = data.get('relative_sources', {}).get(key)
            candidate = Path(path).parent / relative if relative else Path(settings[key])
            if candidate.is_file():
                settings[key] = str(candidate.resolve())
            elif not Path(settings[key]).is_file():
                replacement = filedialog.askopenfilename(title=f'重新指定 {key}')
                if not replacement:
                    return
                settings[key] = replacement
        # Old sessions keep their original output folder when it exists.
        if data.get('schema_version') == 2 or not Path(settings['out']).is_dir():
            settings['out'] = str(Path(path).parent.resolve())
        samples = sorted(data['samples'], key=lambda r: (float(r['distance_m']), int(r['frame_index'])))
        current = data.get('current_frame')
        cursor = next((i for i, row in enumerate(samples) if row['frame_index'] == current), data.get('sample_cursor', 0))
        if self.load_source(settings, samples, current, data.get('source_id'), cursor):
            self.last_saved = None
            recent = data.get('last_saved_frame')
            if recent is not None:
                mask, record = self.store.load(int(recent), self.active['roi'])
                if record is not None:
                    crop = validate_roi(self.active['roi'], self.reader.get(int(recent)))
                    self.last_saved = {'frame': int(recent), 'roi': self.active['roi'].copy(),
                                       'rgb': cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 'mask': mask.copy()}
            self.update_status()
            view = data.get('viewport')
            if view:
                zoom = float(view['zoom'])
                if math.isfinite(zoom) and .05 <= zoom <= self.editor.MAX_ZOOM:
                    self.editor.zoom = zoom
                    self.editor.view_x = float(view['view_x'])
                    self.editor.view_y = float(view['view_y'])
                    self.editor._update_zoom_text()
                    self.editor.refresh_canvas()

    def export_samples(self):
        self.ensure_applied()
        if not messagebox.askyesno('匯出取樣', f'匯出 {len(self.samples)} 張影像，期間介面會暫停回應。繼續？'):
            return
        out = self.store.folder / 'samples'
        out.mkdir(parents=True, exist_ok=True)
        records = []
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        try:
            for row in self.samples:
                crop = validate_roi(self.active['roi'], self.reader.get(row['frame_index']))
                path = out / (self.store.key(row['frame_index'], self.active['roi'])+'.png')
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(path)
                records.append({**row, 'image': path.name, 'crop_roi': json.dumps(self.active['roi'])})
            with (out/'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            self.status.set(f'已匯出 {len(records)} 張影像與排序清單')
        finally:
            self.root.configure(cursor='')

    def export_index(self):
        self.ensure_applied()
        self.status.set(f'已匯出 {self.store.export()} 個已確認標註')

    def close(self):
        if not self.leave():
            return
        if self.active:
            self.save_session()
        self.editor._image_generation += 1
        if self.reader:
            self.reader.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
