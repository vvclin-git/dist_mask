from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .candidate_generators import generate_candidate_from_palette
from .color_analysis import ColorAnalysisResult, analyze_roi_colors
from .editing import ADD, REMOVE, accept_candidate, apply_mask_stroke, candidate_action_at
from .history import MaskHistory
from .io_utils import load_rgb_image, save_annotation
from .mask_ops import keep_component_at, polygon_to_mask
from .model import AnnotationState
from .rendering import render_raster_annotation, visible_mask_pixels


CANDIDATE_HEX = "#ff50b4"
GT_HEX = "#00ffff"
HIGHLIGHT_HEX = "#ffdc00"
GRID_HEX = "#55585c"
POLYGON_HEX = "#ffffff"


class AnnotationApp:
    PIXEL_BORDER_ZOOM = 8.0
    MEDIUM_ZOOM = 4.0
    MAX_ZOOM = 32.0
    MAX_CELL_BORDERS = 16000

    def __init__(self, root: tk.Tk, host=None) -> None:
        self.root = root
        self.host = host or root
        if host is None:
            self.root.title("Pixel Mask Annotator v3")
            self.root.geometry("1480x920")
            self.root.minsize(1040, 680)

        self.state = AnnotationState()
        self.history = MaskHistory(max_steps=40)

        self.tool = tk.StringVar(value="polygon")
        self.display_mode = tk.StringVar(value="auto")
        self.brush_size = tk.IntVar(value=1)
        self.cluster_count = tk.IntVar(value=8)
        self.show_candidate = tk.BooleanVar(value=True)
        self.show_gt = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Open an image to begin.")
        self.zoom_text = tk.StringVar(value="100%")

        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._last_draw_point: tuple[int, int] | None = None
        self._stroke_active = False
        self._stroke_value: int | None = None
        self._pan_active = False
        self._pan_last: tuple[int, int] | None = None
        self._space_down = False
        self._highlight_cluster: int | None = None
        self._palette_vars: dict[int, tk.BooleanVar] = {}
        self._analysis_running = False
        self._analysis_queue: queue.Queue[tuple[ColorAnalysisResult | None, int, Exception | None]] = queue.Queue()
        self._image_generation = 0

        self._build_ui()
        self._bind_events()
        self._update_tool_hint()

    def _build_ui(self) -> None:
        self.host.columnconfigure(0, weight=1)
        self.host.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.host, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        if self.host is self.root:
            ttk.Button(toolbar, text="Open Image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save GT", command=self.save_current).pack(side="left", padx=(6, 12))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=(4, 2))
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Fit", command=self.fit_image).pack(side="left", padx=(8, 2))
        ttk.Label(toolbar, textvariable=self.zoom_text, width=7, anchor="center").pack(side="left", padx=(2, 10))

        ttk.Label(toolbar, text="View").pack(side="left")
        view_combo = ttk.Combobox(
            toolbar,
            state="readonly",
            width=14,
            textvariable=self.display_mode,
            values=("auto", "outline", "outline_fill", "fill", "hidden"),
        )
        view_combo.pack(side="left", padx=(4, 8))
        self.display_mode.trace_add("write", lambda *_: self.refresh_canvas())

        ttk.Checkbutton(
            toolbar,
            text="Candidate",
            variable=self.show_candidate,
            command=self.refresh_canvas,
        ).pack(side="left", padx=(2, 4))
        ttk.Checkbutton(
            toolbar,
            text="GT",
            variable=self.show_gt,
            command=self.refresh_canvas,
        ).pack(side="left", padx=2)

        body = ttk.Panedwindow(self.host, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew")

        canvas_frame = ttk.Frame(body)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        body.add(canvas_frame, weight=5)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#1d1f21",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, padding=10)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(0, weight=1)
        body.add(side, weight=1)

        side_canvas = tk.Canvas(side, highlightthickness=0, width=310)
        side_scroll = ttk.Scrollbar(side, orient="vertical", command=side_canvas.yview)
        side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scroll.grid(row=0, column=1, sticky="ns")
        side_canvas.configure(yscrollcommand=side_scroll.set)

        panel = ttk.Frame(side_canvas, padding=(0, 0, 6, 6))
        panel_window = side_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.columnconfigure(0, weight=1)
        panel.bind("<Configure>", lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(panel_window, width=e.width))

        self._build_assist_panel(panel)
        self._build_candidate_panel(panel)
        self._build_gt_panel(panel)
        self._build_controls_panel(panel)

        status = ttk.Frame(self.host, padding=(8, 4))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_text, anchor="w").grid(row=0, column=0, sticky="ew")
        self.tool_hint = ttk.Label(status, anchor="e")
        self.tool_hint.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _build_assist_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1  ASSIST", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        ttk.Label(
            box,
            text="Polygon and color analysis only help generate the candidate mask.",
            wraplength=270,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        roi_row = ttk.Frame(box)
        roi_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Radiobutton(
            roi_row,
            text="Polygon ROI",
            value="polygon",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left")
        ttk.Button(roi_row, text="Close", command=self.close_polygon).pack(side="left", padx=(8, 3))
        ttk.Button(roi_row, text="Clear ROI", command=self.clear_polygon).pack(side="left", padx=3)

        color_controls = ttk.Frame(box)
        color_controls.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(color_controls, text="Clusters").pack(side="left")
        ttk.Spinbox(color_controls, from_=2, to=16, width=4, textvariable=self.cluster_count).pack(side="left", padx=4)
        self.analyze_button = ttk.Button(color_controls, text="Analyze Colors", command=self.analyze_colors)
        self.analyze_button.pack(side="left", padx=(4, 0))

        palette_holder = ttk.Frame(box)
        palette_holder.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        palette_holder.rowconfigure(0, weight=1)
        palette_holder.columnconfigure(0, weight=1)
        self.palette_canvas = tk.Canvas(palette_holder, height=235, highlightthickness=0)
        self.palette_canvas.grid(row=0, column=0, sticky="nsew")
        palette_scroll = ttk.Scrollbar(palette_holder, orient="vertical", command=self.palette_canvas.yview)
        palette_scroll.grid(row=0, column=1, sticky="ns")
        self.palette_canvas.configure(yscrollcommand=palette_scroll.set)
        self.palette_frame = ttk.Frame(self.palette_canvas)
        self.palette_window = self.palette_canvas.create_window((0, 0), window=self.palette_frame, anchor="nw")
        self.palette_frame.bind("<Configure>", self._sync_palette_scrollregion)
        self.palette_canvas.bind("<Configure>", self._resize_palette_window)

        ttk.Button(
            box,
            text="Generate Candidate from Palette",
            command=self.generate_candidate,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    def _build_candidate_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2  CANDIDATE", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        box.columnconfigure(0, weight=1)

        ttk.Label(
            box,
            text="Click candidate pixels to remove; click background to add. Drag keeps the action chosen on mouse-down.",
            wraplength=270,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        tool_row = ttk.Frame(box)
        tool_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Radiobutton(
            tool_row,
            text="Toggle Edit",
            value="candidate_edit",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left")
        ttk.Radiobutton(
            tool_row,
            text="Keep Component",
            value="component",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left", padx=(8, 0))

        brush_row = ttk.Frame(box)
        brush_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(brush_row, text="Brush size").pack(side="left")
        ttk.Spinbox(brush_row, from_=1, to=31, width=5, textvariable=self.brush_size).pack(side="left", padx=5)
        ttk.Label(brush_row, text="px").pack(side="left")

        action_row = ttk.Frame(box)
        action_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        action_row.columnconfigure((0, 1), weight=1)
        ttk.Button(action_row, text="Clear Candidate", command=self.clear_candidate).grid(row=0, column=0, sticky="ew")
        ttk.Button(action_row, text="Add Candidate to GT (Union)", command=self.accept_candidate_to_gt).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

    def _build_gt_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="3  GROUND TRUTH", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        box.columnconfigure(0, weight=1)

        ttk.Label(
            box,
            text="GT is the final annotation layer. Use explicit Brush or Eraser for pixel-level refinement.",
            wraplength=270,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        tool_row = ttk.Frame(box)
        tool_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Radiobutton(
            tool_row,
            text="Brush",
            value="gt_brush",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left")
        ttk.Radiobutton(
            tool_row,
            text="Eraser",
            value="gt_eraser",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(
            tool_row,
            text="Pan",
            value="pan",
            variable=self.tool,
            command=self._update_tool_hint,
        ).pack(side="left", padx=(8, 0))

        ttk.Button(box, text="Clear GT Mask", command=self.clear_gt).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Controls", padding=8)
        box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(
            box,
            text=(
                "Wheel: zoom around cursor\n"
                "Space + drag / middle drag: pan\n"
                "P: polygon   A: candidate edit\n"
                "C: component   B: GT brush   E: GT eraser\n"
                "Enter: close polygon   Esc: clear ROI\n"
                "Ctrl+Z / Ctrl+Y: undo / redo\n"
                "Auto view: ≥800% uses pixel-cell borders"
            ),
            justify="left",
            wraplength=270,
        ).pack(anchor="w")

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda e: self.refresh_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._hide_edit_cursor())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))

        self.root.bind("<KeyPress-space>", self._on_space_down)
        self.root.bind("<KeyRelease-space>", self._on_space_up)
        self.root.bind("<Return>", lambda e: self.close_polygon())
        self.root.bind("<Escape>", lambda e: self.clear_polygon())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("p", lambda e: self._set_tool("polygon"))
        self.root.bind("a", lambda e: self._set_tool("candidate_edit"))
        self.root.bind("c", lambda e: self._set_tool("component"))
        self.root.bind("b", lambda e: self._set_tool("gt_brush"))
        self.root.bind("e", lambda e: self._set_tool("gt_eraser"))

    def _set_tool(self, tool: str) -> None:
        self.tool.set(tool)
        self._update_tool_hint()

    def _update_tool_hint(self) -> None:
        hints = {
            "polygon": "Polygon ROI",
            "candidate_edit": "Candidate toggle: click + / −",
            "component": "Keep clicked candidate component",
            "gt_brush": "GT Brush",
            "gt_eraser": "GT Eraser",
            "pan": "Pan",
        }
        if hasattr(self, "tool_hint"):
            self.tool_hint.configure(text=hints.get(self.tool.get(), ""))

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            image = load_rgb_image(path)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        self._image_generation += 1
        self.state.reset_for_image(Path(path), image)
        self.history.clear()
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._highlight_cluster = None
        self._clear_palette_ui()
        self._set_tool("polygon")
        self.root.update_idletasks()
        self._fit_image_to_canvas()
        self.status_text.set(f"Loaded {Path(path).name} — {image.shape[1]} × {image.shape[0]}")
        self.refresh_canvas()

    def save_current(self) -> None:
        if not self.state.ready:
            return
        initial = str(self.state.image_path.parent if self.state.image_path else Path.cwd())
        output = filedialog.askdirectory(title="Select output folder", initialdir=initial)
        if not output:
            return
        try:
            mask_path, meta_path = save_annotation(self.state, output)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status_text.set(f"Saved {mask_path.name} and {meta_path.name}")

    def fit_image(self) -> None:
        if not self.state.ready:
            return
        self._fit_image_to_canvas()
        self.refresh_canvas()

    def _fit_image_to_canvas(self) -> None:
        if not self.state.ready or self.state.image_rgb is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        h, w = self.state.image_rgb.shape[:2]
        self.zoom = min(cw / w, ch / h, 1.0)
        visible_w = cw / self.zoom
        visible_h = ch / self.zoom
        self.view_x = (w - visible_w) / 2
        self.view_y = (h - visible_h) / 2
        self._clamp_view()
        self._update_zoom_text()

    def _update_zoom_text(self) -> None:
        self.zoom_text.set(f"{self.zoom * 100:.0f}%")

    def _clamp_view(self) -> None:
        if not self.state.ready or self.state.image_rgb is None:
            return
        h, w = self.state.image_rgb.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        visible_w = cw / self.zoom
        visible_h = ch / self.zoom
        if visible_w >= w:
            self.view_x = (w - visible_w) / 2
        else:
            self.view_x = min(max(self.view_x, 0.0), w - visible_w)
        if visible_h >= h:
            self.view_y = (h - visible_h) / 2
        else:
            self.view_y = min(max(self.view_y, 0.0), h - visible_h)

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.view_x) * self.zoom, (y - self.view_y) * self.zoom)

    def canvas_to_image(self, x: float, y: float, clamp: bool = True) -> tuple[int, int]:
        ix = int(np.floor(x / self.zoom + self.view_x))
        iy = int(np.floor(y / self.zoom + self.view_y))
        if clamp and self.state.image_rgb is not None:
            h, w = self.state.image_rgb.shape[:2]
            ix = min(max(ix, 0), w - 1)
            iy = min(max(iy, 0), h - 1)
        return ix, iy

    def _visible_crop_bounds(self) -> tuple[int, int, int, int]:
        assert self.state.image_rgb is not None
        h, w = self.state.image_rgb.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        x0 = max(0, int(np.floor(self.view_x)))
        y0 = max(0, int(np.floor(self.view_y)))
        x1 = min(w, int(np.ceil(self.view_x + cw / self.zoom)))
        y1 = min(h, int(np.ceil(self.view_y + ch / self.zoom)))
        return x0, y0, x1, y1

    def _resolved_view_mode(self) -> str:
        mode = self.display_mode.get()
        if mode != "auto":
            return mode
        if self.zoom >= self.PIXEL_BORDER_ZOOM:
            return "pixel_border"
        if self.zoom >= self.MEDIUM_ZOOM:
            return "outline_fill"
        return "outline"

    def refresh_canvas(self) -> None:
        if not self.state.ready or self.state.image_rgb is None or self.state.gt_mask is None:
            self.canvas.delete("all")
            self._canvas_image_id = None
            return

        self._clamp_view()
        x0, y0, x1, y1 = self._visible_crop_bounds()
        if x1 <= x0 or y1 <= y0:
            return

        view_mode = self._resolved_view_mode()
        pixel_border_mode = view_mode == "pixel_border"

        highlight = None
        if self._highlight_cluster is not None and self.state.cluster_map is not None:
            highlight = (self.state.cluster_map == self._highlight_cluster).astype(np.uint8)

        rendered = render_raster_annotation(
            self.state.image_rgb[y0:y1, x0:x1],
            self.state.gt_mask[y0:y1, x0:x1],
            None if self.state.candidate_mask is None else self.state.candidate_mask[y0:y1, x0:x1],
            display_mode="hidden" if pixel_border_mode else view_mode,
            candidate_visible=self.show_candidate.get(),
            gt_visible=self.show_gt.get(),
            highlighted_mask=None if pixel_border_mode or highlight is None else highlight[y0:y1, x0:x1],
            preserve_mask_pixels=pixel_border_mode,
        )

        target_w = max(1, int(round((x1 - x0) * self.zoom)))
        target_h = max(1, int(round((y1 - y0) * self.zoom)))
        pil = Image.fromarray(rendered)
        resample = Image.Resampling.NEAREST if self.zoom >= 2.0 else Image.Resampling.BILINEAR
        pil = pil.resize((target_w, target_h), resample=resample)
        self._photo = ImageTk.PhotoImage(pil)

        sx, sy = self.image_to_canvas(x0, y0)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(sx, sy, image=self._photo, anchor="nw", tags=("base",))
        else:
            self.canvas.coords(self._canvas_image_id, sx, sy)
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)

        self.canvas.delete("pixel_grid")
        self.canvas.delete("mask_cell_border")
        self.canvas.delete("cluster_cell_border")
        self.canvas.delete("polygon_overlay")
        self.canvas.delete("edit_cursor")

        if self.zoom >= self.PIXEL_BORDER_ZOOM:
            self._draw_pixel_grid()
        if pixel_border_mode:
            if highlight is not None:
                self._draw_mask_cell_borders(highlight, HIGHLIGHT_HEX, "cluster_cell_border", width=1)
            if self.show_candidate.get() and self.state.candidate_mask is not None:
                self._draw_mask_cell_borders(self.state.candidate_mask, CANDIDATE_HEX, "mask_cell_border")
            if self.show_gt.get():
                self._draw_mask_cell_borders(self.state.gt_mask, GT_HEX, "mask_cell_border")

        self._draw_polygon_overlay()
        if self._canvas_image_id is not None:
            self.canvas.tag_lower(self._canvas_image_id)

    def _draw_polygon_overlay(self) -> None:
        if not self.state.polygon_points:
            return
        coords: list[float] = []
        for x, y in self.state.polygon_points:
            sx, sy = self.image_to_canvas(x + 0.5, y + 0.5)
            coords.extend([sx, sy])
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=POLYGON_HEX, width=2, tags=("polygon_overlay",))
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i + 1]
            self.canvas.create_oval(
                x - 3,
                y - 3,
                x + 3,
                y + 3,
                fill=POLYGON_HEX,
                outline="#000000",
                tags=("polygon_overlay",),
            )
        if self.state.roi_mask is not None and np.any(self.state.roi_mask) and len(coords) >= 6:
            self.canvas.create_line(
                *coords,
                coords[0],
                coords[1],
                fill=POLYGON_HEX,
                width=2,
                dash=(6, 4),
                tags=("polygon_overlay",),
            )

    def _draw_pixel_grid(self) -> None:
        if self.state.image_rgb is None:
            return
        x0, y0, x1, y1 = self._visible_crop_bounds()
        if (x1 - x0) + (y1 - y0) > 600:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        for x in range(x0, x1 + 1):
            sx, _ = self.image_to_canvas(x, 0)
            if 0 <= sx <= cw:
                self.canvas.create_line(sx, 0, sx, ch, fill=GRID_HEX, width=1, tags=("pixel_grid",))
        for y in range(y0, y1 + 1):
            _, sy = self.image_to_canvas(0, y)
            if 0 <= sy <= ch:
                self.canvas.create_line(0, sy, cw, sy, fill=GRID_HEX, width=1, tags=("pixel_grid",))

    def _draw_mask_cell_borders(self, mask: np.ndarray, color: str, tag: str, width: int | None = None) -> None:
        bounds = self._visible_crop_bounds()
        coords = visible_mask_pixels(mask, bounds)
        if len(coords) == 0:
            return
        if len(coords) > self.MAX_CELL_BORDERS:
            self.status_text.set(
                f"Pixel-border view skipped: {len(coords):,} visible mask pixels. Zoom in further for cell borders."
            )
            return
        line_width = width if width is not None else (2 if self.zoom >= 12.0 else 1)
        for x, y in coords:
            sx0, sy0 = self.image_to_canvas(float(x), float(y))
            sx1, sy1 = self.image_to_canvas(float(x + 1), float(y + 1))
            self.canvas.create_rectangle(
                sx0,
                sy0,
                sx1,
                sy1,
                outline=color,
                width=line_width,
                tags=(tag,),
            )

    def _on_left_press(self, event: tk.Event) -> None:
        if not self.state.ready:
            return
        self.canvas.focus_set()
        if self._space_down or self.tool.get() == "pan":
            self._on_pan_press(event)
            return

        ix, iy = self.canvas_to_image(event.x, event.y, clamp=False)
        h, w = self.state.image_rgb.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return
        tool = self.tool.get()

        if tool == "polygon":
            if self.state.roi_mask is not None and np.any(self.state.roi_mask):
                self._reset_roi_analysis(clear_points=True)
            self.state.polygon_points.append((ix, iy))
            self.refresh_canvas()
            return

        if tool == "candidate_edit":
            if self.state.candidate_mask is None or self.state.gt_mask is None:
                return
            self.history.push(self.state.candidate_mask, self.state.gt_mask)
            self._stroke_value = candidate_action_at(self.state.candidate_mask, ix, iy)
            self._stroke_active = True
            self._last_draw_point = (ix, iy)
            apply_mask_stroke(
                self.state.candidate_mask,
                (ix, iy),
                (ix, iy),
                self.brush_size.get(),
                self._stroke_value,
            )
            self.refresh_canvas()
            self._draw_edit_cursor(event)
            return

        if tool in {"gt_brush", "gt_eraser"}:
            if self.state.candidate_mask is None or self.state.gt_mask is None:
                return
            self.history.push(self.state.candidate_mask, self.state.gt_mask)
            self._stroke_value = ADD if tool == "gt_brush" else REMOVE
            self._stroke_active = True
            self._last_draw_point = (ix, iy)
            apply_mask_stroke(
                self.state.gt_mask,
                (ix, iy),
                (ix, iy),
                self.brush_size.get(),
                self._stroke_value,
            )
            self.refresh_canvas()
            self._draw_edit_cursor(event)
            return

        if tool == "component":
            self._pick_component(ix, iy)

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._pan_active:
            self._on_pan_drag(event)
            return
        if not self._stroke_active or self._last_draw_point is None or self._stroke_value is None:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)
        tool = self.tool.get()
        target = None
        if tool == "candidate_edit":
            target = self.state.candidate_mask
        elif tool in {"gt_brush", "gt_eraser"}:
            target = self.state.gt_mask
        if target is None:
            return

        apply_mask_stroke(target, self._last_draw_point, (ix, iy), self.brush_size.get(), self._stroke_value)
        self._last_draw_point = (ix, iy)
        self.refresh_canvas()
        self._draw_edit_cursor(event)

    def _on_left_release(self, event: tk.Event) -> None:
        if self._pan_active:
            self._on_pan_release(event)
        self._stroke_active = False
        self._stroke_value = None
        self._last_draw_point = None

    def _on_pan_press(self, event: tk.Event) -> None:
        self._pan_active = True
        self._pan_last = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _on_pan_drag(self, event: tk.Event) -> None:
        if not self._pan_active or self._pan_last is None:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self.view_x -= dx / self.zoom
        self.view_y -= dy / self.zoom
        self._pan_last = (event.x, event.y)
        self.refresh_canvas()

    def _on_pan_release(self, event: tk.Event) -> None:
        self._pan_active = False
        self._pan_last = None
        self.canvas.configure(cursor="crosshair")

    def _on_space_down(self, event: tk.Event) -> None:
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        self._space_down = True
        self.canvas.configure(cursor="fleur")

    def _on_space_up(self, event: tk.Event) -> None:
        self._space_down = False
        if not self._pan_active:
            self.canvas.configure(cursor="crosshair")

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        if not self.state.ready:
            return
        before_x = cx / self.zoom + self.view_x
        before_y = cy / self.zoom + self.view_y
        new_zoom = min(max(self.zoom * factor, 0.05), self.MAX_ZOOM)
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self.zoom = new_zoom
        self.view_x = before_x - cx / self.zoom
        self.view_y = before_y - cy / self.zoom
        self._update_zoom_text()
        self.refresh_canvas()
        self.status_text.set(f"Zoom {self.zoom * 100:.0f}%")

    def _on_motion(self, event: tk.Event) -> None:
        if not self.state.ready:
            return
        if self._pan_active:
            self._hide_edit_cursor()
            return
        self._draw_edit_cursor(event)

    def _draw_edit_cursor(self, event: tk.Event) -> None:
        self.canvas.delete("edit_cursor")
        tool = self.tool.get()
        if tool not in {"candidate_edit", "gt_brush", "gt_eraser"}:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)
        size = max(1, self.brush_size.get())
        radius = max(4.0, size * self.zoom / 2.0)
        color = GT_HEX
        symbol = "+"

        if tool == "candidate_edit":
            color = CANDIDATE_HEX
            if self._stroke_active and self._stroke_value is not None:
                action = self._stroke_value
            elif self.state.candidate_mask is not None:
                action = candidate_action_at(self.state.candidate_mask, ix, iy)
            else:
                action = ADD
            symbol = "+" if action == ADD else "−"
        elif tool == "gt_eraser":
            symbol = "−"

        if self.zoom >= self.PIXEL_BORDER_ZOOM and size == 1:
            sx0, sy0 = self.image_to_canvas(ix, iy)
            sx1, sy1 = self.image_to_canvas(ix + 1, iy + 1)
            self.canvas.create_rectangle(
                sx0,
                sy0,
                sx1,
                sy1,
                outline=color,
                width=2,
                dash=(3, 2),
                tags=("edit_cursor",),
            )
            cx = (sx0 + sx1) / 2
            cy = (sy0 + sy1) / 2
        else:
            self.canvas.create_oval(
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius,
                outline=color,
                width=2,
                tags=("edit_cursor",),
            )
            cx, cy = event.x, event.y

        if radius >= 7 or self.zoom >= self.MEDIUM_ZOOM:
            self.canvas.create_text(
                cx,
                cy,
                text=symbol,
                fill=color,
                font=("TkDefaultFont", 11, "bold"),
                tags=("edit_cursor",),
            )

    def _hide_edit_cursor(self) -> None:
        self.canvas.delete("edit_cursor")

    def _reset_roi_analysis(self, clear_points: bool) -> None:
        if not self.state.ready or self.state.image_rgb is None:
            return
        self._image_generation += 1
        h, w = self.state.image_rgb.shape[:2]
        if clear_points:
            self.state.polygon_points = []
        self.state.roi_mask = np.zeros((h, w), dtype=np.uint8)
        self.state.cluster_map = np.full((h, w), -1, dtype=np.int16)
        self.state.palette = []
        self._highlight_cluster = None
        self._clear_palette_ui()

    def close_polygon(self) -> None:
        if not self.state.ready or self.state.image_rgb is None:
            return
        if len(self.state.polygon_points) < 3:
            self.status_text.set("Polygon needs at least 3 points.")
            return
        h, w = self.state.image_rgb.shape[:2]
        self._image_generation += 1
        self.state.roi_mask = polygon_to_mask((h, w), self.state.polygon_points)
        self.state.cluster_map = np.full((h, w), -1, dtype=np.int16)
        self.state.palette = []
        self._highlight_cluster = None
        self._clear_palette_ui()
        self.status_text.set(f"Polygon ROI closed — {int(self.state.roi_mask.sum())} pixels")
        self.refresh_canvas()

    def clear_polygon(self) -> None:
        if not self.state.ready:
            return
        self._reset_roi_analysis(clear_points=True)
        self.status_text.set("ROI and color analysis cleared. Candidate and GT are unchanged.")
        self.refresh_canvas()

    def analyze_colors(self) -> None:
        if self._analysis_running:
            return
        if not self.state.ready or self.state.image_rgb is None or self.state.roi_mask is None:
            return
        if not np.any(self.state.roi_mask):
            messagebox.showinfo("ROI required", "Draw and close a polygon ROI first.")
            return

        image = self.state.image_rgb.copy()
        roi = self.state.roi_mask.copy()
        clusters = int(self.cluster_count.get())
        generation = self._image_generation
        self._analysis_running = True
        self.analyze_button.configure(state="disabled")
        self.status_text.set("Analyzing ROI colors...")

        def worker() -> None:
            try:
                result = analyze_roi_colors(image, roi, clusters=clusters)
                self._analysis_queue.put((result, generation, None))
            except Exception as exc:
                self._analysis_queue.put((None, generation, exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_color_analysis)

    def _poll_color_analysis(self) -> None:
        try:
            result, generation, error = self._analysis_queue.get_nowait()
        except queue.Empty:
            if self._analysis_running:
                self.root.after(50, self._poll_color_analysis)
            return
        self._finish_color_analysis(result, generation, error)

    def _finish_color_analysis(
        self,
        result: ColorAnalysisResult | None,
        generation: int,
        error: Exception | None,
    ) -> None:
        self._analysis_running = False
        self.analyze_button.configure(state="normal")
        if generation != self._image_generation:
            return
        if error is not None:
            messagebox.showerror("Color analysis failed", str(error))
            self.status_text.set("Color analysis failed.")
            return
        assert result is not None
        self.state.cluster_map = result.cluster_map
        self.state.palette = result.palette
        self._build_palette_ui()
        self.status_text.set(
            f"Color analysis complete — {len(result.palette)} clusters. Select colors, then Generate Candidate."
        )
        self.refresh_canvas()

    def _build_palette_ui(self) -> None:
        self._clear_palette_ui()
        self._palette_vars = {}
        for row, entry in enumerate(self.state.palette):
            line = ttk.Frame(self.palette_frame, padding=(0, 2))
            line.grid(row=row, column=0, sticky="ew")
            line.columnconfigure(2, weight=1)

            swatch = tk.Label(line, width=3, background=self._rgb_hex(entry.rgb), relief="solid", borderwidth=1)
            swatch.grid(row=0, column=0, padx=(0, 6), sticky="ns")
            swatch.bind("<Enter>", lambda e, cid=entry.cluster_id: self._set_highlight(cid))
            swatch.bind("<Leave>", lambda e: self._set_highlight(None))

            var = tk.BooleanVar(value=entry.keep)
            self._palette_vars[entry.cluster_id] = var
            cb = ttk.Checkbutton(
                line,
                variable=var,
                command=lambda cid=entry.cluster_id: self._toggle_cluster_selection(cid),
            )
            cb.grid(row=0, column=1)

            label = ttk.Label(
                line,
                text=f"#{entry.cluster_id}  {entry.count:,} px  ({entry.fraction * 100:.1f}%)",
            )
            label.grid(row=0, column=2, sticky="w")
            label.bind("<Enter>", lambda e, cid=entry.cluster_id: self._set_highlight(cid))
            label.bind("<Leave>", lambda e: self._set_highlight(None))

    def _clear_palette_ui(self) -> None:
        if not hasattr(self, "palette_frame"):
            return
        for child in self.palette_frame.winfo_children():
            child.destroy()
        self._palette_vars = {}

    def _toggle_cluster_selection(self, cluster_id: int) -> None:
        for entry in self.state.palette:
            if entry.cluster_id == cluster_id:
                entry.keep = bool(self._palette_vars[cluster_id].get())
                break
        selected = sum(1 for entry in self.state.palette if entry.keep)
        self.status_text.set(f"Palette selection: {selected}/{len(self.state.palette)} clusters. Candidate is unchanged until Generate.")

    def _set_highlight(self, cluster_id: int | None) -> None:
        self._highlight_cluster = cluster_id
        self.refresh_canvas()

    def generate_candidate(self) -> None:
        if not self.state.ready or self.state.cluster_map is None or not self.state.palette:
            messagebox.showinfo("Color analysis required", "Analyze the polygon ROI colors first.")
            return
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        self.history.push(self.state.candidate_mask, self.state.gt_mask)
        self.state.candidate_mask = generate_candidate_from_palette(self.state.cluster_map, self.state.palette)
        self._set_tool("candidate_edit")
        self.status_text.set(f"Candidate generated — {int(self.state.candidate_mask.sum())} pixels")
        self.refresh_canvas()

    def _pick_component(self, x: int, y: int) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None or not np.any(self.state.candidate_mask):
            self.status_text.set("No candidate mask to select from.")
            return
        if self.state.candidate_mask[y, x] == 0:
            self.status_text.set("Click inside a candidate component.")
            return
        self.history.push(self.state.candidate_mask, self.state.gt_mask)
        self.state.candidate_mask = keep_component_at(self.state.candidate_mask, x, y)
        self.status_text.set(f"Kept connected component — {int(self.state.candidate_mask.sum())} pixels")
        self.refresh_canvas()

    def clear_candidate(self) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        if not np.any(self.state.candidate_mask):
            return
        self.history.push(self.state.candidate_mask, self.state.gt_mask)
        self.state.candidate_mask.fill(0)
        self.status_text.set("Candidate cleared.")
        self.refresh_canvas()

    def accept_candidate_to_gt(self) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        if not np.any(self.state.candidate_mask):
            self.status_text.set("Candidate mask is empty.")
            return
        accepted_pixels = int(self.state.candidate_mask.sum())
        self.history.push(self.state.candidate_mask, self.state.gt_mask)
        self.state.gt_mask, self.state.candidate_mask = accept_candidate(
            self.state.gt_mask,
            self.state.candidate_mask,
        )
        self._set_tool("gt_brush")
        self.status_text.set(
            f"Candidate accepted ({accepted_pixels} px) — GT area {int(self.state.gt_mask.sum())} pixels; candidate cleared."
        )
        self.refresh_canvas()

    def clear_gt(self) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        if not np.any(self.state.gt_mask):
            return
        self.history.push(self.state.candidate_mask, self.state.gt_mask)
        self.state.gt_mask.fill(0)
        self.status_text.set("GT mask cleared.")
        self.refresh_canvas()

    def undo(self) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        previous = self.history.undo(self.state.candidate_mask, self.state.gt_mask)
        if previous is None:
            self.status_text.set("Nothing to undo.")
            return
        self.state.candidate_mask, self.state.gt_mask = previous
        self.status_text.set("Undo")
        self.refresh_canvas()

    def redo(self) -> None:
        if self.state.candidate_mask is None or self.state.gt_mask is None:
            return
        next_state = self.history.redo(self.state.candidate_mask, self.state.gt_mask)
        if next_state is None:
            self.status_text.set("Nothing to redo.")
            return
        self.state.candidate_mask, self.state.gt_mask = next_state
        self.status_text.set("Redo")
        self.refresh_canvas()

    def _sync_palette_scrollregion(self, event: tk.Event) -> None:
        self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all"))

    def _resize_palette_window(self, event: tk.Event) -> None:
        self.palette_canvas.itemconfigure(self.palette_window, width=event.width)

    @staticmethod
    def _rgb_hex(rgb: tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % rgb


def main() -> None:
    root = tk.Tk()
    AnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
