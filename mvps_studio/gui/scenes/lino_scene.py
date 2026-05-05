"""
mvps_studio.gui.scenes.lino_scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tab 4 – LINO-UniPS Normal Extraction UI.

Grid page  → sidebar (config, exec controls, run history, zoom slider)
              + scrollable FlowLayout of NormalCards
              Ctrl+Scroll on the grid = zoom cards

Detail page → header (Back / Prev View / title / Next View)
              + ZoomPanImageView (scroll=zoom, drag=pan, dbl-click=fit)
              + fixed-height info row (frame label, above the slider row)
              + scrub bar:  ◀  [────slider────]  ▶
              Keyboard: ← / → = prev/next frame
                         Shift+← / Shift+→ = prev/next view
                         Esc = back to grid
"""

import os
import json

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressBar,
    QSpinBox, QComboBox, QScrollArea, QFrame, QStackedWidget, QSlider,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QFont, QImage, QPixmap, QKeySequence, QShortcut

from mvps_studio.gui.widgets.flow_layout import FlowLayout
from mvps_studio.gui.widgets.zoom_pan_image_view import ZoomPanImageView


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
def _np_to_pixmap(arr):
    """Convert a numpy uint8 array (grayscale or RGB) to QPixmap."""
    if arr is None:
        return QPixmap()
    if len(arr.shape) == 2:
        h, w = arr.shape
        return QPixmap.fromImage(
            QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy())
    h, w, c = arr.shape
    if c == 3:
        bpl = 3 * w
        return QPixmap.fromImage(
            QImage(arr.data, w, h, bpl, QImage.Format.Format_RGB888).copy())
    return QPixmap()


def _normal_to_rgb(path):
    """Load a normal NPY/EXR and convert to viewable uint8 RGB."""
    try:
        if path.endswith(".npy"):
            nml = np.load(path)
        else:
            import pyexr
            nml = pyexr.read(path)
        rgb = ((nml * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return rgb
    except Exception:
        return None


# ---------------------------------------------------------------
# NormalCard  – one cell in the grid  (mirrors HybridGroupCard)
# ---------------------------------------------------------------
class NormalCard(QFrame):
    clicked = Signal(int)

    def __init__(self, group_idx, parent=None):
        super().__init__(parent)
        self.group_idx = group_idx
        self.has_normal = False
        self._full_pixmap  = QPixmap()   # camera-frame
        self._world_pixmap = QPixmap()   # world-frame (may be null)
        self._mode         = "camera"    # "camera" | "world"

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.img_label, 1)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self._apply_style(False)

    def _apply_style(self, has_normal):
        self.has_normal = has_normal
        border = "#28a745" if has_normal else "#ced4da"
        self.setStyleSheet(
            f"NormalCard {{ border: 2px solid {border}; background: #fff; "
            f"border-radius: 6px; }}"
            f"NormalCard:hover {{ background: #f1f3f5; }}")

    def _active_pixmap(self):
        if self._mode == "world" and not self._world_pixmap.isNull():
            return self._world_pixmap
        return self._full_pixmap

    def _display(self, zoom_size):
        pm = self._active_pixmap()
        if not pm.isNull():
            self.img_label.setPixmap(
                pm.scaled(zoom_size - 10, zoom_size - 10,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))

    def set_content(self, pixmap, has_normal, zoom_size):
        tag = "✅ Normal" if has_normal else "⏳ Pending"
        self.title_label.setText(f"<b>View {self.group_idx:02d}</b>  {tag}")
        self._apply_style(has_normal)
        if pixmap and not pixmap.isNull():
            self._full_pixmap = pixmap
        self.setFixedSize(zoom_size, zoom_size + 30)
        self._display(zoom_size)

    def set_world_content(self, pixmap):
        """Store the world-frame pixmap; re-renders if currently in world mode."""
        if pixmap and not pixmap.isNull():
            self._world_pixmap = pixmap
            if self._mode == "world":
                self._display(self.width())

    def set_zoom(self, zoom_size):
        self.setFixedSize(zoom_size, zoom_size + 30)
        self._display(zoom_size)

    def set_display_mode(self, mode):
        """Switch between 'camera' and 'world' and refresh."""
        self._mode = mode
        self._display(self.width())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.group_idx)


# ---------------------------------------------------------------
# LinoScene  – main tab widget
# ---------------------------------------------------------------
class LinoScene(QWidget):
    start_lino     = Signal(int, int)       # resolution, max_images
    resume_lino    = Signal(str, int, int)  # run_dir, resolution, max_images
    convert_normals = Signal(str)           # run_dir  (app.py adds colmap path)
    request_pause  = Signal()
    request_resume = Signal()
    request_cancel = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_run_dir = None
        self._workspace_dir   = None
        self._run_cfg         = None
        self._detail_frames   = []
        self._active_view_idx = -1
        self._zoom_size       = 260
        self._normal_mode     = "camera"    # "camera" | "world"
        self._setup_ui()
        self._setup_shortcuts()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        self._build_grid_page()
        self._build_detail_page()

    # ------------------------------------------------------------------
    def _build_grid_page(self):
        self.page_grid = QWidget()
        grid_outer = QHBoxLayout(self.page_grid)
        grid_outer.setContentsMargins(0, 0, 0, 0)

        # ── LEFT sidebar ──────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar.setFixedWidth(320)
        sl = QVBoxLayout(sidebar)

        # config -------------------------------------------------------
        sl.addWidget(QLabel("<b>Configuration</b>"))

        sl.addWidget(QLabel("Max side resolution (px):"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(
            ["512", "768", "960 (Default)", "1024", "1280", "1536", "2048"])
        self.combo_res.setCurrentIndex(2)
        sl.addWidget(self.combo_res)

        sl.addWidget(QLabel("Max images per group:"))
        self.spin_images = QSpinBox()
        self.spin_images.setRange(1, 100)
        self.spin_images.setValue(8)
        sl.addWidget(self.spin_images)

        sl.addSpacing(10)

        # execution ----------------------------------------------------
        sl.addWidget(QLabel("<b>Execution</b>"))

        self.btn_run_lino = QPushButton("🚀 Run LINO-UniPS")
        self.btn_run_lino.setStyleSheet(
            "background:#0d6efd; color:white; font-weight:bold; padding:10px;")
        self.btn_run_lino.clicked.connect(self._on_run_clicked)
        sl.addWidget(self.btn_run_lino)

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_resume = QPushButton("▶ Resume")
        self.btn_cancel = QPushButton("✖ Cancel")
        for b in (self.btn_pause, self.btn_resume, self.btn_cancel):
            b.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_resume.clicked.connect(self._on_resume)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_resume)
        btn_row.addWidget(self.btn_cancel)
        sl.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        sl.addWidget(self.progress_bar)
        self.lbl_status = QLabel("Ready")
        sl.addWidget(self.lbl_status)

        self.btn_resume_run = QPushButton("⏩ Resume Incomplete Run")
        self.btn_resume_run.setStyleSheet(
            "background:#fd7e14; color:white; font-weight:bold; padding:6px;")
        self.btn_resume_run.setVisible(False)
        self.btn_resume_run.clicked.connect(self._on_resume_run_clicked)
        sl.addWidget(self.btn_resume_run)

        sl.addSpacing(10)

        # normal display mode ------------------------------------------
        sl.addWidget(QLabel("<b>Normal Frame:</b>"))
        mode_row = QHBoxLayout()
        self.btn_show_cam = QPushButton("📷 Camera")
        self.btn_show_cam.setCheckable(True)
        self.btn_show_cam.setChecked(True)
        self.btn_show_cam.setToolTip("Show camera-frame normals  (W)")
        self.btn_show_world = QPushButton("🌐 World")
        self.btn_show_world.setCheckable(True)
        self.btn_show_world.setToolTip("Show world-frame normals  (W)")
        for b in (self.btn_show_cam, self.btn_show_world):
            b.setStyleSheet(
                "QPushButton { padding:4px; } "
                "QPushButton:checked { background:#0d6efd; color:white; font-weight:bold; }")
        self.btn_show_cam.clicked.connect(lambda: self.set_normal_mode("camera"))
        self.btn_show_world.clicked.connect(lambda: self.set_normal_mode("world"))
        mode_row.addWidget(self.btn_show_cam)
        mode_row.addWidget(self.btn_show_world)
        sl.addLayout(mode_row)

        sl.addSpacing(6)
        self.btn_convert_world = QPushButton("🌐 Convert to World Frame")
        self.btn_convert_world.setStyleSheet(
            "background:#198754; color:white; font-weight:bold; padding:8px;")
        self.btn_convert_world.setToolTip(
            "Rotate camera-frame normals into world frame using COLMAP poses")
        self.btn_convert_world.clicked.connect(self._on_convert_clicked)
        sl.addWidget(self.btn_convert_world)

        sl.addSpacing(10)

        # run history --------------------------------------------------
        sl.addWidget(QLabel("<b>Run History</b>"))
        sel_row = QHBoxLayout()
        self.combo_runs = QComboBox()
        self.combo_runs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_runs.currentIndexChanged.connect(self._on_run_selected)
        sel_row.addWidget(self.combo_runs)

        self.btn_open_folder = QPushButton("📂")
        self.btn_open_folder.setToolTip("Open run folder in explorer")
        self.btn_open_folder.setFixedWidth(36)
        self.btn_open_folder.clicked.connect(self._open_run_folder)
        sel_row.addWidget(self.btn_open_folder)
        sl.addLayout(sel_row)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setFont(QFont("Consolas", 9))
        sl.addWidget(self.lbl_info)

        sl.addSpacing(10)

        # grid zoom ----------------------------------------------------
        sl.addWidget(QLabel("<b>Grid Zoom:</b>"))
        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(150, 600)
        self.zoom_slider.setValue(self._zoom_size)
        self.lbl_zoom_val = QLabel(f"{self._zoom_size}px")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.lbl_zoom_val)
        sl.addLayout(zoom_row)

        sl.addWidget(QLabel(
            "<small><i>Ctrl+Scroll on grid = zoom cards</i></small>"))

        sl.addStretch()
        grid_outer.addWidget(sidebar)

        # ── RIGHT scroll area ─────────────────────────────────────────
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_flow = FlowLayout(self.grid_container)
        self.grid_flow.setSpacing(8)
        self.grid_scroll.setWidget(self.grid_container)
        # Install event filter for Ctrl+Scroll zoom on the grid
        self.grid_scroll.viewport().installEventFilter(self)
        grid_outer.addWidget(self.grid_scroll, 1)

        self.stack.addWidget(self.page_grid)

    # ------------------------------------------------------------------
    def _build_detail_page(self):
        self.page_detail = QWidget()
        self.page_detail.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        det_lyt = QVBoxLayout(self.page_detail)

        # ── header: back / prev / title / next ────────────────────────
        det_header = QHBoxLayout()

        btn_back = QPushButton("◀ Back to Grid")
        btn_back.clicked.connect(self._close_detail)

        self.btn_view_prev = QPushButton("◁ Prev View")
        self.btn_view_prev.clicked.connect(self._detail_prev)
        self.btn_view_prev.setToolTip("Shift+←")

        self.lbl_detail_title = QLabel("<h2>View 00</h2>")
        self.lbl_detail_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_view_next = QPushButton("Next View ▷")
        self.btn_view_next.clicked.connect(self._detail_next)
        self.btn_view_next.setToolTip("Shift+→")

        det_header.addWidget(btn_back)
        det_header.addStretch()
        det_header.addWidget(self.btn_view_prev)
        det_header.addWidget(self.lbl_detail_title)
        det_header.addWidget(self.btn_view_next)
        det_header.addStretch()

        # camera / world toggle in detail header
        self.btn_det_cam   = QPushButton("📷 Camera")
        self.btn_det_cam.setCheckable(True)
        self.btn_det_cam.setChecked(True)
        self.btn_det_world = QPushButton("🌐 World")
        self.btn_det_world.setCheckable(True)
        for b in (self.btn_det_cam, self.btn_det_world):
            b.setStyleSheet(
                "QPushButton { padding:4px; } "
                "QPushButton:checked { background:#0d6efd; color:white; font-weight:bold; }")
        self.btn_det_cam.clicked.connect(
            lambda: self._set_detail_normal_mode("camera"))
        self.btn_det_world.clicked.connect(
            lambda: self._set_detail_normal_mode("world"))
        det_header.addWidget(self.btn_det_cam)
        det_header.addWidget(self.btn_det_world)

        det_lyt.addLayout(det_header)

        # ── zoomable image viewer ──────────────────────────────────────
        self.detail_player = ZoomPanImageView()
        det_lyt.addWidget(self.detail_player, 1)

        # ── frame info label (fixed line above slider) ─────────────────
        self.lbl_frame_info = QLabel("Frame 1 / 1")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setFixedHeight(22)
        det_lyt.addWidget(self.lbl_frame_info)

        # ── scrub bar (fixed height, no stretchy label) ────────────────
        nav = QHBoxLayout()
        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setFixedWidth(36)
        self.btn_prev_frame.setToolTip("← or ◀")
        self.btn_prev_frame.clicked.connect(lambda: self._scrub(-1))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)

        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setFixedWidth(36)
        self.btn_next_frame.setToolTip("→ or ▶")
        self.btn_next_frame.clicked.connect(lambda: self._scrub(1))

        nav.addWidget(self.btn_prev_frame)
        nav.addWidget(self.frame_slider, 1)   # slider takes all available space
        nav.addWidget(self.btn_next_frame)
        det_lyt.addLayout(nav)

        self.stack.addWidget(self.page_detail)

    # ------------------------------------------------------------------
    def _setup_shortcuts(self):
        """Keyboard shortcuts active inside this widget.
        Arrow keys / Shift+Arrows are handled by the main window.
        W toggles camera/world normal mode.
        Escape closes the detail page.
        """
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(
            lambda: self._close_detail() if self.stack.currentIndex() == 1 else None)

        QShortcut(QKeySequence(Qt.Key.Key_W), self).activated.connect(
            self._toggle_normal_mode)

    # ==================================================================
    # eventFilter – Ctrl+Scroll on the grid scroll area zooms cards
    # ==================================================================
    def eventFilter(self, obj, event):
        if (obj is self.grid_scroll.viewport()
                and event.type() == QEvent.Type.Wheel
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            delta = 20 if event.angleDelta().y() > 0 else -20
            new_val = max(150, min(600, self.zoom_slider.value() + delta))
            self.zoom_slider.setValue(new_val)
            return True
        return super().eventFilter(obj, event)

    # ==================================================================
    # button handlers
    # ==================================================================
    def _on_run_clicked(self):
        res_str = self.combo_res.currentText().replace(" (Default)", "")
        try:
            res_val = int(res_str)
        except ValueError:
            res_val = 960
        self.btn_run_lino.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_resume_run.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting …")
        self.start_lino.emit(res_val, self.spin_images.value())

    def _on_resume_run_clicked(self):
        """Resume an incomplete run — skip groups that already have normals."""
        run_dir = self.combo_runs.currentData()
        if not run_dir or not os.path.isdir(run_dir):
            return
        res_str = self.combo_res.currentText().replace(" (Default)", "")
        try:
            res_val = int(res_str)
        except ValueError:
            res_val = 960
        self.btn_run_lino.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_resume_run.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Resuming run …")
        self.resume_lino.emit(run_dir, res_val, self.spin_images.value())

    def _on_pause(self):
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.lbl_status.setText("Paused")
        self.request_pause.emit()

    def _on_resume(self):
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.lbl_status.setText("Resuming …")
        self.request_resume.emit()

    def _on_cancel(self):
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling …")
        self.request_cancel.emit()

    def _on_zoom_changed(self, val):
        self._zoom_size = val
        self.lbl_zoom_val.setText(f"{val}px")
        for i in range(self.grid_flow.count()):
            item = self.grid_flow.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), NormalCard):
                item.widget().set_zoom(val)

    # ==================================================================
    # public API  (called from app.py)
    # ==================================================================
    def set_progress(self, text, value):
        self.lbl_status.setText(text)
        self.progress_bar.setValue(value)

    def reset_ui(self):
        self.btn_run_lino.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(0)
        # Re-check completion so Resume button appears after a cancel
        if self._current_run_dir and os.path.isdir(self._current_run_dir):
            self._load_run_params(self._current_run_dir)
        else:
            self.lbl_status.setText("Ready")

    def set_active_run_dir(self, run_dir):
        self._current_run_dir = run_dir

    def update_card_for_group(self, group_idx, exr_path):
        """Live-update a single card after inference produces its normal."""
        for i in range(self.grid_flow.count()):
            item = self.grid_flow.itemAt(i)
            if item and item.widget():
                card = item.widget()
                if isinstance(card, NormalCard) and card.group_idx == group_idx:
                    nml_rgb = _normal_to_rgb(exr_path)
                    if nml_rgb is not None:
                        pm = _np_to_pixmap(nml_rgb)
                        card.set_content(pm, True, self._zoom_size)
                    break

    def update_card_world_normal(self, group_idx, world_path):
        """Live-update a card's world-frame pixmap after conversion finishes."""
        for i in range(self.grid_flow.count()):
            item = self.grid_flow.itemAt(i)
            if item and item.widget():
                card = item.widget()
                if isinstance(card, NormalCard) and card.group_idx == group_idx:
                    nml_rgb = _normal_to_rgb(world_path)
                    if nml_rgb is not None:
                        card.set_world_content(_np_to_pixmap(nml_rgb))
                    break

    # Normal display mode (camera / world) ----------------------------
    def set_normal_mode(self, mode):
        """Switch all grid cards and detail toggle to camera or world mode."""
        self._normal_mode = mode
        self.btn_show_cam.setChecked(mode == "camera")
        self.btn_show_world.setChecked(mode == "world")
        for i in range(self.grid_flow.count()):
            item = self.grid_flow.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), NormalCard):
                item.widget().set_display_mode(mode)
        # Sync detail toggle if open
        self.btn_det_cam.setChecked(mode == "camera")
        self.btn_det_world.setChecked(mode == "world")

    def _toggle_normal_mode(self):
        self.set_normal_mode("world" if self._normal_mode == "camera" else "camera")

    def _set_detail_normal_mode(self, mode):
        """Switch only the detail-page view and sync sidebar toggle."""
        self._normal_mode = mode
        self.btn_det_cam.setChecked(mode == "camera")
        self.btn_det_world.setChecked(mode == "world")
        self.btn_show_cam.setChecked(mode == "camera")
        self.btn_show_world.setChecked(mode == "world")
        # Jump to the matching frame in the carousel
        for i, (lbl, _) in enumerate(self._detail_frames):
            if mode == "world" and "World" in lbl:
                self.frame_slider.setValue(i)
                return
            if mode == "camera" and "Camera" in lbl and "Normal" in lbl:
                self.frame_slider.setValue(i)
                return

    def _on_convert_clicked(self):
        if not self._current_run_dir:
            return
        self.convert_normals.emit(self._current_run_dir)


    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_normal(outputs_dir, group_idx):
        """Return the path to the camera-frame normal map, or None."""
        for ext in (".exr", ".npy"):
            p = os.path.join(outputs_dir, f"{group_idx:02d}_normal{ext}")
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _find_normal_world(outputs_dir, group_idx):
        """Return the path to the world-frame normal map, or None."""
        for ext in (".exr", ".npy"):
            p = os.path.join(outputs_dir, f"{group_idx:02d}_normal_world{ext}")
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # run selector / population
    # ------------------------------------------------------------------
    def populate_runs(self, workspace_dir):
        self.combo_runs.blockSignals(True)
        self.combo_runs.clear()
        self._workspace_dir = workspace_dir

        runs_root = os.path.join(workspace_dir, "lino_runs")
        if not os.path.isdir(runs_root):
            self.combo_runs.blockSignals(False)
            return

        runs = [d for d in os.listdir(runs_root)
                if os.path.isdir(os.path.join(runs_root, d))]
        runs.sort(key=lambda x: os.path.getmtime(
            os.path.join(runs_root, x)), reverse=True)

        for r in runs:
            self.combo_runs.addItem(r, os.path.join(runs_root, r))

        self.combo_runs.blockSignals(False)

        if runs:
            self.combo_runs.setCurrentIndex(0)
            self._on_run_selected(0)

    def _on_run_selected(self, index):
        run_dir = self.combo_runs.itemData(index)
        if not run_dir or not os.path.isdir(run_dir):
            return
        self._current_run_dir = run_dir
        self._load_run_grid(run_dir)
        self._load_run_params(run_dir)

    def _load_run_params(self, run_dir):
        """Read run_config.json and restore UI controls + show resume option."""
        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            return
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return

        # ── Restore resolution combo ───────────────────────────────────
        res = cfg.get("resolution", 960)
        res_options = [512, 768, 960, 1024, 1280, 1536, 2048]
        best = min(range(len(res_options)), key=lambda i: abs(res_options[i] - res))
        self.combo_res.blockSignals(True)
        self.combo_res.setCurrentIndex(best)
        self.combo_res.blockSignals(False)

        # ── Restore max images ─────────────────────────────────────────
        mi = cfg.get("max_images", 8)
        self.spin_images.blockSignals(True)
        self.spin_images.setValue(mi)
        self.spin_images.blockSignals(False)

        # ── Completion status ──────────────────────────────────────────
        groups = cfg.get("groups", [])
        outputs_dir = os.path.join(run_dir, "outputs")
        done = sum(1 for g in groups
                   if self._find_normal(outputs_dir, g["group_idx"]) is not None)
        total = len(groups)

        if done < total:
            self.lbl_status.setText(
                f"Incomplete: {done}/{total} groups done — click Resume to continue")
            self.btn_resume_run.setVisible(True)
        else:
            self.lbl_status.setText(f"Complete: {done}/{total} groups")
            self.btn_resume_run.setVisible(False)


    def _open_run_folder(self):
        run_dir = self.combo_runs.currentData()
        if run_dir and os.path.isdir(run_dir):
            if os.name == "nt":
                os.startfile(run_dir)
            else:
                import subprocess
                subprocess.run(["xdg-open", run_dir])

    # ------------------------------------------------------------------
    # grid rendering
    # ------------------------------------------------------------------
    def _clear_grid(self):
        while self.grid_flow.count() > 0:
            item = self.grid_flow.takeAt(0)
            if item and item.widget():
                item.widget().hide()
                item.widget().deleteLater()

    def _load_run_grid(self, run_dir):
        """Build (or rebuild) the card grid for the selected run."""
        self._clear_grid()

        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            self.lbl_info.setText("run_config.json not found.")
            return

        with open(cfg_path) as f:
            self._run_cfg = json.load(f)

        groups = self._run_cfg.get("groups", [])
        n_groups = len(groups)
        res = self._run_cfg.get("resolution", "?")
        mi = self._run_cfg.get("max_images", "?")
        self.lbl_info.setText(
            f"Resolution: {res}px  ·  Max imgs: {mi}  ·  Views: {n_groups}")

        inputs_dir = os.path.join(run_dir, "inputs")
        outputs_dir = os.path.join(run_dir, "outputs")

        for g in groups:
            idx = g["group_idx"]
            view_dir = os.path.join(inputs_dir, f"view_{idx:02d}.data")

            normal_path = self._find_normal(outputs_dir, idx)

            if normal_path:
                nml_rgb = _normal_to_rgb(normal_path)
                pm  = _np_to_pixmap(nml_rgb) if nml_rgb is not None else QPixmap()
                has = True
            else:
                first = os.path.join(view_dir, "L00.jpg")
                if os.path.exists(first):
                    bgr = cv2.imread(first)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pm  = _np_to_pixmap(rgb)
                else:
                    pm = QPixmap()
                has = False

            card = NormalCard(idx, parent=self.grid_container)
            card.set_content(pm, has, self._zoom_size)
            card.set_display_mode(self._normal_mode)

            # Load world-frame pixmap if available
            world_path = self._find_normal_world(outputs_dir, idx)
            if world_path:
                w_rgb = _normal_to_rgb(world_path)
                if w_rgb is not None:
                    card.set_world_content(_np_to_pixmap(w_rgb))

            card.clicked.connect(self._open_detail)
            self.grid_flow.addWidget(card)

    @staticmethod
    def _find_normal(outputs_dir, idx):
        for ext in (".exr", ".npy"):
            p = os.path.join(outputs_dir, f"{idx:02d}_normal{ext}")
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # detail page
    # ------------------------------------------------------------------
    def _open_detail(self, group_idx):
        if not self._current_run_dir:
            return
        self._active_view_idx = group_idx
        self._populate_detail(group_idx)
        self.stack.setCurrentIndex(1)
        self.page_detail.setFocus()

    def _close_detail(self):
        self._active_view_idx = -1
        self.stack.setCurrentIndex(0)

    def _detail_prev(self):
        if self._active_view_idx > 0:
            self._open_detail(self._active_view_idx - 1)

    def _detail_next(self):
        if self._run_cfg:
            n = len(self._run_cfg.get("groups", []))
            if self._active_view_idx < n - 1:
                self._open_detail(self._active_view_idx + 1)

    def _populate_detail(self, group_idx):
        """Load all carousel frames for a given view into the detail page."""
        run = self._current_run_dir
        view_dir = os.path.join(run, "inputs", f"view_{group_idx:02d}.data")
        outputs_dir = os.path.join(run, "outputs")

        self.lbl_detail_title.setText(f"<h2>View {group_idx:02d}</h2>")

        n_groups = len(self._run_cfg.get("groups", [])) if self._run_cfg else 0
        self.btn_view_prev.setVisible(group_idx > 0)
        self.btn_view_next.setVisible(group_idx < n_groups - 1)

        # Build frame list: input images → mask → normal map
        self._detail_frames = []

        if os.path.isdir(view_dir):
            img_files = sorted(
                f for f in os.listdir(view_dir)
                if f.startswith("L") and f.endswith(".jpg"))
            for fname in img_files:
                bgr = cv2.imread(os.path.join(view_dir, fname))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self._detail_frames.append(
                    (f"Input: {fname}", _np_to_pixmap(rgb)))

            mask_path = os.path.join(view_dir, "mask.png")
            if os.path.exists(mask_path):
                mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_gray is not None:
                    self._detail_frames.append(
                        ("Mask", _np_to_pixmap(mask_gray)))

        normal_path = self._find_normal(outputs_dir, group_idx)
        world_path  = self._find_normal_world(outputs_dir, group_idx)

        if normal_path:
            nml_rgb = _normal_to_rgb(normal_path)
            if nml_rgb is not None:
                self._detail_frames.append(
                    ("📷 Normal (Camera Frame)", _np_to_pixmap(nml_rgb)))

        if world_path:
            w_rgb = _normal_to_rgb(world_path)
            if w_rgb is not None:
                self._detail_frames.append(
                    ("🌐 Normal (World Frame)", _np_to_pixmap(w_rgb)))

        # Configure slider
        n = max(len(self._detail_frames), 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        # Default: world-frame normal if present, else camera-frame, else first frame
        if world_path:
            start = next((i for i, (lbl, _) in enumerate(self._detail_frames)
                          if "World" in lbl), n - 1)
            self._normal_mode = "world"
            self.btn_det_world.setChecked(True)
            self.btn_det_cam.setChecked(False)
        elif normal_path:
            start = next((i for i, (lbl, _) in enumerate(self._detail_frames)
                          if "Camera" in lbl), n - 1)
            self._normal_mode = "camera"
            self.btn_det_cam.setChecked(True)
            self.btn_det_world.setChecked(False)
        else:
            start = 0
        self.frame_slider.setValue(start)
        self.frame_slider.blockSignals(False)
        self._show_frame(start)

    def _on_frame_slider(self, val):
        self._show_frame(val)

    def _scrub(self, delta):
        if self.stack.currentIndex() != 1:
            return
        new_val = self.frame_slider.value() + delta
        new_val = max(0, min(new_val, self.frame_slider.maximum()))
        self.frame_slider.setValue(new_val)

    def _show_frame(self, idx):
        if not self._detail_frames:
            self.detail_player.setText("No frames")
            self.lbl_frame_info.setText("—")
            return
        idx = max(0, min(idx, len(self._detail_frames) - 1))
        label, pixmap = self._detail_frames[idx]
        self.detail_player.setPixmap(pixmap)
        self.lbl_frame_info.setText(
            f"{label}    ({idx + 1} / {len(self._detail_frames)})")
