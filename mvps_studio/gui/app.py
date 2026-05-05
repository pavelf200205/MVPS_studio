import sys
import os
import re
import numpy as np
import cv2
import time
import concurrent.futures
import tempfile
import hashlib
import shutil
import json
import scipy.ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device='cuda').view(1, 1, 3, 3)
        SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device='cuda').view(1, 1, 3, 3)
        WHITE_COLOR = torch.tensor([255, 255, 255], device='cuda', dtype=torch.uint8)
except ImportError:
    HAS_TORCH = False

try:
    from transparent_background import Remover
    from PIL import Image as PILImage
    HAS_INSPYRENET = True
except ImportError:
    HAS_INSPYRENET = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QComboBox, QFileDialog, QCheckBox, QSlider, QScrollArea,
    QProgressDialog, QLayout, QSizePolicy, QStackedWidget, QSplitter, QFrame,
    QGridLayout, QTabWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMenu, QToolBar, QListWidget, QMessageBox
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QPoint, QRect, QTimer, QEvent, QPointF, QRectF
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QDragEnterEvent, QDropEvent, 
    QImageReader, QColor, QPen, QCursor, QKeySequence, 
    QShortcut, QPolygonF, QWheelEvent, QMouseEvent
)


from mvps_studio.core.utils import ProfileTimer, load_image_fast, numpy_to_qpixmap, natsort_key
from mvps_studio.core.magnetic_scissors import MagneticScissorsBackend
from mvps_studio.workers.dataset_loader import DatasetLoaderThread
from mvps_studio.workers.max_image import MaxImageWorker
from mvps_studio.workers.inspyrenet import InSPyReNetWorker
from mvps_studio.workers.diagnostic import DiagnosticWorker
from mvps_studio.gui.widgets.zoom_pan_view import ZoomPanView
from mvps_studio.gui.widgets.zoom_pan_image_view import ZoomPanImageView
from mvps_studio.gui.widgets.flow_layout import FlowLayout
from mvps_studio.gui.widgets.scaled_image_label import ScaledImageLabel
from mvps_studio.gui.widgets.settings_panel import SettingsPanel
from mvps_studio.gui.widgets.hybrid_group_card import HybridGroupCard
from mvps_studio.gui.scenes.masking_scene import MaskingScene, StateManager
from mvps_studio.gui.scenes.sfm_scene import SfmScene
from mvps_studio.gui.scenes.lino_scene import LinoScene
from mvps_studio.gui.scenes.normalization_scene import NormalizationScene
from mvps_studio.gui.scenes.supernormal_scene import SuperNormalScene
from mvps_studio.workers.colmap_worker import ColmapWorker
from mvps_studio.modules.lino.lino_worker import LinoWorker

try:
    from mvps_studio.core.magnetic_scissors import MagneticScissorsBackend
except ImportError: pass
try:
    from mvps_studio.workers.dataset_loader import DatasetLoaderThread
except ImportError: pass
try:
    from mvps_studio.workers.max_image import MaxImageWorker
except ImportError: pass
try:
    from mvps_studio.workers.inspyrenet import InSPyReNetWorker
except ImportError: pass
try:
    from mvps_studio.workers.diagnostic import DiagnosticWorker
except ImportError: pass
try:
    from mvps_studio.gui.widgets.zoom_pan_view import ZoomPanView
except ImportError: pass
try:
    from mvps_studio.gui.widgets.flow_layout import FlowLayout
except ImportError: pass
try:
    from mvps_studio.gui.widgets.scaled_image_label import ScaledImageLabel
except ImportError: pass
try:
    from mvps_studio.gui.widgets.settings_panel import SettingsPanel
except ImportError: pass
try:
    from mvps_studio.gui.widgets.hybrid_group_card import HybridGroupCard
except ImportError: pass
try:
    from mvps_studio.gui.scenes.masking_scene import MaskingScene, StateManager
except ImportError: pass
try:
    from mvps_studio.modules.lino.convert_worker import WorldFrameConverter
except ImportError: pass



class MVPSStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MVPS Studio")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.image_paths = []
        self.suppressed_paths = set()
        self.groups = []
        
        self.cache_dir_obj = tempfile.TemporaryDirectory(prefix="mvps_hybrid_cache_")
        self.cache_dir = self.cache_dir_obj.name
        self.workspace_dir = self.cache_dir
        self.input_imgs_dir = self.cache_dir
        self.max_imgs_dir = self.cache_dir
        self.masks_dir = self.cache_dir
        self.colmap_dir = self.cache_dir
        
        self.image_cache = {}
        self.global_mags_cache = {}
        self.edge_map_cache = {}       # grouping tab edge-diagnostic images
        self.mask_overlay_cache = {}   # masking tab red-tint previews
        self.dirty_groups = set()
        self.group_overrides = {}
        
        self.max_images_paths = {}    
        self.soft_masks_paths = {}     
        self.manual_edits_paths = {}   
        
        self.preview_maxes = {}        
        self.preview_soft_masks = {}   
        self.preview_hard_masks = {}   
        self.preview_manual_edits = {} 
        self.mask_thresholds = {}      
        self.global_mask_threshold = 0.5
        
        self.view_mode = "animation" 
        self.left_split = None 
        
        self.worker = None
        self.active_detail_group = -1
        self.active_frame_idx = 0
        
        self.global_tick = 0
        self.is_playing = True
        
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.tick_animation)
        self.magnetic_backend = MagneticScissorsBackend()
        
        self._setup_ui()
        self._setup_shortcuts()

    def closeEvent(self, event):
        self.cache_dir_obj.cleanup()
        super().closeEvent(event)
        
    def cancel_worker(self):
        if self.worker and self.worker.isRunning(): 
            self.worker.is_interrupted = True

    def _setup_shortcuts(self):
        # Masking tab: lasso operations
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.on_shortcut_add)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.on_shortcut_sub)
        QShortcut(QKeySequence("Esc"),    self).activated.connect(self.on_shortcut_esc)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.on_shortcut_undo)

        # Space — toggle animation (grouping) / toggle mask (masking)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.on_shortcut_space)

        # Tab — toggle animation ↔ edge map in grouping tab
        QShortcut(QKeySequence(Qt.Key.Key_Tab), self).activated.connect(self._shortcut_toggle_mode)

        # Left / Right → scrub frame in grouping carousel
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self).activated.connect(self._shortcut_frame_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._shortcut_frame_next)
        # Shift+Left / Shift+Right → previous / next angle
        QShortcut(QKeySequence("Shift+Left"),  self).activated.connect(self._shortcut_group_prev)
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(self._shortcut_group_next)

    def on_shortcut_add(self):
        if self.main_tabs.currentIndex() == 1 and self.mask_stack.currentIndex() == 1:
            self.mask_scene.apply_boolean("add")

    def on_shortcut_sub(self):
        if self.main_tabs.currentIndex() == 1 and self.mask_stack.currentIndex() == 1:
            self.mask_scene.apply_boolean("subtract")

    def on_shortcut_esc(self):
        tab = self.main_tabs.currentIndex()
        if tab == 1 and self.mask_stack.currentIndex() == 1:
            self.mask_scene.clear_state()
        elif tab == 0 and self.stack.currentIndex() == 1:
            self.close_detail_view()

    def on_shortcut_undo(self):
        if self.main_tabs.currentIndex() == 1 and self.mask_stack.currentIndex() == 1:
            self.mask_scene.state_manager.undo()

    def on_shortcut_space(self):
        tab = self.main_tabs.currentIndex()
        if tab == 0:                                              # Grouping tab
            self.toggle_global_playback()
        elif tab == 1 and self.mask_stack.currentIndex() == 1:   # Masking detail
            self.mask_scene.toggle_mask()

    def _shortcut_toggle_mode(self):
        """Tab → toggle animation ↔ edge map in the grouping tab."""
        if self.main_tabs.currentIndex() == 0:
            new_mode = "edge" if self.view_mode == "animation" else "animation"
            self.set_view_mode(new_mode)

    # ---- Grouping / Masking / LINO arrow shortcuts ----
    def _shortcut_frame_prev(self):
        tab = self.main_tabs.currentIndex()
        if tab == 0 and self.stack.currentIndex() == 1:
            self.scrub_frame(-1)
        elif tab == 3:                            # LINO tab
            self.lino_scene._scrub(-1)

    def _shortcut_frame_next(self):
        tab = self.main_tabs.currentIndex()
        if tab == 0 and self.stack.currentIndex() == 1:
            self.scrub_frame(1)
        elif tab == 3:                            # LINO tab
            self.lino_scene._scrub(1)

    def _shortcut_group_prev(self):
        tab = self.main_tabs.currentIndex()
        if tab == 0 and self.stack.currentIndex() == 1:
            self.view_prev_group()
        elif tab == 1 and self.mask_stack.currentIndex() == 1:
            self.view_mask_prev()
        elif tab == 3:                            # LINO tab
            self.lino_scene._detail_prev()

    def _shortcut_group_next(self):
        tab = self.main_tabs.currentIndex()
        if tab == 0 and self.stack.currentIndex() == 1:
            self.view_next_group()
        elif tab == 1 and self.mask_stack.currentIndex() == 1:
            self.view_mask_next()
        elif tab == 3:                            # LINO tab
            self.lino_scene._detail_next()

    def on_journal_clicked(self, item):
        row = self.history_list.row(item)
        if self.mask_scene.state_manager.goto_index(row):
            self.on_mask_edited(self.active_detail_group, self.mask_scene.manual_mask_np)

    def _setup_ui(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        toolbar.addAction("💾 Save Project", self.force_save_project)
        
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        
        self.tab_grouping = QWidget()
        self._setup_grouping_tab()
        self.main_tabs.addTab(self.tab_grouping, "1. Grouping & Alignment")
        
        self.tab_masking = QWidget()
        self._setup_masking_tab()
        self.main_tabs.addTab(self.tab_masking, "2. Masking & Background Removal")
        self.main_tabs.setTabEnabled(1, False)
        
        self.tab_sfm = QWidget()
        sfm_layout = QVBoxLayout(self.tab_sfm)
        sfm_layout.setContentsMargins(0, 0, 0, 0)
        self.sfm_scene = SfmScene(self.tab_sfm)
        self.sfm_scene.start_reconstruction.connect(self.run_colmap_sfm)
        sfm_layout.addWidget(self.sfm_scene)
        self.main_tabs.addTab(self.tab_sfm, "3. SfM Reconstruction")
        self.main_tabs.setTabEnabled(2, False)

        self.tab_lino = QWidget()
        lino_layout = QVBoxLayout(self.tab_lino)
        lino_layout.setContentsMargins(0, 0, 0, 0)
        self.lino_scene = LinoScene(self.tab_lino)
        self.lino_scene.start_lino.connect(self.run_lino_inference)
        self.lino_scene.resume_lino.connect(self._on_lino_resume)
        self.lino_scene.convert_normals.connect(self._on_convert_to_world_frame)
        self.lino_scene.request_pause.connect(self._lino_pause)
        self.lino_scene.request_resume.connect(self._lino_resume)
        self.lino_scene.request_cancel.connect(self._lino_cancel)
        lino_layout.addWidget(self.lino_scene)
        self.main_tabs.addTab(self.tab_lino, "4. Normal Extraction (LINO)")
        self.main_tabs.setTabEnabled(3, False)
        
        self.tab_normalization = QWidget()
        norm_layout = QVBoxLayout(self.tab_normalization)
        norm_layout.setContentsMargins(0, 0, 0, 0)
        self.norm_scene = NormalizationScene(self, self.tab_normalization)
        norm_layout.addWidget(self.norm_scene)
        self.main_tabs.addTab(self.tab_normalization, "5. Camera Normalization")
        self.main_tabs.setTabEnabled(4, False)

        self.tab_supernormal = QWidget()
        sn_layout = QVBoxLayout(self.tab_supernormal)
        sn_layout.setContentsMargins(0, 0, 0, 0)
        self.sn_scene = SuperNormalScene(self, self.tab_supernormal)
        sn_layout.addWidget(self.sn_scene)
        self.main_tabs.addTab(self.tab_supernormal, "6. SuperNormal Reconstruction")
        self.main_tabs.setTabEnabled(5, False)
        
        self.main_tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        if index == 3: # LINO Tab
            if hasattr(self, 'workspace_dir') and self.workspace_dir:
                self.lino_scene.populate_runs(self.workspace_dir)

    # -----------------------------------------------------
    # TAB 1: GROUPING & ALIGNMENT
    # -----------------------------------------------------
    def _setup_grouping_tab(self):
        main_layout = QHBoxLayout(self.tab_grouping)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)
        
        btn_new_proj = QPushButton("✨ New Project (Create Workspace)")
        btn_new_proj.clicked.connect(self.create_new_project)
        btn_load_proj = QPushButton("📂 Open Existing Workspace")
        btn_load_proj.clicked.connect(self.load_project)
        side_layout.addWidget(btn_new_proj)
        side_layout.addWidget(btn_load_proj)
        
        self.lbl_loaded_count = QLabel("<b>Loaded: 0 images</b>")
        self.lbl_loaded_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.lbl_loaded_count)
        
        if HAS_TORCH:
            lbl_gpu = QLabel("<span style='color:green;'><b>⚡ GPU Acceleration Active</b></span>")
            lbl_gpu.setAlignment(Qt.AlignmentFlag.AlignCenter)
            side_layout.addWidget(lbl_gpu)
            
        mode_box = QFrame()
        mode_box.setStyleSheet("QFrame { background-color: #e9ecef; border-radius: 8px; }")
        mode_lyt = QVBoxLayout(mode_box)
        mode_btns = QHBoxLayout()
        
        self.btn_mode_anim = QPushButton("🎬 Animation")
        self.btn_mode_anim.setCheckable(True)
        self.btn_mode_anim.setChecked(True)
        self.btn_mode_anim.clicked.connect(lambda: self.set_view_mode("animation"))
        
        self.btn_mode_edge = QPushButton("🌈 Edge Map")
        self.btn_mode_edge.setCheckable(True)
        self.btn_mode_edge.clicked.connect(lambda: self.set_view_mode("edge"))
        
        self.btn_mode_anim.setStyleSheet("QPushButton:checked { background-color: #0d6efd; color: white; font-weight: bold; }")
        self.btn_mode_edge.setStyleSheet("QPushButton:checked { background-color: #6f42c1; color: white; font-weight: bold; }")
        
        mode_btns.addWidget(self.btn_mode_anim)
        mode_btns.addWidget(self.btn_mode_edge)
        mode_lyt.addLayout(mode_btns)
        side_layout.addWidget(mode_box)
        
        side_layout.addWidget(QLabel("<b>Dataset Grouping</b>"))
        box_batch = QHBoxLayout()
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(2, 200)
        self.spin_batch.setValue(10)
        box_batch.addWidget(self.spin_batch)
        
        self.btn_apply_batch = QPushButton("Regroup")
        self.btn_apply_batch.clicked.connect(self.hard_regroup)
        box_batch.addWidget(self.btn_apply_batch)
        side_layout.addLayout(box_batch)
        
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Filename (Natural)", "Date Modified"])
        side_layout.addWidget(QLabel("Sort Method:"))
        side_layout.addWidget(self.combo_sort)
        
        self.combo_res = QComboBox()
        self.combo_res.addItems(["400px (Fastest)", "600px (Balanced)", "800px (High Quality)", "Native"])
        self.combo_res.setCurrentIndex(1)
        self.combo_res.currentIndexChanged.connect(self.on_resolution_changed)
        side_layout.addWidget(QLabel("Processing Resolution:"))
        side_layout.addWidget(self.combo_res)
        
        self.chk_disk_cache = QCheckBox("Use Disk Cache (Saves RAM)")
        self.chk_disk_cache.setChecked(False)
        self.chk_disk_cache.toggled.connect(self.on_resolution_changed)
        side_layout.addWidget(self.chk_disk_cache)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(150, 600)
        self.zoom_slider.setValue(260)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        lbl_zoom_val = QLabel("260px")
        self.zoom_slider.valueChanged.connect(lambda v: lbl_zoom_val.setText(f"{v}px"))
        
        box_zoom = QHBoxLayout()
        box_zoom.addWidget(self.zoom_slider)
        box_zoom.addWidget(lbl_zoom_val)
        side_layout.addWidget(QLabel("Grid Zoom:"))
        side_layout.addLayout(box_zoom)
        
        self.controls_stack = QStackedWidget()
        
        self.anim_controls = QWidget()
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(2, 60)
        self.fps_slider.setValue(12)
        self.lbl_fps = QLabel("12 FPS")
        self.fps_slider.valueChanged.connect(self.on_fps_changed)
        
        box_fps = QHBoxLayout()
        box_fps.addWidget(self.fps_slider)
        box_fps.addWidget(self.lbl_fps)
        
        self.btn_global_play = QPushButton("⏸ Pause Animation")
        self.btn_global_play.clicked.connect(self.toggle_global_playback)
        
        anim_lyt = QVBoxLayout(self.anim_controls)
        anim_lyt.addWidget(QLabel("Playback Speed:"))
        anim_lyt.addLayout(box_fps)
        anim_lyt.addWidget(self.btn_global_play)
        anim_lyt.addStretch()
        self.controls_stack.addWidget(self.anim_controls)
        
        self.edge_controls = QWidget()
        self.global_settings = SettingsPanel("Global Edge Settings")
        self.global_settings.settings_changed.connect(self.on_global_settings_changed)
        
        self.lbl_recalc_warning = QLabel("⚠️ Update Needed!")
        self.lbl_recalc_warning.setStyleSheet("color: #dc3545; font-weight: bold;")
        self.lbl_recalc_warning.setVisible(False)
        
        self.btn_run_alignment = QPushButton("🔍 Run Alignment Check")
        self.btn_run_alignment.clicked.connect(self.run_alignment_check)
        
        edge_lyt = QVBoxLayout(self.edge_controls)
        edge_lyt.addWidget(self.global_settings)
        edge_lyt.addWidget(self.lbl_recalc_warning)
        edge_lyt.addWidget(self.btn_run_alignment)
        edge_lyt.addStretch()
        
        self.controls_stack.addWidget(self.edge_controls)
        side_layout.addWidget(self.controls_stack)
        
        self.btn_proceed = QPushButton("Confirm Grouping & Proceed ➔")
        self.btn_proceed.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 12px;")
        self.btn_proceed.clicked.connect(self.proceed_to_masking)
        
        side_layout.addStretch()
        side_layout.addWidget(self.btn_proceed)
        main_layout.addWidget(sidebar)
        
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)
        
        self.page_grid = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.flow_container = QWidget()
        self.flow_layout = FlowLayout(self.flow_container)
        self.scroll_area.setWidget(self.flow_container)
        
        grid_lyt = QVBoxLayout(self.page_grid)
        grid_lyt.addWidget(self.scroll_area)
        self.stack.addWidget(self.page_grid)
        
        self.page_detail = QWidget()
        det_lyt = QVBoxLayout(self.page_detail)
        det_header = QHBoxLayout()
        
        btn_back_grid = QPushButton("◀ Back to Grid")
        btn_back_grid.clicked.connect(self.close_detail_view)
        
        self.btn_view_prev = QPushButton("◁ Prev Angle")
        self.btn_view_prev.clicked.connect(self.view_prev_group)
        
        self.lbl_detail_title = QLabel("<h2>Angle 01</h2>")
        self.lbl_detail_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_view_next = QPushButton("Next Angle ▷")
        self.btn_view_next.clicked.connect(self.view_next_group)
        
        det_header.addWidget(btn_back_grid)
        det_header.addStretch()
        det_header.addWidget(self.btn_view_prev)
        det_header.addWidget(self.lbl_detail_title)
        det_header.addWidget(self.btn_view_next)
        det_header.addStretch()
        det_lyt.addLayout(det_header)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_split = QWidget()
        ls_layout = QVBoxLayout(self.left_split)
        
        self.local_settings = SettingsPanel("Local Override", is_override=True)
        self.local_settings.settings_changed.connect(self.on_local_settings_changed)
        ls_layout.addWidget(self.local_settings)
        ls_layout.addStretch()
        
        splitter.addWidget(self.left_split)
        self.left_split.hide()

        right_split = QWidget()
        rs_layout = QVBoxLayout(right_split)
        
        self.detail_player = ZoomPanImageView()
        rs_layout.addWidget(self.detail_player, 1)

        # frame info label on its own fixed-height line
        self.lbl_frame_info = QLabel("Frame 1 / 10")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setFixedHeight(22)
        rs_layout.addWidget(self.lbl_frame_info)

        self.nav_layout_widget = QWidget()
        nav_lyt = QHBoxLayout(self.nav_layout_widget)
        
        self.btn_detail_play = QPushButton("⏸ Pause")
        self.btn_detail_play.clicked.connect(self.toggle_global_playback)
        
        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setFixedWidth(36)
        self.btn_prev_frame.clicked.connect(lambda: self.scrub_frame(-1))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setFixedWidth(36)
        self.btn_next_frame.clicked.connect(lambda: self.scrub_frame(1))

        nav_lyt.addWidget(self.btn_detail_play)
        nav_lyt.addWidget(self.btn_prev_frame)
        nav_lyt.addWidget(self.frame_slider, 1)
        nav_lyt.addWidget(self.btn_next_frame)

        rs_layout.addWidget(self.nav_layout_widget)
        
        act_lyt = QGridLayout()
        self.btn_cascade_left = QPushButton("◀◀ Shift Left (Cascade)")
        self.btn_cascade_left.clicked.connect(lambda: self.move_cascade(-1))
        
        self.btn_local_left = QPushButton("◀ Move Prev (Local)")
        self.btn_local_left.clicked.connect(lambda: self.move_local(-1))
        
        self.btn_suppress_img = QPushButton("🚫 Suppress Current Frame")
        self.btn_suppress_img.clicked.connect(self.suppress_image)
        
        self.btn_local_right = QPushButton("Move Next (Local) ▶")
        self.btn_local_right.clicked.connect(lambda: self.move_local(1))
        
        self.btn_cascade_right = QPushButton("Shift Right (Cascade) ▶▶")
        self.btn_cascade_right.clicked.connect(lambda: self.move_cascade(1))
        
        act_lyt.addWidget(self.btn_cascade_left, 0, 0)
        act_lyt.addWidget(self.btn_local_left, 0, 1)
        act_lyt.addWidget(self.btn_suppress_img, 0, 2)
        act_lyt.addWidget(self.btn_local_right, 0, 3)
        act_lyt.addWidget(self.btn_cascade_right, 0, 4)
        rs_layout.addLayout(act_lyt)
        
        splitter.addWidget(right_split)
        splitter.setSizes([300, 700])
        det_lyt.addWidget(splitter, 1)

        self.stack.addWidget(self.page_detail)
        self.scroll_area.viewport().installEventFilter(self)
        self.on_fps_changed(12)
        self.set_view_mode("animation")

    # -----------------------------------------------------
    # TAB 2: MASKING & BACKGROUND REMOVAL
    # -----------------------------------------------------
    def _setup_masking_tab(self):
        mask_layout = QHBoxLayout(self.tab_masking)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mask_stack = QStackedWidget()
        mask_layout.addWidget(self.mask_stack, 1)

        self.mask_page_grid = QWidget()
        mgrid_main_lyt = QHBoxLayout(self.mask_page_grid)
        mgrid_main_lyt.setContentsMargins(0, 0, 0, 0)

        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        side_lyt = QVBoxLayout(sidebar)
        
        self.btn_run_inspyrenet = QPushButton("✨ Generate Initial Masks (InSPyReNet)")
        self.btn_run_inspyrenet.clicked.connect(self.run_inspyrenet)
        self.btn_run_inspyrenet.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold; padding: 12px;")
        
        if not HAS_INSPYRENET: 
            self.btn_run_inspyrenet.setEnabled(False)
            self.btn_run_inspyrenet.setText("⚠️ Please install transparent-background")
            
        side_lyt.addWidget(self.btn_run_inspyrenet)
        
        self.masking_res_combo = QComboBox()
        self.masking_res_combo.addItems(["1024px (Fast)", "2048px (Balanced)", "4096px (High Qual)", "Native (Original)"])
        self.masking_res_combo.setCurrentIndex(1)
        side_lyt.addWidget(QLabel("Inference Resolution:"))
        side_lyt.addWidget(self.masking_res_combo)
        
        self.mask_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_thresh_slider.setRange(0, 100)
        self.mask_thresh_slider.setValue(50)
        self.lbl_mask_thresh = QLabel("0.50")
        
        def on_global_thresh_change(v):
            self.lbl_mask_thresh.setText(f"{v/100.0:.2f}")
            self.global_mask_threshold = v/100.0
            self.update_all_hard_masks()
            
        self.mask_thresh_slider.valueChanged.connect(on_global_thresh_change)
        
        box_thresh = QHBoxLayout()
        box_thresh.addWidget(self.mask_thresh_slider)
        box_thresh.addWidget(self.lbl_mask_thresh)
        side_lyt.addWidget(QLabel("<b>Global Threshold</b> (Soft to Hard Mask):"))
        side_lyt.addLayout(box_thresh)
        
        self.mask_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_zoom_slider.setRange(150, 600)
        self.mask_zoom_slider.setValue(260)
        lbl_mzoom_val = QLabel("260px")
        
        def on_mzoom_change(v):
            lbl_mzoom_val.setText(f"{v}px")
            self.on_mask_zoom_changed(v)
            
        self.mask_zoom_slider.valueChanged.connect(on_mzoom_change)
        
        box_mzoom = QHBoxLayout()
        box_mzoom.addWidget(self.mask_zoom_slider)
        box_mzoom.addWidget(lbl_mzoom_val)
        side_lyt.addWidget(QLabel("<b>Mask Grid Zoom:</b>"))
        side_lyt.addLayout(box_mzoom)
        side_lyt.addStretch()
        mgrid_main_lyt.addWidget(sidebar)

        self.mask_scroll = QScrollArea()
        self.mask_scroll.setWidgetResizable(True)
        self.mask_flow_container = QWidget()
        self.mask_flow_layout = FlowLayout(self.mask_flow_container)
        self.mask_scroll.setWidget(self.mask_flow_container)
        mgrid_main_lyt.addWidget(self.mask_scroll, 1)
        self.mask_stack.addWidget(self.mask_page_grid)

        self.mask_page_detail = QWidget()
        mdet_lyt = QVBoxLayout(self.mask_page_detail)
        mdet_header = QHBoxLayout()
        
        btn_mback = QPushButton("◀ Back to Mask Grid")
        btn_mback.clicked.connect(self.close_mask_detail_view)
        
        self.btn_mask_prev = QPushButton("◁ Prev Angle")
        self.btn_mask_prev.clicked.connect(self.view_mask_prev)
        
        self.lbl_mask_detail_title = QLabel("<h2>Angle 01</h2>")
        self.lbl_mask_detail_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_mask_next = QPushButton("Next Angle ▷")
        self.btn_mask_next.clicked.connect(self.view_mask_next)
        
        mdet_header.addWidget(btn_mback)
        mdet_header.addStretch()
        mdet_header.addWidget(self.btn_mask_prev)
        mdet_header.addWidget(self.lbl_mask_detail_title)
        mdet_header.addWidget(self.btn_mask_next)
        mdet_header.addStretch()
        mdet_lyt.addLayout(mdet_header)
        
        # New Masking Toolbar
        tbar_lyt = QHBoxLayout()
        self.btn_undo = QPushButton("↩ Undo")
        self.btn_undo.clicked.connect(lambda: self.mask_scene.state_manager.undo())
        self.btn_hide_mask = QPushButton("👁 Toggle Mask (Space)")
        self.btn_hide_mask.clicked.connect(lambda: self.mask_scene.toggle_mask())
        
        self.btn_activate_scissors = QPushButton("🧲 Activate Magnetic Scissors")
        self.btn_activate_scissors.setCheckable(True)
        self.btn_activate_scissors.setStyleSheet("QPushButton:checked { background-color: #0d6efd; color: white; font-weight: bold; }")
        self.btn_activate_scissors.clicked.connect(self.on_activate_scissors)
        
        self.btn_lasso_add = QPushButton("➕ Add Selection")
        self.btn_lasso_add.clicked.connect(lambda: self.mask_scene.apply_boolean("add"))
        
        self.btn_lasso_sub = QPushButton("➖ Subtract Selection")
        self.btn_lasso_sub.clicked.connect(lambda: self.mask_scene.apply_boolean("subtract"))
        
        self.btn_lasso_cancel = QPushButton("❌ Cancel")
        self.btn_lasso_cancel.clicked.connect(lambda: self.mask_scene.clear_state())
        
        tbar_lyt.addWidget(self.btn_undo)
        tbar_lyt.addWidget(self.btn_hide_mask)
        tbar_lyt.addSpacing(20)
        tbar_lyt.addWidget(self.btn_activate_scissors)
        tbar_lyt.addWidget(self.btn_lasso_add)
        tbar_lyt.addWidget(self.btn_lasso_sub)
        tbar_lyt.addWidget(self.btn_lasso_cancel)
        tbar_lyt.addStretch()
        
        mdet_lyt.addLayout(tbar_lyt)
        
        editor_layout = QHBoxLayout()
        m_left = QWidget()
        m_left.setFixedWidth(280) 
        ml_layout = QVBoxLayout(m_left)
        ml_layout.addWidget(QLabel("<b>Local Angle Override</b>"))
        
        self.chk_local_mask_thresh = QCheckBox("Override Global Threshold")
        self.chk_local_mask_thresh.toggled.connect(self.on_local_mask_override_toggled)
        ml_layout.addWidget(self.chk_local_mask_thresh)
        
        self.local_mask_thresh_container = QWidget()
        lmt_layout = QHBoxLayout(self.local_mask_thresh_container)
        lmt_layout.setContentsMargins(0,0,0,0)
        self.local_mask_slider = QSlider(Qt.Orientation.Horizontal)
        self.local_mask_slider.setRange(0, 100)
        self.lbl_lmask_val = QLabel("0.50")
        self.local_mask_slider.valueChanged.connect(self.on_local_mask_thresh_changed)
        lmt_layout.addWidget(self.local_mask_slider)
        lmt_layout.addWidget(self.lbl_lmask_val)
        ml_layout.addWidget(self.local_mask_thresh_container)
        self.local_mask_thresh_container.setEnabled(False)
        
        ml_layout.addSpacing(10)
        
        lbl_hint = QLabel("<i>Hold Ctrl to use Magnetic\nSnap when active!</i>")
        lbl_hint.setStyleSheet("color: #666;")
        ml_layout.addWidget(lbl_hint)
        ml_layout.addSpacing(10)

        lbl_hint = QLabel("<i>Hold Ctrl to use Magnetic\nSnap when active!</i>")
        lbl_hint.setStyleSheet("color: #666;")
        ml_layout.addWidget(lbl_hint)
        ml_layout.addSpacing(10)
        
        btn_inv = QPushButton("🌓 Invert Manual Edits")
        btn_inv.clicked.connect(lambda: self.mask_scene.invert_mask())
        ml_layout.addWidget(btn_inv)
        
        btn_clr = QPushButton("↺ Clear Manual Edits")
        btn_clr.clicked.connect(self.clear_manual_edits)
        ml_layout.addWidget(btn_clr)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_journal_clicked)
        ml_layout.addWidget(QLabel("<b>Journal / History</b> (Ctrl+Z)"))
        ml_layout.addWidget(self.history_list)
        editor_layout.addWidget(m_left)
        
        self.mask_scene = MaskingScene(self.magnetic_backend, self.history_list)
        self.mask_scene.maskEdited.connect(self.on_mask_edited)
        self.mask_view = ZoomPanView(self.mask_scene)
        
        editor_layout.addWidget(self.mask_view, 1) 
        mdet_lyt.addLayout(editor_layout, 1)
        self.mask_stack.addWidget(self.mask_page_detail)
        
        self.mask_scroll.viewport().installEventFilter(self)

    def on_activate_scissors(self, checked):
        if not checked:
            self.mask_scene.magnetic_scissors_active = False
            self.btn_activate_scissors.setText("🧲 Activate Magnetic Scissors")
        else:
            self.btn_activate_scissors.setText("⏳ Loading Graph...")
            self.btn_activate_scissors.setEnabled(False)
            QApplication.processEvents() # Force UI update before halting thread
            
            success = self.mask_scene.activate_magnetic_scissors()
            
            if success:
                self.btn_activate_scissors.setText("✂️ Scissors Active (Press to Disable)")
            else:
                # Backend failed or already active
                self.btn_activate_scissors.setChecked(False)
                self.btn_activate_scissors.setText("🧲 Activate Magnetic Scissors")
            
            self.btn_activate_scissors.setEnabled(True)

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.btn_mode_anim.setChecked(mode == "animation")
        self.btn_mode_edge.setChecked(mode == "edge")
        self.controls_stack.setCurrentIndex(0 if mode == "animation" else 1)

        if mode == "animation":
            if self.left_split is not None:
                self.left_split.hide()
            self.nav_layout_widget.show()
            if not self.is_playing:
                self.toggle_global_playback()
        else:
            if self.left_split is not None:
                self.left_split.show()
            self.nav_layout_widget.hide()
            if self.is_playing:
                self.toggle_global_playback()
            # Evict stale edge maps so compute_diagnostic_sync recomputes
            self.edge_map_cache.clear()
            self.dirty_groups = set(range(len(self.groups)))

        for i in range(self.flow_layout.count()):
            item = self.flow_layout.itemAt(i)
            if item and item.widget():
                item.widget().set_view_mode(mode)

        if self.stack.currentIndex() == 1:
            self.update_detail_player()

        if mode == "edge" and self.dirty_groups:
            self.run_alignment_check()

    def update_recalc_ui(self):
        if self.dirty_groups:
            self.lbl_recalc_warning.setVisible(True)
            self.btn_run_alignment.setStyleSheet("QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }")
            self.btn_run_alignment.setText("⚠️ Run Full Alignment Check")
        else:
            self.lbl_recalc_warning.setVisible(False)
            self.btn_run_alignment.setStyleSheet("QPushButton { background-color: #6f42c1; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }")
            self.btn_run_alignment.setText("🔍 Run Full Alignment Check")

    def on_fps_changed(self, value):
        self.lbl_fps.setText(f"{value} FPS")
        self.playback_timer.setInterval(int(1000 / value))

    def toggle_global_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing: 
            self.playback_timer.start()
            self.btn_global_play.setText("⏸ Pause Animation")
            self.btn_detail_play.setText("⏸ Pause")
        else: 
            self.playback_timer.stop()
            self.btn_global_play.setText("▶ Play Animation")
            self.btn_detail_play.setText("▶ Play")

    def tick_animation(self):
        self.global_tick += 1
        if self.stack.currentIndex() == 0 and self.view_mode == "animation":
            for i in range(self.flow_layout.count()):
                item = self.flow_layout.itemAt(i)
                if item and item.widget(): 
                    item.widget().update_view(self.global_tick)
        elif self.stack.currentIndex() == 1 and self.view_mode == "animation":
            if self.active_detail_group >= 0:
                group = self.groups[self.active_detail_group]
                if group: 
                    self.active_frame_idx = self.global_tick % len(group)
                    self.update_detail_player()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            delta = 20 if event.angleDelta().y() > 0 else -20
            if obj == self.scroll_area.viewport():
                new_val = max(150, min(600, self.zoom_slider.value() + delta))
                self.zoom_slider.setValue(new_val)
                return True
            elif obj == self.mask_scroll.viewport():
                new_val = max(150, min(600, self.mask_zoom_slider.value() + delta))
                self.mask_zoom_slider.setValue(new_val)
                return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): 
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
                for f in os.listdir(path):
                    if f.lower().endswith(valid_exts):
                        paths.append(os.path.join(path, f))
            elif path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): 
                paths.append(path)
                
        if paths:
            self.image_paths = list(set(paths))
            self.lbl_loaded_count.setText(f"<b>Loaded: {len(self.image_paths)} images</b>")
            
            self.image_cache.clear()
            self.global_mags_cache.clear()
            self.group_consensus_cache.clear()
            self.group_overrides.clear()
            
            self.max_images_paths.clear()
            self.soft_masks_paths.clear()
            self.manual_edits_paths.clear()
            self.preview_maxes.clear()
            self.preview_soft_masks.clear()
            self.preview_hard_masks.clear()
            self.preview_manual_edits.clear()
            
            for f in os.listdir(self.cache_dir):
                try: 
                    os.remove(os.path.join(self.cache_dir, f))
                except Exception: 
                    pass

    def _init_workspace_dirs(self, dir_path):
        self.workspace_dir = dir_path
        self.input_imgs_dir = os.path.join(dir_path, "input_imgs")
        self.max_imgs_dir = os.path.join(dir_path, "max_stack")
        self.masks_dir = os.path.join(dir_path, "masks")
        self.colmap_dir = os.path.join(dir_path, "colmap")
        
        legacy_average = os.path.join(dir_path, "average")
        if os.path.exists(legacy_average) and not os.path.exists(self.max_imgs_dir):
            try:
                os.rename(legacy_average, self.max_imgs_dir)
            except Exception as e:
                print(f"Failed to migrate legacy average folder: {e}")
                
        os.makedirs(self.input_imgs_dir, exist_ok=True)
        os.makedirs(self.max_imgs_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        os.makedirs(self.colmap_dir, exist_ok=True)

    def create_new_project(self):
        workspace_dir = QFileDialog.getExistingDirectory(self, "1. Select New Empty Workspace Directory")
        if not workspace_dir: return
        
        source_dir = QFileDialog.getExistingDirectory(self, "2. Select Raw Dataset Images Directory")
        if not source_dir: return
        
        self._init_workspace_dirs(workspace_dir)
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        progress = QProgressDialog("Copying input images...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        root_imgs = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(valid_exts)]
        for idx, f in enumerate(root_imgs):
            if progress.wasCanceled(): break
            shutil.copy2(os.path.join(source_dir, f), os.path.join(self.input_imgs_dir, f))
            progress.setValue(int((idx / max(1, len(root_imgs))) * 100))
        progress.setValue(100)

        self._load_internal_workspace()

    def load_project(self):
        workspace_dir = QFileDialog.getExistingDirectory(self, "Select Existing Workspace Directory")
        if workspace_dir:
            self._init_workspace_dirs(workspace_dir)
            self._load_internal_workspace()
            
    def _load_internal_workspace(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = []
        for f in os.listdir(self.input_imgs_dir):
            if f.lower().endswith(valid_exts):
                self.image_paths.append(os.path.join(self.input_imgs_dir, f))
                
        self.lbl_loaded_count.setText(f"<b>Loaded: {len(self.image_paths)} images</b>")
        
        self.image_cache.clear()
        self.global_mags_cache.clear()
        self.edge_map_cache.clear()
        self.mask_overlay_cache.clear()
        self.group_overrides.clear()
        
        self.max_images_paths.clear()
        self.soft_masks_paths.clear()
        self.manual_edits_paths.clear()
        self.preview_maxes.clear()
        self.preview_soft_masks.clear()
        self.preview_hard_masks.clear()
        self.preview_manual_edits.clear()
        
        for f in os.listdir(self.cache_dir):
            try: os.remove(os.path.join(self.cache_dir, f))
            except Exception: pass
            
        json_path = os.path.join(self.workspace_dir, "project.json")
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                self.suppressed_paths = set(os.path.join(self.input_imgs_dir, os.path.basename(p)) for p in metadata.get("suppressed_paths", []))
                self.groups = [[os.path.join(self.input_imgs_dir, os.path.basename(p)) for p in g] for g in metadata.get("groups", [])]
                self.group_overrides = {int(k): v for k, v in metadata.get("group_overrides", {}).items()}
                self.mask_thresholds = {int(k): float(v) for k, v in metadata.get("mask_thresholds", {}).items()}
                self.global_mask_threshold = float(metadata.get("global_mask_threshold", 0.5))
                self.spin_batch.blockSignals(True)
                self.combo_sort.blockSignals(True)
                self.combo_res.blockSignals(True)
                self.spin_batch.setValue(metadata.get("spin_batch", 10))
                self.combo_sort.setCurrentIndex(metadata.get("combo_sort", 0))
                self.combo_res.setCurrentIndex(metadata.get("combo_res", 1))
                self.spin_batch.blockSignals(False)
                self.combo_sort.blockSignals(False)
                self.combo_res.blockSignals(False)
                self.global_settings.set_settings(metadata.get("global_settings", self.global_settings.get_settings()))
                self.mask_thresh_slider.setValue(int(self.global_mask_threshold * 100))
                
                # Restore LINO tab settings
                if hasattr(self, 'lino_scene'):
                    self.lino_scene.combo_res.setCurrentIndex(metadata.get("lino_resolution_idx", 2))
                    self.lino_scene.spin_images.setValue(metadata.get("lino_max_images", 8))
                
                self.dirty_groups = set()
                self.update_recalc_ui()
                self.load_dataset_into_ram(clear_cache=False)
            except Exception as e:
                print(f"Error loading project.json: {e}")
                self.hard_regroup()
        else:
            self.hard_regroup()
        # Look for existing max_stack images to restore pipeline state automatically
        if os.path.exists(self.max_imgs_dir):
            for f in os.listdir(self.max_imgs_dir):
                if f.startswith("mean_group_"):
                    try: os.rename(os.path.join(self.max_imgs_dir, f), os.path.join(self.max_imgs_dir, f.replace("mean_group_", "max_group_")))
                    except Exception: pass
                    
            avg_imgs = [f for f in os.listdir(self.max_imgs_dir) if f.startswith("max_group_") and f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']]
            for img in avg_imgs:
                idx = int(img.split("_")[2].split(".")[0])
                max_path = os.path.join(self.max_imgs_dir, img)
                self.max_images_paths[idx] = max_path
                p_rgb = cv2.cvtColor(cv2.imread(max_path), cv2.COLOR_BGR2RGB)
                h, w = p_rgb.shape[:2]
                scale = min(800/w, 800/h)
                preview = cv2.resize(p_rgb, (int(w*scale), int(h*scale))) if scale < 1.0 else p_rgb.copy()
                self.preview_maxes[idx] = preview
                
                ph, pw = preview.shape[:2]
                edit_path = os.path.join(self.masks_dir, f"manual_edits_{idx}.png")
                self.manual_edits_paths[idx] = edit_path
                if os.path.exists(edit_path):
                    self.preview_manual_edits[idx] = cv2.resize(cv2.imread(edit_path, cv2.IMREAD_GRAYSCALE), (pw, ph), interpolation=cv2.INTER_NEAREST)
                else:
                    self.preview_manual_edits[idx] = np.full((ph, pw), 128, dtype=np.uint8)
                    
                soft_path = os.path.join(self.masks_dir, f"soft_mask_{idx}.png")
                soft_path_webp = os.path.join(self.masks_dir, f"soft_mask_{idx}.webp")
                
                if os.path.exists(soft_path):
                    self.soft_masks_paths[idx] = soft_path
                    self.preview_soft_masks[idx] = cv2.resize(cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0, (pw, ph), interpolation=cv2.INTER_AREA)
                elif os.path.exists(soft_path_webp):
                    self.soft_masks_paths[idx] = soft_path_webp
                    self.preview_soft_masks[idx] = cv2.resize(cv2.imread(soft_path_webp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0, (pw, ph), interpolation=cv2.INTER_AREA)
                else:
                    self.preview_soft_masks[idx] = np.zeros((ph, pw), dtype=np.float32)
                    
                self.preview_hard_masks[idx] = np.zeros((ph, pw), dtype=np.uint8)

            
            if self.max_images_paths:
                self.main_tabs.setTabEnabled(1, True)
                self.main_tabs.setTabEnabled(2, True)
                self.main_tabs.setTabEnabled(3, True)
                self.main_tabs.setTabEnabled(4, True)
                self.main_tabs.setTabEnabled(5, True)
                self.main_tabs.setCurrentIndex(1)  # Jump straight to Mask/SfM pipeline
                self.update_all_hard_masks()
                self.render_mask_grid()
                
                sparse_0_path = os.path.join(self.colmap_dir, "sparse", "0")
                if os.path.exists(sparse_0_path):
                    self.sfm_scene.load_reconstruction_model(sparse_0_path)


    def hard_regroup(self):
        if not self.image_paths: 
            return
            
        self.playback_timer.stop()
        if self.combo_sort.currentIndex() == 0:
            self.image_paths.sort(key=lambda x: natsort_key(os.path.basename(x)))
        else:
            self.image_paths.sort(key=lambda x: os.path.getmtime(x))
            
        active = [p for p in self.image_paths if p not in self.suppressed_paths]
        bs = self.spin_batch.value()
        self.groups = [active[i:i+bs] for i in range(0, len(active), bs)]
        
        self.edge_map_cache.clear()
        self.mask_overlay_cache.clear()
        self.group_overrides.clear()
        
        self.max_images_paths.clear()
        self.soft_masks_paths.clear()
        self.manual_edits_paths.clear()
        self.preview_maxes.clear()
        self.preview_soft_masks.clear()
        self.preview_hard_masks.clear()
        self.preview_manual_edits.clear()
                
        self.dirty_groups = set(range(len(self.groups)))
        self._auto_save_project()
        self.update_recalc_ui()
        self.load_dataset_into_ram(clear_cache=False)

    def on_resolution_changed(self):
        if self.groups:
            self.playback_timer.stop()
            self.global_mags_cache.clear()
            self.edge_map_cache.clear()
            self.mask_overlay_cache.clear()
            
            self.max_images_paths.clear()
            self.soft_masks_paths.clear()
            self.manual_edits_paths.clear()
            self.preview_maxes.clear()
            self.preview_soft_masks.clear()
            self.preview_hard_masks.clear()
            self.preview_manual_edits.clear()
            
            for f in os.listdir(self.cache_dir):
                try: 
                    os.remove(os.path.join(self.cache_dir, f))
                except Exception: 
                    pass
                    
            self.dirty_groups = set(range(len(self.groups)))
            self.update_recalc_ui()
            self.load_dataset_into_ram(clear_cache=True)

    def on_zoom_changed(self, value):
        for i in range(self.flow_layout.count()):
            item = self.flow_layout.itemAt(i)
            if item and item.widget(): 
                item.widget().set_zoom(value)

    def load_dataset_into_ram(self, clear_cache=True):
        if self.worker and self.worker.isRunning(): 
            self.worker.is_interrupted = True
            self.worker.wait()
            
        if clear_cache: 
            self.image_cache.clear()
            
        res_map = {0: 400, 1: 600, 2: 800, 3: 99999}
        target_res = res_map[self.combo_res.currentIndex()]
        paths_to_load = []
        for group in self.groups:
            for p in group:
                if p not in self.image_cache:
                    paths_to_load.append(p)
        
        if not paths_to_load: 
            self._on_dataset_loaded()
            return
            
        self.progress = QProgressDialog("Loading Dataset into RAM...", "Cancel", 0, len(paths_to_load), self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.canceled.connect(self.cancel_worker)
        
        self.worker = DatasetLoaderThread(paths_to_load, target_res)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.image_ready.connect(self._on_image_loaded)
        self.worker.finished_loading.connect(self._on_dataset_loaded)
        self.worker.start()

    def _on_image_loaded(self, path, img_rgb, q_img):
        self.image_cache[path] = {'rgb': img_rgb, 'pixmap': QPixmap.fromImage(q_img)}
        
    def _on_dataset_loaded(self):
        if self.worker and self.worker.is_interrupted: 
            return
            
        self.render_basic_grid()
        if self.is_playing and self.view_mode == "animation": 
            self.playback_timer.start()
            
        if self.stack.currentIndex() == 1: 
            self.update_detail_player()
            
        if self.view_mode == "edge" and self.dirty_groups: 
            self.run_alignment_check()

    def render_basic_grid(self):
        if not self.groups:
            return

        while self.flow_layout.count() > 0:
            item = self.flow_layout.takeAt(0)
            if item and item.widget():
                item.widget().hide()
                item.widget().deleteLater()

        zoom_val = self.zoom_slider.value()
        for i, group in enumerate(self.groups):
            card = HybridGroupCard(
                i, group, self.spin_batch.value(),
                self.image_cache, self.edge_map_cache,
                zoom_size=zoom_val, parent=self.flow_container
            )
            card.set_view_mode(self.view_mode)
            card.clicked.connect(self.open_detail_view)
            self.flow_layout.addWidget(card)

    def run_alignment_check(self):
        if not self.groups or not self.dirty_groups: 
            return
        
        for i in self.dirty_groups:
            item = self.flow_layout.itemAt(i)
            if item and item.widget(): 
                item.widget().set_loading()
                
        res_map = {0: 400, 1: 600, 2: 800, 3: 99999}
        target_res = res_map[self.combo_res.currentIndex()]
        
        if self.worker and self.worker.isRunning(): 
            self.worker.is_interrupted = True
            self.worker.wait()
            
        self.progress = QProgressDialog("Running GPU Check..." if HAS_TORCH else "Running CPU Check...", "Cancel", 0, len(self.dirty_groups), self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.canceled.connect(self.cancel_worker)
        
        self.worker = DiagnosticWorker(
            self.groups, self.dirty_groups, target_res, self.global_settings.get_settings(), 
            self.group_overrides.copy(), self.image_cache, self.global_mags_cache, self.cache_dir, self.chk_disk_cache.isChecked()
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.group_ready.connect(self._on_worker_group_ready)
        self.worker.start()

    def _on_worker_group_ready(self, idx, num_images, q_img):
        self.edge_map_cache[idx] = QPixmap.fromImage(q_img)
        if idx in self.dirty_groups:
            self.dirty_groups.remove(idx)

        if self.view_mode == "edge" and self.main_tabs.currentIndex() == 0:
            item = self.flow_layout.itemAt(idx)
            if item and item.widget():
                item.widget().update_view(0)

        self.update_recalc_ui()

    def compute_diagnostic_sync(self, group_idx):
        if group_idx >= len(self.groups) or not self.groups[group_idx]: 
            return None
            
        s = self.group_overrides.get(group_idx, self.global_settings.get_settings())
        paths = self.groups[group_idx]
        num_images = len(paths)
        res_map = {0: 400, 1: 600, 2: 800, 3: 99999}
        target_res = res_map[self.combo_res.currentIndex()]
        
        missing_paths = []
        for p in paths:
            if (p, target_res, s['blur']) not in self.global_mags_cache:
                missing_paths.append(p)
        
        if missing_paths:
            images_gray = []
            for p in missing_paths:
                if p in self.image_cache: 
                    images_gray.append(cv2.cvtColor(self.image_cache[p]['rgb'], cv2.COLOR_RGB2GRAY).astype(np.float32))
            if images_gray:
                if HAS_TORCH:
                    with torch.no_grad():
                        tensor_gray = torch.from_numpy(np.stack(images_gray)).unsqueeze(1).to('cuda', non_blocking=True)
                        if s['blur'] > 0:
                            k = s['blur'] * 2 + 1
                            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
                            x = torch.arange(-k // 2 + 1., k // 2 + 1., device='cuda')
                            xx, yy = torch.meshgrid(x, x, indexing='ij')
                            kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                            kernel = (kernel / kernel.sum()).view(1, 1, k, k)
                            tensor_gray = F.conv2d(tensor_gray, kernel, padding=k//2)
                        gx = F.conv2d(tensor_gray, SOBEL_X, padding=1)
                        gy = F.conv2d(tensor_gray, SOBEL_Y, padding=1)
                        all_mags_missing = torch.sqrt(gx**2 + gy**2).squeeze(1).cpu().numpy()
                    torch.cuda.empty_cache()
                else:
                    all_mags_missing = np.zeros((len(images_gray), images_gray[0].shape[0], images_gray[0].shape[1]), dtype=np.float32)
                    for mi, gray in enumerate(images_gray):
                        if s['blur'] > 0: 
                            gray = cv2.GaussianBlur(gray, (s['blur']*2+1, s['blur']*2+1), 0)
                        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                        all_mags_missing[mi] = np.sqrt(gx**2 + gy**2)
                        
                for mi, p in enumerate(missing_paths):
                    if self.chk_disk_cache.isChecked():
                        safe_name = hashlib.md5(p.encode('utf-8')).hexdigest()
                        cache_path = os.path.join(self.cache_dir, f"{safe_name}_{target_res}_{s['blur']}.npy")
                        np.save(cache_path, all_mags_missing[mi])
                        self.global_mags_cache[(p, target_res, s['blur'])] = cache_path
                    else: 
                        self.global_mags_cache[(p, target_res, s['blur'])] = all_mags_missing[mi]
        try:
            if self.chk_disk_cache.isChecked(): 
                all_mags = np.stack([np.load(self.global_mags_cache[(p, target_res, s['blur'])]) for p in paths])
            else: 
                all_mags = np.stack([self.global_mags_cache[(p, target_res, s['blur'])] for p in paths])
        except Exception: 
            return None 
            
        h = all_mags.shape[1]
        w = all_mags.shape[2]
        min_consensus = max(1, int(num_images * (s['consensus'] / 100.0)))
        frame_colors = np.zeros((num_images, 3), dtype=np.uint8)
        hue_step = 300 / (num_images - 1) if num_images > 1 else 0
        
        for i in range(num_images):
            color_bgr = cv2.cvtColor(np.uint8([[[int((i * hue_step) / 2), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            frame_colors[i] = [color_bgr[2], color_bgr[1], color_bgr[0]]
            
        if HAS_TORCH:
            with torch.no_grad():
                all_mags_t = torch.from_numpy(all_mags).to('cuda', non_blocking=True)
                masks_t = all_mags_t > s['threshold']
                counts_t = masks_t.sum(dim=0)
                
                frame_indices_t = torch.arange(1, num_images + 1, device='cuda').view(num_images, 1, 1)
                last_seen_t = torch.max(masks_t.to(torch.int32) * frame_indices_t, dim=0)[0] - 1
                
                out_img_t = torch.zeros((h, w, 3), dtype=torch.uint8, device='cuda')
                anomaly_mask_t = (counts_t > 0) & (counts_t < min_consensus)
                consensus_mask_t = (counts_t >= min_consensus)
                
                frame_colors_t = torch.tensor(frame_colors, device='cuda', dtype=torch.uint8)
                out_img_t[anomaly_mask_t] = frame_colors_t[last_seen_t[anomaly_mask_t]]
                if s['show_white']: 
                    out_img_t[consensus_mask_t] = WHITE_COLOR
                out_img = out_img_t.cpu().numpy()
        else:
            masks = all_mags > s['threshold']
            counts = np.sum(masks, axis=0)
            
            frame_indices = np.arange(1, num_images + 1).reshape(num_images, 1, 1)
            last_seen = np.max(masks * frame_indices, axis=0) - 1
            
            out_img = np.zeros((h, w, 3), dtype=np.uint8)
            anomaly_mask = (counts > 0) & (counts < min_consensus)
            consensus_mask = (counts >= min_consensus)
            
            out_img[anomaly_mask] = frame_colors[last_seen[anomaly_mask]]
            if s['show_white']: 
                out_img[consensus_mask] = [255, 255, 255]
                
        pixmap = QPixmap.fromImage(QImage(out_img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        self.edge_map_cache[group_idx] = pixmap
        if group_idx in self.dirty_groups:
            self.dirty_groups.remove(group_idx)
            self.update_recalc_ui()

        return pixmap

    def on_global_settings_changed(self):
        if self.groups:
            self.dirty_groups = set(range(len(self.groups)))
            self.update_recalc_ui()
            self.render_basic_grid()
            if self.stack.currentIndex() == 1 and self.active_detail_group not in self.group_overrides: 
                self.update_detail_player()

    def on_local_settings_changed(self):
        idx = self.active_detail_group
        if idx < 0: 
            return
            
        if self.local_settings.chk_override.isChecked(): 
            self.group_overrides[idx] = self.local_settings.get_settings()
        else:
            if idx in self.group_overrides: 
                del self.group_overrides[idx]
                
        self.update_detail_player()

    def open_detail_view(self, idx):
        self.active_detail_group = idx
        self.active_frame_idx = 0
        self.lbl_detail_title.setText(f"<h2>Angle {idx+1:02d}</h2>")
        self.btn_view_prev.setVisible(idx > 0)
        self.btn_view_next.setVisible(idx < len(self.groups) - 1)
        
        self.local_settings.chk_override.blockSignals(True)
        if idx in self.group_overrides: 
            self.local_settings.chk_override.setChecked(True)
            self.local_settings.controls_widget.setEnabled(True)
            self.local_settings.set_settings(self.group_overrides[idx])
        else: 
            self.local_settings.chk_override.setChecked(False)
            self.local_settings.controls_widget.setEnabled(False)
            self.local_settings.set_settings(self.global_settings.get_settings())
        self.local_settings.chk_override.blockSignals(False)
        
        self.global_tick = 0
        self.update_detail_player()
        self.stack.setCurrentIndex(1)
        self.setFocus() 

    def close_detail_view(self):
        self.active_detail_group = -1
        self.render_basic_grid()
        self.stack.setCurrentIndex(0)
        self.setFocus()
        
    def view_prev_group(self):
        if self.active_detail_group > 0: 
            self.open_detail_view(self.active_detail_group - 1)
            
    def view_next_group(self):
        if self.active_detail_group < len(self.groups) - 1: 
            self.open_detail_view(self.active_detail_group + 1)
            
    def on_frame_slider_changed(self, value):
        idx = self.active_detail_group
        if 0 <= idx < len(self.groups) and 0 <= value < len(self.groups[idx]): 
            self.active_frame_idx = value
            self.update_detail_player()

    def update_detail_player(self):
        idx = self.active_detail_group
        if idx < 0 or idx >= len(self.groups) or not self.groups[idx]:
            self.detail_player.setText("No Images")
            return
            
        group = self.groups[idx]
        self.active_frame_idx = max(0, min(self.active_frame_idx, len(group) - 1))
        f_idx = self.active_frame_idx
        path = group[f_idx]
        
        # Strict if/else: edge-map cache and mask-overlay cache are separate
        if self.view_mode == "animation":
            if path in self.image_cache:
                self.detail_player.setPixmap(self.image_cache[path]['pixmap'])
            else:
                self.detail_player.setText("Loading...")
        else:
            pixmap = self.compute_diagnostic_sync(idx)
            if pixmap:
                self.detail_player.setPixmap(pixmap)

        self.lbl_frame_info.setText(
            f"<b>Frame {f_idx+1} / {len(group)}</b> — "
            f"<span style='font-size:10px'>{os.path.basename(path)}</span>")
        self.btn_cascade_left.setVisible(idx > 0)
        self.btn_local_left.setVisible(idx > 0)
        self.btn_cascade_right.setVisible(idx < len(self.groups)-1)
        self.btn_local_right.setVisible(idx < len(self.groups)-1)
        
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(len(group) - 1)
        self.frame_slider.setValue(f_idx)
        self.frame_slider.blockSignals(False)

    def scrub_frame(self, delta):
        if self.active_detail_group < 0: 
            return
            
        if self.view_mode == "animation" and self.is_playing: 
            self.toggle_global_playback()
            
        self.active_frame_idx = (self.active_frame_idx + delta) % len(self.groups[self.active_detail_group])
        self.update_detail_player()

    def _invalidate_masking_for_groups(self, indices):
        """Selectively discard masking/max-stack data for the given group indices.

        Data for all other groups is preserved so the user doesn't lose work
        on unaffected groups when they make a local grouping edit.
        Also invalidates the SFM model since the image topology changed.
        """
        for i in indices:
            for d in (self.max_images_paths, self.soft_masks_paths,
                      self.manual_edits_paths, self.preview_maxes,
                      self.preview_soft_masks, self.preview_hard_masks,
                      self.preview_manual_edits, self.mask_thresholds,
                      self.mask_overlay_cache, self.edge_map_cache):
                d.pop(i, None)
        # SFM is topology-dependent — always invalidate
        colmap_sparse = os.path.join(self.colmap_dir, "sparse")
        if os.path.isdir(colmap_sparse):
            import shutil as _shutil
            try:
                _shutil.rmtree(colmap_sparse)
            except Exception:
                pass

    def refresh_after_edit(self):
        idx = self.active_detail_group
        # Selectively invalidate masking data only for the dirty groups
        if self.dirty_groups:
            self._invalidate_masking_for_groups(self.dirty_groups)
        self.update_recalc_ui()
        if self.view_mode == "edge" and self.dirty_groups:
            self.compute_diagnostic_sync(idx)
            self.run_alignment_check()

        if not self.groups[idx]:
            self.groups.pop(idx)
            self.render_basic_grid()
            self.close_detail_view()
        else:
            self.active_frame_idx = min(self.active_frame_idx, len(self.groups[idx]) - 1)
            self.update_detail_player()


    def move_local(self, direction):
        idx = self.active_detail_group
        group = self.groups[idx]
        
        if direction == -1: 
            if idx == 0: 
                return
            img = group.pop(0)
            self.groups[idx-1].append(img)
            self.dirty_groups.update([idx, idx-1])
        else: 
            img = group.pop(-1)
            if idx == len(self.groups) - 1: 
                self.groups.append([img])
            else: 
                self.groups[idx+1].insert(0, img)
            self.dirty_groups.update([idx, idx+1])
            
        self.refresh_after_edit()

    def move_cascade(self, direction):
        idx = self.active_detail_group
        group = self.groups[idx]
        
        if direction == -1: 
            if idx == 0: 
                return
            img = group.pop(0)
            self.groups[idx-1].append(img)
            for i in range(idx, len(self.groups) - 1):
                if self.groups[i+1]: 
                    self.groups[i].append(self.groups[i+1].pop(0))
            if not self.groups[-1]: 
                self.groups.pop()
            self.dirty_groups.update(range(idx-1, len(self.groups)))
        else: 
            img = group.pop(-1)
            if idx == len(self.groups) - 1: 
                self.groups.append([img])
            else:
                self.groups[idx+1].insert(0, img)
                for i in range(idx + 1, len(self.groups) - 1):
                    if self.groups[i]: 
                        self.groups[i+1].insert(0, self.groups[i].pop(-1))
                if len(self.groups[-1]) > self.spin_batch.value(): 
                    self.groups.append([self.groups[-1].pop(-1)])
            self.dirty_groups.update(range(idx, len(self.groups)))
            
        self.refresh_after_edit()

    def suppress_image(self):
        self.suppressed_paths.add(self.groups[self.active_detail_group].pop(self.active_frame_idx))
        self.dirty_groups.add(self.active_detail_group)
        self.refresh_after_edit()

    def proceed_to_masking(self):
        if not self.groups: 
            return
            
        self.playback_timer.stop()
        if self.worker and self.worker.isRunning(): 
            self.worker.is_interrupted = True
            self.worker.wait()
            
        self.global_mags_cache.clear()
        self.edge_map_cache.clear()
        self.mask_overlay_cache.clear()

        for f in os.listdir(self.cache_dir):
            try: 
                os.remove(os.path.join(self.cache_dir, f))
            except Exception: 
                pass
                
        self.max_images_paths.clear()
        self.soft_masks_paths.clear()
        self.manual_edits_paths.clear()
        self.preview_maxes.clear()
        self.preview_soft_masks.clear()
        self.preview_hard_masks.clear()
        self.preview_manual_edits.clear()
        self.mask_thresholds.clear()
        
        self.progress = QProgressDialog("Computing Full Resolution Mean Images...", "Cancel", 0, len(self.groups), self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.canceled.connect(self.cancel_worker)
        
        self.worker = MaxImageWorker(self.groups, self.max_imgs_dir)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.group_max_ready.connect(self._on_mean_image_ready)
        self.worker.finished.connect(self._on_mean_images_completed)
        self.worker.start()
        
    def _on_mean_image_ready(self, idx, max_path, preview_rgb, full_h, full_w):
        self.max_images_paths[idx] = max_path
        self.preview_maxes[idx] = preview_rgb
        
        manual_edits_path = os.path.join(self.masks_dir, f"manual_edits_{idx}.png")
        cv2.imwrite(manual_edits_path, np.full((full_h, full_w), 128, dtype=np.uint8))
        self.manual_edits_paths[idx] = manual_edits_path
        
        ph = preview_rgb.shape[0]
        pw = preview_rgb.shape[1]
        self.preview_manual_edits[idx] = np.full((ph, pw), 128, dtype=np.uint8)
        self.preview_soft_masks[idx] = np.zeros((ph, pw), dtype=np.float32)
        self.preview_hard_masks[idx] = np.zeros((ph, pw), dtype=np.uint8)

    def _on_mean_images_completed(self):
        if self.worker and self.worker.is_interrupted: 
            return
            
        self.main_tabs.setTabEnabled(1, True)
        self.main_tabs.setCurrentIndex(1)
        self.update_all_hard_masks()
        self.render_mask_grid()

    def on_mask_zoom_changed(self, value):
        for i in range(self.mask_flow_layout.count()):
            item = self.mask_flow_layout.itemAt(i)
            if item and item.widget(): 
                item.widget().set_zoom(value)

    def run_inspyrenet(self):
        if not self.max_images_paths or not HAS_INSPYRENET: 
            return
            
        res_map = {0: 1024, 1: 2048, 2: 4096, 3: 0}
        target_res = res_map[self.masking_res_combo.currentIndex()]
        
        self.progress = QProgressDialog(f"Running InSPyReNet (Res: {target_res if target_res > 0 else 'Native'})...", "Cancel", 0, len(self.max_images_paths), self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.canceled.connect(self.cancel_worker)
        
        self.worker = InSPyReNetWorker(self.max_images_paths, self.masks_dir, target_res)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.mask_ready.connect(self._on_soft_mask_ready)
        self.worker.finished.connect(self._on_inspyrenet_completed)
        self.worker.start()

    def _on_soft_mask_ready(self, idx, mask_path):
        self.soft_masks_paths[idx] = mask_path
        
        full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        h = self.preview_maxes[idx].shape[0]
        w = self.preview_maxes[idx].shape[1]
        self.preview_soft_masks[idx] = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_AREA)

    def _on_inspyrenet_completed(self):
        if self.worker and self.worker.is_interrupted: 
            return
        self.update_all_hard_masks()

    def update_all_hard_masks(self):
        for idx in self.max_images_paths.keys(): 
            self._recalc_hard_mask(idx)
            
        if self.mask_stack.currentIndex() == 0: 
            self.render_mask_grid()
        elif self.mask_stack.currentIndex() == 1: 
            self.render_mask_grid()
            self.mask_scene.update_mask_visuals()

    def _recalc_hard_mask(self, idx):
        if idx not in self.preview_soft_masks: 
            return
            
        thresh = self.mask_thresholds.get(idx, self.global_mask_threshold)
        s_mask = self.preview_soft_masks[idx]
        m_edit = self.preview_manual_edits[idx]
        
        if len(s_mask.shape) > 2:
            if s_mask.shape[-1] == 4:
                s_mask = s_mask[:, :, 3] 
            else:
                s_mask = np.mean(s_mask, axis=-1)
                
        base_hard = (s_mask > thresh).astype(np.uint8) * 255
        
        final_mask = np.where(m_edit == 255, 255, base_hard)
        final_mask = np.where(m_edit == 0, 0, final_mask)
        self.preview_hard_masks[idx] = final_mask
        
        max_img = self.preview_maxes[idx]
        h = max_img.shape[0]
        w = max_img.shape[1]
        
        overlay = max_img.copy()
        mask_bool = final_mask == 255
        
        # Red tint overlay directly applied via fast matrix math
        # overlay[~mask_bool] means "for every background pixel"
        overlay_bg = overlay[~mask_bool]
        
        # Multiply existing pixels by 0.5 and add pure red (255, 0, 0) * 0.5
        overlay[~mask_bool] = (overlay_bg * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        
        q_img = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.mask_overlay_cache[idx] = QPixmap.fromImage(q_img)

    def render_mask_grid(self):
        while self.mask_flow_layout.count() > 0:
            item = self.mask_flow_layout.takeAt(0)
            if item and item.widget():
                item.widget().hide()
                item.widget().deleteLater()

        zoom_val = self.mask_zoom_slider.value()

        for idx in self.max_images_paths.keys():
            card = HybridGroupCard(
                idx, self.groups[idx], self.spin_batch.value(),
                self.image_cache, self.mask_overlay_cache,
                zoom_size=zoom_val, parent=self.mask_flow_container
            )
            card.set_view_mode("masking")
            card.clicked.connect(self.open_mask_detail_view)
            self.mask_flow_layout.addWidget(card)

        self.main_tabs.setTabEnabled(2, len(self.max_images_paths) > 0)
        self.main_tabs.setTabEnabled(3, len(self.max_images_paths) > 0)

    def force_save_project(self):
        self._auto_save_project()
        QMessageBox.information(self, "Saved", "Project configuration saved.")

    def _auto_save_project(self):
        if not hasattr(self, 'workspace_dir') or not self.workspace_dir:
            return
            
        import json
        metadata = {
            "version": "1.2", 
            "suppressed_paths": list(self.suppressed_paths) if hasattr(self, 'suppressed_paths') else [], 
            "groups": getattr(self, 'groups', []), 
            "global_settings": self.global_settings.get_settings() if hasattr(self, 'global_settings') else {}, 
            "group_overrides": getattr(self, 'group_overrides', {}), 
            "global_mask_threshold": getattr(self, 'global_mask_threshold', 0.5), 
            "mask_thresholds": getattr(self, 'mask_thresholds', {}), 
            "spin_batch": self.spin_batch.value() if hasattr(self, 'spin_batch') else 10, 
            "combo_sort": self.combo_sort.currentIndex() if hasattr(self, 'combo_sort') else 0, 
            "combo_res": self.combo_res.currentIndex() if hasattr(self, 'combo_res') else 1,
            "lino_resolution_idx": self.lino_scene.combo_res.currentIndex() if hasattr(self, 'lino_scene') else 2,
            "lino_max_images": self.lino_scene.spin_images.value() if hasattr(self, 'lino_scene') else 8
        }
        
        try:
            with open(os.path.join(self.workspace_dir, "project.json"), "w") as f: 
                json.dump(metadata, f, indent=4)
        except Exception:
            pass

    # -----------------------------------------------------
    # TAB 3: STRUCTURE FROM MOTION (COLMAP)
    # -----------------------------------------------------
    def run_colmap_sfm(self, use_masks):
        self._auto_save_project()
        if hasattr(self, 'cache_dir') and not hasattr(self, 'workspace_dir'):
            self.workspace_dir = self.cache_dir
            
        if not hasattr(self, 'workspace_dir'):
            QMessageBox.warning(self, "No Workspace", "Please load a dataset folder to establish a workspace first.")
            return

        self.sfm_scene.btn_run_sfm.setEnabled(False)
        self.sfm_scene.set_progress("Initializing COLMAP...", 5)
        
        # Save hard masks to disk if use_masks is checked
        if use_masks:
            try:
                for idx, mask_np in self.preview_hard_masks.items():
                    original_name = os.path.basename(self.max_images_paths[idx])
                    mask_filename = original_name + ".png" # PyColmap specific mask naming
                    save_path = os.path.join(self.masks_dir, mask_filename)
                    cv2.imwrite(save_path, mask_np)
            except Exception as e:
                self.sfm_scene.set_progress("Failed to save masks.", 0)
                self.sfm_scene.btn_run_sfm.setEnabled(True)
                return

        try:
            self.colmap_worker = ColmapWorker(self.workspace_dir, use_masks)
            self.colmap_worker.progress.connect(self.sfm_scene.set_progress)
            self.colmap_worker.finished_reconstruction.connect(self._on_colmap_finished)
            self.colmap_worker.start()
        except ImportError:
            QMessageBox.critical(self, "COLMAP Missing", "pycolmap is not installed or failed to initialize.")
            self.sfm_scene.btn_run_sfm.setEnabled(True)

    def _on_colmap_finished(self, success, message):
        self.sfm_scene.btn_run_sfm.setEnabled(True)
        if success:
            self.sfm_scene.set_progress("Reconstruction successful. Loading viewer...", 100)
            self.sfm_scene.load_reconstruction_model(message)
        else:
            self.sfm_scene.set_progress("Reconstruction failed.", 0)
            QMessageBox.critical(self, "COLMAP Error", f"Structure from Motion failed:\n{message}")

    # -----------------------------------------------------
    # TAB 4: LINO-UNIPS NORMAL EXTRACTION
    # -----------------------------------------------------
    def run_lino_inference(self, resolution, max_images):
        self._auto_save_project()
        
        if not hasattr(self, 'workspace_dir') or not self.workspace_dir:
            QMessageBox.warning(self, "No Workspace", "Please load a dataset folder to establish a workspace first.")
            self.lino_scene.reset_ui()
            return
            
        if not hasattr(self, 'groups') or not self.groups:
            QMessageBox.warning(self, "No Groups", "Please generate views in Grouping tab first.")
            self.lino_scene.reset_ui()
            return

        # Prepare FULL-RESOLUTION mask paths (1:1 with groups).
        # preview_hard_masks are at thumbnail resolution and CANNOT be used
        # for cropping full-res images — the bbox coordinates would be wrong.
        mask_paths = []
        for idx in range(len(self.groups)):
            if idx not in self.preview_hard_masks or idx not in self.max_images_paths:
                self.lino_scene.reset_ui()
                QMessageBox.critical(self, "Error", f"Mask for group {idx} not fully generated! Make sure you wait for all masks.")
                return

            # Build full-resolution hard mask from the full-res assets on disk
            thresh = self.mask_thresholds.get(idx, self.global_mask_threshold)

            if idx in self.soft_masks_paths:
                soft_path = self.soft_masks_paths[idx]
                print(f"[LINO] Group {idx}: reading soft mask from {soft_path} (exists={os.path.exists(soft_path)})")
                full_soft = cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE)
                if full_soft is not None:
                    print(f"[LINO] Group {idx}: soft mask shape={full_soft.shape}, min={full_soft.min()}, max={full_soft.max()}")
                    full_soft = full_soft.astype(np.float32) / 255.0
                else:
                    print(f"[LINO] Group {idx}: WARNING soft mask read returned None!")
                    max_img = cv2.imread(self.max_images_paths[idx])
                    full_soft = np.zeros(max_img.shape[:2], dtype=np.float32)
            else:
                print(f"[LINO] Group {idx}: WARNING no soft mask path stored! Using zeros.")
                max_img = cv2.imread(self.max_images_paths[idx])
                full_soft = np.zeros(max_img.shape[:2], dtype=np.float32)

            full_manual = cv2.imread(self.manual_edits_paths[idx], cv2.IMREAD_GRAYSCALE)
            if full_manual is None:
                full_manual = np.full(full_soft.shape, 128, dtype=np.uint8)

            base_hard = (full_soft > thresh).astype(np.uint8) * 255
            full_hard = np.where(full_manual == 255, 255, base_hard)
            full_hard = np.where(full_manual == 0, 0, full_hard)
            
            white_px = np.count_nonzero(full_hard)
            total_px = full_hard.shape[0] * full_hard.shape[1]
            print(f"[LINO] Group {idx}: thresh={thresh:.3f}, hard mask white pixels={white_px}/{total_px} ({100*white_px/total_px:.1f}%)")

            original_name = os.path.basename(self.max_images_paths[idx])
            mask_filename = f"lino_fullres_mask_{idx:02d}.png"
            save_path = os.path.join(self.masks_dir, mask_filename)
            cv2.imwrite(save_path, full_hard)
            mask_paths.append(save_path)

        # Unique run name
        base_name = f"run_res{resolution}_max{max_images}"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{base_name}_{timestamp}"
        
        self.lino_worker = LinoWorker(self.workspace_dir, run_name, self.groups, mask_paths, resolution, max_images)
        self.lino_worker.progress.connect(self.lino_scene.set_progress)
        self.lino_worker.formatting_done.connect(self._on_lino_formatting_done)
        self.lino_worker.group_done.connect(self._on_lino_group_done)
        self.lino_worker.finished_reconstruction.connect(self._on_lino_finished)
        self.lino_worker.start()

    def _on_lino_resume(self, run_dir, resolution, max_images):
        """Resume an existing run — skip groups whose outputs already exist."""
        import json as _json
        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            QMessageBox.warning(self, "Resume Failed",
                                f"Cannot find run_config.json in:\n{run_dir}")
            self.lino_scene.reset_ui()
            return
        try:
            with open(cfg_path) as f:
                cfg = _json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Resume Failed", str(e))
            self.lino_scene.reset_ui()
            return

        outputs_dir = os.path.join(run_dir, "outputs")
        groups_cfg = cfg.get("groups", [])

        # Build skip set — groups that already have a normal map output
        skip_indices = set()
        for g in groups_cfg:
            gi = g["group_idx"]
            for ext in (".exr", ".npy"):
                if os.path.exists(os.path.join(outputs_dir, f"{gi:02d}_normal{ext}")):
                    skip_indices.add(gi)
                    break

        self.lino_scene.set_active_run_dir(run_dir)
        self.lino_scene._load_run_grid(run_dir)

        self.lino_worker = LinoWorker(
            self.workspace_dir,
            os.path.basename(run_dir),   # keep same run name / directory
            self.groups, [], resolution, max_images,
            resume_run_dir=run_dir,
            skip_group_indices=skip_indices,
        )
        self.lino_worker.progress.connect(self.lino_scene.set_progress)
        self.lino_worker.formatting_done.connect(self._on_lino_formatting_done)
        self.lino_worker.group_done.connect(self._on_lino_group_done)
        self.lino_worker.finished_reconstruction.connect(self._on_lino_finished)
        self.lino_worker.start()

    def _on_lino_formatting_done(self, run_dir):
        """Formatting complete — build the grid showing first-input images."""
        self.lino_scene.set_active_run_dir(run_dir)
        self.lino_scene._load_run_grid(run_dir)

    def _on_lino_group_done(self, group_idx, exr_path):
        """Live-update the grid card after each group finishes inference."""
        self.lino_scene.update_card_for_group(group_idx, exr_path)

    def _on_lino_finished(self, success, message, run_dir):
        self.lino_scene.reset_ui()    # reset_ui now calls _load_run_params if run_dir known
        if success:
            self.lino_scene.set_progress("LINO Reconstruction successful.", 100)
            self.lino_scene.populate_runs(self.workspace_dir)
            QMessageBox.information(self, "Success", "LINO-UniPS extracted standard normals successfully.")
        else:
            self.lino_scene.set_progress("Reconstruction failed or cancelled.", 0)
            if "Cancelled" not in message:
                QMessageBox.critical(self, "LINO Error", f"LINO-UniPS Inference failed:\n{message}")
            # Refresh run params so Resume button appears for incomplete runs
            if run_dir and os.path.isdir(run_dir):
                self.lino_scene.set_active_run_dir(run_dir)
                self.lino_scene._load_run_params(run_dir)

    # ── World-frame conversion ─────────────────────────────────────────
    def _on_convert_to_world_frame(self, run_dir):
        """Validate COLMAP model exists then start WorldFrameConverter."""
        colmap_sparse = os.path.join(self.colmap_dir, "sparse", "0")
        if not os.path.isdir(colmap_sparse) or \
                not os.path.exists(os.path.join(colmap_sparse, "images.txt")):
            QMessageBox.information(
                self, "SfM Model Required",
                "No SfM model found.\n\n"
                "Please run the SfM Reconstruction on the SfM tab first, "
                "then return here to convert normals to world frame.")
            return

        self.lino_scene.set_progress("Starting world-frame conversion …", 0)
        self.convert_worker = WorldFrameConverter(run_dir, colmap_sparse)
        self.convert_worker.progress.connect(
            lambda msg, pct: self.lino_scene.set_progress(msg, pct))
        self.convert_worker.group_done.connect(self._on_convert_group_done)
        self.convert_worker.finished.connect(self._on_convert_finished)
        self.convert_worker.start()

    def _on_convert_group_done(self, group_idx, world_path):
        self.lino_scene.update_card_world_normal(group_idx, world_path)

    def _on_convert_finished(self, success, message):
        if success:
            self.lino_scene.set_progress("World-frame conversion complete!", 100)
            # Auto-switch grid to world mode
            self.lino_scene.set_normal_mode("world")
        else:
            msg = message or "Unknown error"
            if "Cancelled" not in msg:
                QMessageBox.critical(self, "Conversion Error",
                                     f"World-frame conversion failed:\n{msg}")
            self.lino_scene.set_progress("Conversion failed.", 0)

    def _lino_pause(self):
        if hasattr(self, 'lino_worker') and self.lino_worker and self.lino_worker.isRunning():
            self.lino_worker.pause()

    def _lino_resume(self):
        if hasattr(self, 'lino_worker') and self.lino_worker and self.lino_worker.isRunning():
            self.lino_worker.resume()

    def _lino_cancel(self):
        if hasattr(self, 'lino_worker') and self.lino_worker and self.lino_worker.isRunning():
            self.lino_worker.cancel()

    def _sync_mask_edits_from_view(self):
        idx = self.active_detail_group
        if idx >= 0 and self.mask_scene.manual_mask_np is not None:
            cv2.imwrite(self.manual_edits_paths[idx], self.mask_scene.manual_mask_np)
            
            preview_h = self.preview_maxes[idx].shape[0]
            preview_w = self.preview_maxes[idx].shape[1]
            
            self.preview_manual_edits[idx] = cv2.resize(
                self.mask_scene.manual_mask_np, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST
            )
            self._recalc_hard_mask(idx)

    # ==========================================
    # MASK DETAIL EDITOR (LASSO / SCISSORS)
    # ==========================================
    def open_mask_detail_view(self, idx):
        self.active_detail_group = idx
        self.lbl_mask_detail_title.setText(f"<h2>Angle {idx+1:02d}</h2>")
        
        self.btn_mask_prev.setVisible(idx > 0)
        self.btn_mask_next.setVisible(idx < len(self.max_images_paths) - 1)
        
        self.chk_local_mask_thresh.blockSignals(True)
        if idx in self.mask_thresholds: 
            self.chk_local_mask_thresh.setChecked(True)
            self.local_mask_thresh_container.setEnabled(True)
            self.local_mask_slider.setValue(int(self.mask_thresholds[idx] * 100))
            self.lbl_lmask_val.setText(f"{self.mask_thresholds[idx]:.2f}")
        else: 
            self.chk_local_mask_thresh.setChecked(False)
            self.local_mask_thresh_container.setEnabled(False)
            self.local_mask_slider.setValue(int(self.global_mask_threshold * 100))
            self.lbl_lmask_val.setText(f"{self.global_mask_threshold:.2f}")
        self.chk_local_mask_thresh.blockSignals(False)
        
        max_img_np = cv2.cvtColor(cv2.imread(self.max_images_paths[idx]), cv2.COLOR_BGR2RGB)
        
        if idx in self.soft_masks_paths:
            soft_mask_np = cv2.imread(self.soft_masks_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            soft_mask_np = np.zeros(max_img_np.shape[:2], dtype=np.float32)
            
        manual_mask_np = cv2.imread(self.manual_edits_paths[idx], cv2.IMREAD_GRAYSCALE)
        thresh = self.mask_thresholds.get(idx, self.global_mask_threshold)

        self.mask_scene.set_data(idx, max_img_np, soft_mask_np, manual_mask_np, thresh)
        self.mask_view.fitInView(self.mask_scene.image_rect, Qt.KeepAspectRatio)
        
        self.mask_stack.setCurrentIndex(1)
        
    def close_mask_detail_view(self):
        self._sync_mask_edits_from_view()
        self.active_detail_group = -1
        self.render_mask_grid()
        self.mask_stack.setCurrentIndex(0)
        
    def view_mask_prev(self):
        if self.active_detail_group > 0: 
            self._sync_mask_edits_from_view()
            self.open_mask_detail_view(self.active_detail_group - 1)
            
    def view_mask_next(self):
        if self.active_detail_group < len(self.max_images_paths) - 1: 
            self._sync_mask_edits_from_view()
            self.open_mask_detail_view(self.active_detail_group + 1)

    def update_mask_detail(self):
        idx = self.active_detail_group
        if idx not in self.preview_maxes: 
            return
        self.mask_scene.update_mask_visuals()

    def on_local_mask_override_toggled(self, checked):
        idx = self.active_detail_group
        self.local_mask_thresh_container.setEnabled(checked)
        if checked: 
            val = self.local_mask_slider.value() / 100.0
            self.mask_thresholds[idx] = val
        else:
            if idx in self.mask_thresholds: 
                del self.mask_thresholds[idx]
                
        self.mask_scene.threshold = self.mask_thresholds.get(idx, self.global_mask_threshold)
        self.mask_scene.update_mask_visuals()

    def on_local_mask_thresh_changed(self, value):
        idx = self.active_detail_group
        val = value / 100.0
        self.lbl_lmask_val.setText(f"{val:.2f}")
        
        if self.chk_local_mask_thresh.isChecked(): 
            self.mask_thresholds[idx] = val
            self.mask_scene.threshold = val
            self.mask_scene.update_mask_visuals()

    def on_mask_edited(self, idx, new_manual_mask):
        preview_h = self.preview_maxes[idx].shape[0]
        preview_w = self.preview_maxes[idx].shape[1]
        
        self.preview_manual_edits[idx] = cv2.resize(
            new_manual_mask, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST
        )
        self._recalc_hard_mask(idx)

    def clear_manual_edits(self):
        idx = self.active_detail_group
        if self.mask_scene.manual_mask_np is not None:
            h = self.mask_scene.manual_mask_np.shape[0]
            w = self.mask_scene.manual_mask_np.shape[1]
            cleared = np.full((h, w), 128, dtype=np.uint8)
            
            self.mask_scene.manual_mask_np = cleared
            self.mask_scene.state_manager.push("Clear Manual Edits", [], None, cleared)
            self.mask_scene.update_mask_visuals()
        self.on_mask_edited(idx, cleared)
