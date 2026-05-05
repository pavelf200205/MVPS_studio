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



class SettingsPanel(QWidget):
    settings_changed = Signal()
    
    def __init__(self, title, is_override=False):
        super().__init__()
        self.is_override = is_override
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        
        self.chk_override = None
        if is_override:
            self.chk_override = QCheckBox("Override Global Settings")
            self.chk_override.toggled.connect(self._toggle_override)
            layout.addWidget(self.chk_override)
            
        self.controls_widget = QWidget()
        ctrl_layout = QVBoxLayout(self.controls_widget)
        ctrl_layout.setContentsMargins(0, 10, 0, 0)
        
        self.chk_white = QCheckBox("Show Static Background (White)")
        self.chk_white.setChecked(False)
        self.blur_slider, self.blur_val = self._make_slider("Pre-Blur", 0, 6, 0, "px", instant=False)
        self.thresh_slider, self.thresh_val = self._make_slider("Edge Threshold", 10, 250, 65, instant=True)
        self.cons_slider, self.cons_val = self._make_slider("Consensus Req.", 10, 100, 30, "%", instant=True)
        
        for w in [self.chk_white, self.blur_slider, self.thresh_slider, self.cons_slider]: 
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
        ctrl_layout.addWidget(self.chk_white)
        ctrl_layout.addWidget(self.blur_slider.parentWidget())
        ctrl_layout.addWidget(self.thresh_slider.parentWidget())
        ctrl_layout.addWidget(self.cons_slider.parentWidget())
        
        layout.addWidget(self.controls_widget)
        self.chk_white.toggled.connect(lambda _: self.settings_changed.emit())
        
        if is_override: 
            self.controls_widget.setEnabled(False)

    def _make_slider(self, name, min_v, max_v, default, suffix="", instant=True):
        container = QWidget(self) 
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        header = QHBoxLayout()
        lbl_val = QLabel(f"<b>{default}{suffix}</b>")
        header.addWidget(QLabel(name))
        header.addStretch()
        header.addWidget(lbl_val)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        
        def on_change(v): 
            lbl_val.setText(f"<b>{v}{suffix}</b>")
            if instant: 
                self.settings_changed.emit()
                
        slider.valueChanged.connect(on_change)
        if not instant: 
            slider.sliderReleased.connect(self.settings_changed.emit)
            
        layout.addLayout(header)
        layout.addWidget(slider)
        return slider, lbl_val

    def _toggle_override(self, state):
        self.controls_widget.setEnabled(state)
        self.settings_changed.emit()

    def get_settings(self):
        return {
            'show_white': self.chk_white.isChecked(), 
            'blur': self.blur_slider.value(), 
            'threshold': self.thresh_slider.value(), 
            'consensus': self.cons_slider.value()
        }

    def set_settings(self, s):
        for w, k in [(self.chk_white, 'show_white'), (self.blur_slider, 'blur'), (self.thresh_slider, 'threshold'), (self.cons_slider, 'consensus')]:
            w.blockSignals(True)
            if k == 'show_white': 
                w.setChecked(s[k])
            else: 
                w.setValue(s[k])
            w.blockSignals(False)
            
        self.blur_val.setText(f"<b>{s['blur']}px</b>")
        self.thresh_val.setText(f"<b>{s['threshold']}</b>")
        self.cons_val.setText(f"<b>{s['consensus']}%</b>")

