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



class HybridGroupCard(QFrame):
    clicked = Signal(int)
    
    def __init__(self, group_idx, paths, target_count, image_cache, consensus_cache, zoom_size=260, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.group_idx = group_idx
        self.paths = paths
        self.target_count = target_count
        self.image_cache = image_cache
        self.consensus_cache = consensus_cache
        self._zoom_size = zoom_size
        self.view_mode = "animation"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.img_label, 1)
        
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        self.setStyleSheet("HybridGroupCard { border: 1px solid #ced4da; background-color: #ffffff; border-radius: 6px; } HybridGroupCard:hover { background-color: #f1f3f5; }")
        if len(self.paths) != self.target_count:
            self.setStyleSheet("HybridGroupCard { border: 2px solid #dc3545; background-color: #ffffff; border-radius: 6px; } HybridGroupCard:hover { background-color: #f8d7da; }")

        self.setFixedSize(self._zoom_size, self._zoom_size + 30)

    def set_zoom(self, zoom_size):
        self._zoom_size = zoom_size
        self.setFixedSize(self._zoom_size, self._zoom_size + 30)
        self.update_view(getattr(self, 'last_tick', 0))

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.update_view(getattr(self, 'last_tick', 0))

    def set_loading(self, text="⏳ Processing..."):
        self.title_label.setText(f"<b>Angle {self.group_idx+1:02d}</b>")
        self.img_label.setText(text)
        self.setStyleSheet("HybridGroupCard { border: 1px solid #ced4da; background-color: #ffffff; border-radius: 6px; }")

    def update_view(self, global_tick):
        self.last_tick = global_tick
        if not self.paths:
            self.title_label.setText(f"<b>Angle {self.group_idx+1:02d}</b> (Empty)")
            self.img_label.setText("No Images")
            return
            
        count_color = "red" if len(self.paths) != self.target_count else "black"
        
        if self.view_mode == "animation":
            local_idx = global_tick % len(self.paths)
            path = self.paths[local_idx]
            self.title_label.setText(f"<b>Angle {self.group_idx+1:02d}</b> - {local_idx+1}/{len(self.paths)} (<span style='color:{count_color};'>{len(self.paths)}</span>)")
            if path in self.image_cache:
                pixmap = self.image_cache[path]['pixmap']
                self.img_label.setPixmap(pixmap.scaled(self._zoom_size - 10, self._zoom_size - 10, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        elif self.view_mode in ["edge", "masking"]:
            self.title_label.setText(f"<b>Angle {self.group_idx+1:02d}</b> (<span style='color:{count_color};'>{len(self.paths)}</span>)")
            if self.group_idx in self.consensus_cache:
                self.img_label.setPixmap(self.consensus_cache[self.group_idx].scaled(self._zoom_size - 10, self._zoom_size - 10, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else: 
                self.img_label.setText("⏳ Processing...")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: 
            self.clicked.emit(self.group_idx)

