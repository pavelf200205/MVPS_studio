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



class ZoomPanView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.pan_active = False
        self.last_pan_pos = None
        self.auto_pan_margin = 60 
        self._ignore_next_pos = None 

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        if event.angleDelta().y() > 0: 
            self.scale(zoom_in_factor, zoom_in_factor)
        else: 
            self.scale(1 / zoom_in_factor, 1 / zoom_in_factor)
        self.scene().invalidate()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.pan_active = True
            self.last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            if hasattr(self.scene(), 'is_panning'):
                self.scene().is_panning = True
                self.scene().invalidate()
            event.accept()
        else: 
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        if self._ignore_next_pos is not None:
            if abs(pos.x() - self._ignore_next_pos.x()) + abs(pos.y() - self._ignore_next_pos.y()) <= 2:
                self._ignore_next_pos = None
                return 

        if self.pan_active:
            delta = pos - self.last_pan_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.last_pan_pos = pos
            event.accept()
        else:
            super().mouseMoveEvent(event)
            if self.scene() and getattr(self.scene(), 'active_seed', None) is not None:
                viewport_rect = self.viewport().rect()
                margin = self.auto_pan_margin 
                if viewport_rect.contains(pos):
                    safe_rect = viewport_rect.adjusted(margin, margin, -margin, -margin)
                    if not safe_rect.contains(pos):
                        clamped_x = max(safe_rect.left(), min(pos.x(), safe_rect.right()))
                        clamped_y = max(safe_rect.top(), min(pos.y(), safe_rect.bottom()))
                        
                        dx = pos.x() - clamped_x
                        dy = pos.y() - clamped_y
                        
                        h_bar = self.horizontalScrollBar()
                        v_bar = self.verticalScrollBar()
                        
                        old_h = h_bar.value()
                        old_v = v_bar.value()
                        h_bar.setValue(old_h + dx)
                        v_bar.setValue(old_v + dy)
                        actual_dx = h_bar.value() - old_h
                        actual_dy = v_bar.value() - old_v
                        
                        if actual_dx != 0 or actual_dy != 0:
                            warp_x = pos.x() - actual_dx
                            warp_y = pos.y() - actual_dy
                            self._ignore_next_pos = QPoint(warp_x, warp_y)
                            QCursor.setPos(self.viewport().mapToGlobal(QPoint(warp_x, warp_y)))
                            self.scene().update_cursor_pos(self.mapToScene(QPoint(warp_x, warp_y)), QApplication.keyboardModifiers())
                            return
                            
            if self.scene(): 
                self.scene().update_cursor_pos(self.mapToScene(pos), QApplication.keyboardModifiers())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.pan_active = False
            self.setCursor(Qt.ArrowCursor)
            if hasattr(self.scene(), 'is_panning'):
                self.scene().is_panning = False
                self.scene().update_cursor_pos(self.mapToScene(self.mapFromGlobal(QCursor.pos())), QApplication.keyboardModifiers())
                self.scene().invalidate()
            event.accept()
        else: 
            super().mouseReleaseEvent(event)

