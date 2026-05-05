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



class StateManager:
    def __init__(self, history_list_widget, scene):
        self.history = []
        self.list_widget = history_list_widget
        self.scene = scene

    def push(self, action_name, segments, active_seed, manual_mask_np, active_polygon=None):
        state = {
            "name": action_name,
            "segments": [seg.copy() for seg in segments],
            "active_seed": QPointF(active_seed) if active_seed else None,
            "mask": manual_mask_np.copy() if manual_mask_np is not None else None,
            "active_polygon": QPolygonF(active_polygon) if active_polygon else None
        }
        self.history.append(state)
        self.list_widget.addItem(action_name)
        self.list_widget.setCurrentRow(len(self.history) - 1)
        self.list_widget.scrollToBottom()

    def goto_index(self, index):
        if index < 0 or index >= len(self.history): 
            return False
            
        while len(self.history) - 1 > index:
            self.history.pop()
            self.list_widget.takeItem(self.list_widget.count() - 1)
            
        prev_state = self.history[-1]
        self.scene.committed_segments = [seg.copy() for seg in prev_state["segments"]]
        self.scene.uncommitted_segments = []
        self.scene.active_seed = prev_state["active_seed"]
        self.scene.active_polygon = QPolygonF(prev_state["active_polygon"]) if prev_state["active_polygon"] else None
        
        if prev_state["mask"] is not None:
            self.scene.manual_mask_np = prev_state["mask"].copy()
            self.scene.update_mask_visuals()
            self.scene.maskEdited.emit(self.scene.group_idx, self.scene.manual_mask_np)
            
        if self.scene.active_seed:
            self.scene.backend.set_seed(int(self.scene.active_seed.x()), int(self.scene.active_seed.y()))
            
        self.scene.current_preview = []
        self.scene.suspend_auto_anchor = True
        self.scene.invalidate()
        self.list_widget.setCurrentRow(index)
        return True

    def undo(self):
        if len(self.history) > 1:
            return self.goto_index(len(self.history) - 2)
        return False


class MaskingScene(QGraphicsScene):
    maskEdited = Signal(int, np.ndarray)

    def __init__(self, backend, history_widget, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.state_manager = StateManager(history_widget, self)
        
        self.group_idx = -1
        self.manual_mask_np = None
        self.soft_mask_np = None
        self.threshold = 0.5
        self.max_img_shape = None
        self._last_img_hash = None
        
        self.committed_segments = []
        self.uncommitted_segments = [] 
        self.current_preview = []
        
        self.active_seed = None
        self.active_polygon = None
        self.last_mouse_pos = None
        
        self.is_mouse_down = False
        self.is_dragging = False
        self.is_panning = False
        
        self.auto_anchor_threshold = 150
        self.suspend_auto_anchor = False 
        
        self.image_rect = QRectF() 
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.addItem(self.pixmap_item)
        
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(0.5) 
        self.mask_visible = True
        self.addItem(self.mask_item)
        
        self.preview_pen = QPen(QColor(0, 255, 0, 180), 2)
        self.preview_pen.setCosmetic(True) 
        
        self.committed_pen = QPen(QColor(255, 0, 0, 255), 2)
        self.committed_pen.setCosmetic(True)
        
        self.poly_pen = QPen(QColor(0, 200, 255, 255), 2, Qt.DashLine)
        self.poly_pen.setCosmetic(True)
        
        self.snap_reticle_pen = QPen(QColor(255, 255, 0, 255), 2)
        self.snap_reticle_pen.setCosmetic(True)

    def set_data(self, group_idx, max_img_np, soft_mask_np, manual_mask_np, threshold):
        is_new_image = (self.group_idx != group_idx)
        if is_new_image:
            self.clear_state()
            self.state_manager.history.clear()
            self.state_manager.list_widget.clear()
            
        self.group_idx = group_idx
        self.manual_mask_np = manual_mask_np.copy()
        self.soft_mask_np = soft_mask_np
        self.threshold = threshold
        self.max_img_shape = max_img_np.shape[:2]
        self._cached_img = max_img_np  # Save for deferred activation
        self.magnetic_scissors_active = False # Flag
        
        h = max_img_np.shape[0]
        w = max_img_np.shape[1]
        
        qimage = QImage(max_img_np.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.image_rect = self.pixmap_item.boundingRect()
        
        padding = 2000 
        self.setSceneRect(-padding, -padding, self.image_rect.width() + padding*2, self.image_rect.height() + padding*2)
        
        self.update_mask_visuals()
        
        if is_new_image:
            self.state_manager.push("Image Loaded", [], None, self.manual_mask_np)
            
        self.update()

    def activate_magnetic_scissors(self):
        if not self.magnetic_scissors_active and hasattr(self, '_cached_img') and self._cached_img is not None:
            # Now build the cost graphs!
            img_hash = self._cached_img[::20, ::20].copy()
            if self.backend.image_shape != self.max_img_shape or self._last_img_hash is None or not np.array_equal(self._last_img_hash, img_hash):
                self.backend.load_image(self._cached_img)
                self._last_img_hash = img_hash
            self.magnetic_scissors_active = True
            
            # We can clear the cached image if we want to save ram, but it's passed by reference anyway
            self._cached_img = None
            return True
        return False

    def update_mask_visuals(self):
        if self.soft_mask_np is None or self.manual_mask_np is None: 
            return
            
        h = self.max_img_shape[0]
        w = self.max_img_shape[1]
        
        base_hard = (self.soft_mask_np > self.threshold).astype(np.uint8) * 255
        final_mask = np.where(self.manual_mask_np == 255, 255, base_hard)
        final_mask = np.where(self.manual_mask_np == 0, 0, final_mask)
        
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = final_mask == 255
        
        # Red tint on background to emulate standard rubylith masking behavior
        overlay[~mask_bool] = [255, 0, 0, 120] 
        
        mask_qimage = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.mask_item.setPixmap(QPixmap.fromImage(mask_qimage))
        
    def set_mask_visible(self, state):
        self.mask_visible = state
        self.mask_item.setVisible(state)

    def toggle_mask(self):
        self.set_mask_visible(not self.mask_visible)

    def clear_state(self):
        self.committed_segments = []
        self.uncommitted_segments = []
        self.current_preview = []
        self.active_seed = None
        self.active_polygon = None
        self.suspend_auto_anchor = False
        self.invalidate()

    def get_snap_target(self, pos):
        if not self.committed_segments: 
            return None
            
        first_pt = self.committed_segments[0][0]
        scale = self.views()[0].transform().m11() if self.views() else 1.0
        snap_radius = 15.0 / scale 
        
        dx = pos.x() - first_pt.x()
        dy = pos.y() - first_pt.y()
        
        if (dx * dx + dy * dy) < (snap_radius * snap_radius): 
            return first_pt
            
        return None

    def close_loop(self):
        if not self.committed_segments: 
            return
            
        poly = QPolygonF()
        for seg in self.committed_segments:
            for pt in seg: 
                poly.append(pt)
                
        poly.append(self.committed_segments[0][0])
        
        self.active_polygon = poly
        self.committed_segments = []
        self.uncommitted_segments = []
        self.current_preview = []
        self.active_seed = None
        
        self.invalidate()
        self.state_manager.push("Loop Closed", self.committed_segments, self.active_seed, self.manual_mask_np, self.active_polygon)

    def apply_boolean(self, operation="add"):
        if not self.active_polygon or self.manual_mask_np is None: 
            return
            
        pts = []
        for i in range(self.active_polygon.count()):
            pt = self.active_polygon.at(i)
            pts.append([int(pt.x()), int(pt.y())])
            
        pts = np.array([pts], dtype=np.int32)
        
        if operation == "add":
            cv2.fillPoly(self.manual_mask_np, pts, 255)
            action_name = "Add Selection to Mask"
        elif operation == "subtract":
            cv2.fillPoly(self.manual_mask_np, pts, 0)
            action_name = "Subtract Selection from Mask"
            
        self.active_polygon = None
        self.update_mask_visuals()
        self.invalidate()
        
        self.maskEdited.emit(self.group_idx, self.manual_mask_np)
        self.state_manager.push(action_name, self.committed_segments, self.active_seed, self.manual_mask_np)

    def invert_mask(self):
        if self.manual_mask_np is None: 
            return
            
        self.manual_mask_np = np.where(self.manual_mask_np == 255, 0, np.where(self.manual_mask_np == 0, 255, self.manual_mask_np))
        self.update_mask_visuals()
        self.maskEdited.emit(self.group_idx, self.manual_mask_np)
        self.state_manager.push("Invert Manual Edits", self.committed_segments, self.active_seed, self.manual_mask_np)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.active_seed:
            self.committed_segments.extend(self.uncommitted_segments)
            self.uncommitted_segments.clear()
            self.committed_segments.append(self.current_preview)
            self.close_loop()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            menu = QMenu()
            if self.active_polygon:
                add_act = menu.addAction("➕ Add Selection to Mask (Ctrl+A)")
                sub_act = menu.addAction("➖ Subtract Selection from Mask (Ctrl+S)")
                menu.addSeparator()
                
            cancel_act = menu.addAction("❌ Cancel Selection (Esc)")
            
            action = menu.exec(event.screenPos())
            if self.active_polygon:
                if action == add_act: 
                    self.apply_boolean("add")
                elif action == sub_act: 
                    self.apply_boolean("subtract")
                    
            if action == cancel_act: 
                self.clear_state()
            return
            
        if event.button() == Qt.LeftButton:
            if self.active_polygon:
                self.active_polygon = None
                self.invalidate()
                
            pos = event.scenePos()
            if not self.image_rect.isNull():
                pos.setX(max(0, min(pos.x(), self.image_rect.width() - 1)))
                pos.setY(max(0, min(pos.y(), self.image_rect.height() - 1)))
                
            snap_pt = self.get_snap_target(pos)
            if snap_pt:
                if self.active_seed and self.current_preview:
                    self.committed_segments.extend(self.uncommitted_segments)
                    self.uncommitted_segments.clear()
                    self.committed_segments.append(self.current_preview)
                self.close_loop()
                return
                
            if self.active_seed and self.current_preview:
                path_end = self.current_preview[-1]
                self.committed_segments.extend(self.uncommitted_segments)
                self.uncommitted_segments.clear()
                self.committed_segments.append(self.current_preview)
                self.state_manager.push("Add Anchor", self.committed_segments, path_end, self.manual_mask_np)
                
                self.active_seed = path_end
                self.current_preview = [path_end]
            else: 
                self.active_seed = pos
                self.current_preview = [pos]
                
            self.is_mouse_down = True
            self.is_dragging = False
            self.suspend_auto_anchor = False
            self.invalidate()
            
        super().mousePressEvent(event)

    def update_cursor_pos(self, pos, modifiers):
        if not self.active_seed or self.active_polygon: 
            return
            
        if not self.image_rect.isNull():
            pos.setX(max(0, min(pos.x(), self.image_rect.width() - 1)))
            pos.setY(max(0, min(pos.y(), self.image_rect.height() - 1)))
            
        self.last_mouse_pos = pos
        snap_pt = self.get_snap_target(pos)
        
        if snap_pt: 
            pos = snap_pt
            
        if self.is_mouse_down:
            self.is_dragging = True
            self.current_preview.append(pos)
        else:
            if (modifiers & Qt.ControlModifier) and self.magnetic_scissors_active: 
                self.backend.set_seed(int(self.active_seed.x()), int(self.active_seed.y()))
                path = self.backend.get_path(int(pos.x()), int(pos.y()))
                
                if path:
                    self.current_preview = path
                    dist = ((pos.x() - self.active_seed.x())**2 + (pos.y() - self.active_seed.y())**2)**0.5
                    if self.suspend_auto_anchor:
                        if dist < 30: 
                            self.suspend_auto_anchor = False
                    elif dist > self.auto_anchor_threshold and not snap_pt: 
                        anchor_idx = int(len(path) * 0.75) 
                        if anchor_idx > 0:
                            auto_anchor_pt = path[anchor_idx]
                            self.uncommitted_segments.append(path[:anchor_idx+1])
                            self.active_seed = auto_anchor_pt
                            self.backend.set_seed(int(self.active_seed.x()), int(self.active_seed.y()))
            else: 
                self.current_preview = [self.active_seed, pos]
                
        self.invalidate()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.update_cursor_pos(event.scenePos(), QApplication.keyboardModifiers())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_mouse_down = False
            if self.is_dragging:
                if len(self.current_preview) > 1 and self.current_preview[0] != self.current_preview[-1]:
                    self.committed_segments.append(self.current_preview)
                    self.active_seed = self.current_preview[-1]
                    self.current_preview = [self.active_seed] 
                    self.state_manager.push("Freehand Segment", self.committed_segments, self.active_seed, self.manual_mask_np)
                    
            if self.active_seed and self.magnetic_scissors_active: 
                self.backend.set_seed(int(self.active_seed.x()), int(self.active_seed.y()))
                
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        if self.active_polygon:
            painter.setPen(self.poly_pen)
            painter.setBrush(QColor(0, 200, 255, 60)) 
            painter.drawPolygon(self.active_polygon)
            return 
            
        painter.setPen(self.committed_pen)
        for segment in self.committed_segments:
            if len(segment) > 1: 
                painter.drawPolyline(segment)
                
        painter.setPen(self.preview_pen)
        for segment in self.uncommitted_segments:
            if len(segment) > 1: 
                painter.drawPolyline(segment)
                
        if not self.is_panning and self.current_preview and len(self.current_preview) > 1:
            painter.drawPolyline(self.current_preview)
            
        if self.last_mouse_pos and self.get_snap_target(self.last_mouse_pos):
            scale = self.views()[0].transform().m11() if self.views() else 1.0
            radius = 8.0 / scale
            painter.setPen(self.snap_reticle_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.last_mouse_pos, radius, radius)

