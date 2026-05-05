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



class MagneticScissorsBackend:
    def __init__(self):
        self.full_cost_map = None
        self.local_predecessors = None
        self.image_shape = None
        self.current_seed = None
        self.window_size = 600 
        self.window_rect = None 
        
    def load_image(self, image_array):
        self.image_shape = image_array.shape[:2]
        self.current_seed = None
        
        img_cpu = np.asarray(image_array, dtype=np.float32)
        if len(img_cpu.shape) == 3:
            img_cpu = np.mean(img_cpu, axis=2)
            
        smoothed = ndi.gaussian_filter(img_cpu, sigma=2.0)
        grad_y = ndi.sobel(smoothed, axis=0)
        grad_x = ndi.sobel(smoothed, axis=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        max_mag = np.max(magnitude)
        if max_mag > 0: 
            magnitude /= max_mag
            
        laplacian = ndi.laplace(smoothed)
        zero_cross_penalty = np.abs(laplacian)
        max_lap = np.max(zero_cross_penalty)
        if max_lap > 0: 
            zero_cross_penalty /= max_lap

        self.full_cost_map = (0.7 * (1.0 - magnitude)) + (0.3 * zero_cross_penalty) + 0.01
        
    def set_seed(self, x, y):
        if self.full_cost_map is None or self.current_seed == (x, y):
            return
            
        h = self.image_shape[0]
        w = self.image_shape[1]
        half_w = self.window_size // 2
        
        x_min = max(0, x - half_w)
        x_max = min(w, x + half_w)
        y_min = max(0, y - half_w)
        y_max = min(h, y + half_w)
        
        self.window_rect = (x_min, y_min, x_max, y_max)
        local_cost = self.full_cost_map[y_min:y_max, x_min:x_max]
        
        lh = local_cost.shape[0]
        lw = local_cost.shape[1]
        num_nodes = lh * lw
        nodes = np.arange(num_nodes, dtype=np.int32).reshape(lh, lw)
        
        h_src = nodes[:, :-1].ravel()
        h_dst = nodes[:, 1:].ravel()
        h_cost = (local_cost[:, :-1] + local_cost[:, 1:]).ravel() * 0.5
        
        v_src = nodes[:-1, :].ravel()
        v_dst = nodes[1:, :].ravel()
        v_cost = (local_cost[:-1, :] + local_cost[1:, :]).ravel() * 0.5
        
        d1_src = nodes[:-1, :-1].ravel()
        d1_dst = nodes[1:, 1:].ravel()
        d1_cost = (local_cost[:-1, :-1] + local_cost[1:, 1:]).ravel() * 0.7071
        
        d2_src = nodes[:-1, 1:].ravel()
        d2_dst = nodes[1:, :-1].ravel()
        d2_cost = (local_cost[:-1, 1:] + local_cost[1:, :-1]).ravel() * 0.7071
        
        src = np.concatenate([h_src, h_dst, v_src, v_dst, d1_src, d1_dst, d2_src, d2_dst])
        dst = np.concatenate([h_dst, h_src, v_dst, v_src, d1_dst, d1_src, d2_dst, d2_src])
        weights = np.concatenate([h_cost, h_cost, v_cost, v_cost, d1_cost, d1_cost, d2_cost, d2_cost])
        
        local_graph = coo_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
        
        local_seed_node = int((y - y_min) * lw + (x - x_min))
        _, self.local_predecessors = dijkstra(local_graph, directed=False, indices=local_seed_node, return_predecessors=True)
        self.current_seed = (x, y)
        
    def get_path(self, target_x, target_y):
        if self.local_predecessors is None: 
            return []
            
        x_min = self.window_rect[0]
        y_min = self.window_rect[1]
        x_max = self.window_rect[2]
        y_max = self.window_rect[3]
        
        target_x = max(x_min, min(target_x, x_max - 1))
        target_y = max(y_min, min(target_y, y_max - 1))
        
        lh = y_max - y_min
        lw = x_max - x_min
        curr = int((target_y - y_min) * lw + (target_x - x_min))
        
        path = []
        count = 0
        limit = lh * lw
        
        while curr >= 0 and count < limit:
            path.append(QPointF(curr % lw + x_min, curr // lw + y_min))
            next_curr = self.local_predecessors[curr]
            if next_curr == curr: 
                break
            curr = next_curr
            count += 1
            
        return path[::-1]

