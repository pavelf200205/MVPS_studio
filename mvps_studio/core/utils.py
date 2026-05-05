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



class ProfileTimer:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        print(f"[Profile] {self.name}: {elapsed:.2f} ms")


def natsort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_image_fast(path, target_res):
    reader = QImageReader(path)
    size = reader.size()
    w = size.width()
    h = size.height()
    
    reduction_flag = cv2.IMREAD_COLOR
    if w > 0 and h > 0 and target_res < 99999:
        scale = min(target_res / w, target_res / h)
        if scale <= 0.125: 
            reduction_flag = getattr(cv2, 'IMREAD_REDUCED_COLOR_8', 65)
        elif scale <= 0.25: 
            reduction_flag = getattr(cv2, 'IMREAD_REDUCED_COLOR_4', 33)
        elif scale <= 0.5: 
            reduction_flag = getattr(cv2, 'IMREAD_REDUCED_COLOR_2', 17)
            
    img = cv2.imread(path, reduction_flag)
    if img is None: 
        return path, None, None
        
    curr_h = img.shape[0]
    curr_w = img.shape[1]
    scale = min(target_res / curr_w, target_res / curr_h)
    
    if scale < 1.0:
        img = cv2.resize(img, (int(curr_w * scale), int(curr_h * scale)), interpolation=cv2.INTER_AREA)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = img_rgb.shape[0]
    w = img_rgb.shape[1]
    
    q_img = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return path, img_rgb, q_img


def numpy_to_qpixmap(img_np):
    if len(img_np.shape) == 2:
        h = img_np.shape[0]
        w = img_np.shape[1]
        q_img = QImage(img_np.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        h = img_np.shape[0]
        w = img_np.shape[1]
        c = img_np.shape[2]
        if c == 3:
            q_img = QImage(img_np.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        else:
            q_img = QImage(img_np.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            
    return QPixmap.fromImage(q_img)

