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



class MaxImageWorker(QThread):
    progress = Signal(int, int)
    group_max_ready = Signal(int, str, np.ndarray, int, int) 
    finished = Signal()

    def __init__(self, groups, cache_dir):
        super().__init__()
        self.groups = groups
        self.cache_dir = cache_dir
        self.is_interrupted = False

    def _process_single_group(self, args):
        idx, paths = args
        if self.is_interrupted: return None
        
        accumulator = None
        valid_count = 0
        
        for p in paths:
            if self.is_interrupted: return None
            
            img_bgr = cv2.imread(p)
            if img_bgr is not None:
                if accumulator is None:
                    accumulator = img_bgr.copy()
                else:
                    np.maximum(accumulator, img_bgr, out=accumulator)
                valid_count += 1
        
        if valid_count > 0 and not self.is_interrupted:
            max_bgr = accumulator
            h, w = max_bgr.shape[:2]
            
            max_rgb = cv2.cvtColor(max_bgr, cv2.COLOR_BGR2RGB)
            max_path = os.path.join(self.cache_dir, f"max_group_{idx}.jpg")
            cv2.imwrite(max_path, max_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            scale = min(800 / w, 800 / h)
            if scale < 1.0:
                preview_rgb = cv2.resize(max_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                preview_rgb = max_rgb.copy()
                
            return (idx, max_path, preview_rgb, h, w)
        return None

    def run(self):
        total_groups = len(self.groups)
        args_list = [(idx, paths) for idx, paths in enumerate(self.groups)]
        processed_count = 0
        
        # JPEG decode and np.maximum natively drop the GIL in C++
        # Running them in a ThreadPool gives massive multi-core speedup.
        max_workers = min(os.cpu_count() or 4, 8)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_group, arg): arg for arg in args_list}
            
            for future in concurrent.futures.as_completed(futures):
                if self.is_interrupted:
                    for f in futures: f.cancel()
                    break
                    
                res = future.result()
                if res is not None:
                    idx, max_path, preview_rgb, h, w = res
                    self.group_max_ready.emit(idx, max_path, preview_rgb, h, w)
                    
                processed_count += 1
                self.progress.emit(processed_count, total_groups)
                
        self.finished.emit()

