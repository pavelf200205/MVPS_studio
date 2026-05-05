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



class InSPyReNetWorker(QThread):
    progress = Signal(int, int)
    mask_ready = Signal(int, str) 
    finished = Signal()

    def __init__(self, max_images_paths, cache_dir, inference_res=1024):
        super().__init__()
        self.max_images_paths = max_images_paths 
        self.cache_dir = cache_dir
        self.inference_res = inference_res 
        self.is_interrupted = False

    def run(self):
        if not HAS_INSPYRENET:
            self.finished.emit()
            return
            
        total = len(self.max_images_paths)
        completed = 0
        
        try:
            remover = Remover(mode='base')
        except Exception as e:
            print(f"Error initializing transparent-background: {e}")
            self.finished.emit()
            return
            
        for idx, path in self.max_images_paths.items():
            if self.is_interrupted: 
                break
            try:
                pil_img = PILImage.open(path).convert('RGB')
                orig_w = pil_img.size[0]
                orig_h = pil_img.size[1]
                
                if self.inference_res > 0:
                    scale = min(self.inference_res / orig_w, self.inference_res / orig_h)
                    if scale < 1.0:
                        pil_img = pil_img.resize((int(orig_w * scale), int(orig_h * scale)), resample=PILImage.BILINEAR)
                        
                out_map = remover.process(pil_img, type='map')
                soft_mask = np.array(out_map, dtype=np.float32) / 255.0
                
                if len(soft_mask.shape) > 2:
                    if soft_mask.shape[-1] == 4:
                        soft_mask = soft_mask[:, :, 3] 
                    else:
                        soft_mask = np.mean(soft_mask, axis=-1) 
                        
                if self.inference_res > 0 and (soft_mask.shape[1] != orig_w or soft_mask.shape[0] != orig_h):
                    soft_mask = cv2.resize(soft_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    
                soft_mask_uint8 = (soft_mask * 255).astype(np.uint8)
                mask_path = os.path.join(self.cache_dir, f"soft_mask_{idx}.webp")
                cv2.imwrite(mask_path, soft_mask_uint8, [cv2.IMWRITE_WEBP_QUALITY, 85])
                self.mask_ready.emit(idx, mask_path)
                
            except Exception as e:
                print(f"Failed processing mask for {path}: {e}")
                
            completed += 1
            self.progress.emit(completed, total)
            
        self.finished.emit()

