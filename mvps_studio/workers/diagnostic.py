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



class DiagnosticWorker(QThread):
    progress = Signal(int, int)
    group_ready = Signal(int, int, QImage)
    finished_loading = Signal()

    def __init__(self, groups, dirty_indices, target_res, global_settings, overrides, image_cache, mags_cache, cache_dir, use_disk):
        super().__init__()
        self.groups = groups
        self.dirty_indices = sorted(list(dirty_indices))
        self.target_res = target_res
        self.settings = global_settings
        self.overrides = overrides
        self.image_cache = image_cache
        self.mags_cache = mags_cache
        self.cache_dir = cache_dir
        self.use_disk = use_disk
        self.is_interrupted = False

    def run(self):
        for i_iter, idx in enumerate(self.dirty_indices):
            if self.is_interrupted: 
                break
                
            s = self.overrides.get(idx, self.settings)
            paths = self.groups[idx]
            num_images = len(paths)
            
            missing_paths = []
            for p in paths:
                if (p, self.target_res, s['blur']) not in self.mags_cache:
                    missing_paths.append(p)
                    
            if missing_paths:
                images_gray = []
                for p in missing_paths:
                    if p in self.image_cache:
                        rgb = self.image_cache[p]['rgb']
                        images_gray.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
                        
                if not images_gray: 
                    continue
                    
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
                    if self.use_disk:
                        safe_name = hashlib.md5(p.encode('utf-8')).hexdigest()
                        cache_path = os.path.join(self.cache_dir, f"{safe_name}_{self.target_res}_{s['blur']}.npy")
                        np.save(cache_path, all_mags_missing[mi])
                        self.mags_cache[(p, self.target_res, s['blur'])] = cache_path
                    else:
                        self.mags_cache[(p, self.target_res, s['blur'])] = all_mags_missing[mi]
                        
            try:
                if self.use_disk:
                    all_mags = np.stack([np.load(self.mags_cache[(p, self.target_res, s['blur'])]) for p in paths])
                else:
                    all_mags = np.stack([self.mags_cache[(p, self.target_res, s['blur'])] for p in paths])
            except KeyError:
                continue 
                
            h = all_mags.shape[1]
            w = all_mags.shape[2]
            min_consensus = max(1, int(num_images * (s['consensus'] / 100.0)))
            
            frame_colors = np.zeros((num_images, 3), dtype=np.uint8)
            hue_step = 300 / (num_images - 1) if num_images > 1 else 0
            for i in range(num_images):
                color_hsv = np.uint8([[[int((i * hue_step) / 2), 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
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
                    
            bytes_per_line = 3 * w
            q_img = QImage(out_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            if not self.is_interrupted:
                self.group_ready.emit(idx, num_images, q_img)
                self.progress.emit(i_iter + 1, len(self.dirty_indices))
                
        if not self.is_interrupted:
            self.finished_loading.emit()

