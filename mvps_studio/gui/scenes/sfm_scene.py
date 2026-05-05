import os
import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QProgressBar, QMessageBox, QFrame, QSplitter, QSlider
)
from PySide6.QtCore import Qt, QPoint, Signal, QPointF
from PySide6.QtGui import QVector3D, QMatrix4x4, QQuaternion, QVector4D

# 1. ARC BALL VIEWPORT
class ArcballGLViewWidget(gl.GLViewWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_rot = QQuaternion()
        self.mousePos = QPoint()
        self.last_arcball_vec = None
        self.opts['distance'] = 100.0
        self.opts['center'] = QVector3D(0, 0, 0)
        self.camera_view_active = False 

    def viewMatrix(self):
        view = QMatrix4x4()
        d = self.opts.get('distance', 100.0)
        view.translate(0, 0, -d)
        view.rotate(self.q_rot)
        c = self.opts.get('center', QVector3D(0, 0, 0))
        if not isinstance(c, QVector3D):
            c = QVector3D(c.x(), c.y(), c.z())
        view.translate(-c.x(), -c.y(), -c.z())
        return view

    def _project_to_sphere(self, pos):
        w, h = float(self.width()), float(self.height())
        if w == 0 or h == 0: return QVector3D(0, 0, 1)
        r = min(w, h) / 2.0
        nx, ny = (pos.x() - w / 2.0) / r, (h / 2.0 - pos.y()) / r 
        d = nx*nx + ny*ny
        if d <= 1.0: nz = np.sqrt(1.0 - d)
        else:
            length = np.sqrt(d)
            nx, ny, nz = nx / length, ny / length, 0.0
        return QVector3D(nx, ny, nz)

    def mousePressEvent(self, ev):
        self.mousePos = ev.position().toPoint()
        self.last_arcball_vec = self._project_to_sphere(self.mousePos)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        current_pos = ev.position().toPoint()
        dx = current_pos.x() - self.mousePos.x()
        dy = current_pos.y() - self.mousePos.y()

        if ev.buttons() == Qt.MouseButton.LeftButton:
            current_arcball_vec = self._project_to_sphere(current_pos)
            if self.last_arcball_vec is not None:
                rot = QQuaternion.rotationTo(self.last_arcball_vec, current_arcball_vec)
                self.q_rot = rot * self.q_rot
            self.last_arcball_vec = current_arcball_vec
            self.mousePos = current_pos
            self.update()
            
        elif ev.buttons() == Qt.MouseButton.MiddleButton or ev.buttons() == Qt.MouseButton.RightButton:
            d = self.opts.get('distance', 100.0)
            pan_x, pan_y = -dx * (d / 1000.0), dy * (d / 1000.0)
            rot_mat = QMatrix4x4()
            rot_mat.rotate(self.q_rot.conjugated())
            right = rot_mat.mapVector(QVector3D(1, 0, 0))
            up = rot_mat.mapVector(QVector3D(0, 1, 0))
            delta = (right * pan_x) + (up * pan_y)
            c = self.opts.get('center', QVector3D(0, 0, 0))
            if not isinstance(c, QVector3D):
                self.opts['center'] += type(c)(delta.x(), delta.y(), delta.z())
            else:
                self.opts['center'] = c + delta
            self.mousePos = current_pos
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta > 0: self.opts['distance'] *= 0.85
        elif delta < 0: self.opts['distance'] *= 1.15
        self.update()

# 2. SFM SCENE
class SfmScene(QWidget):
    start_reconstruction = Signal(bool) # Trigger sfm with masks option, use sys_colmap

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter_plot = None
        self.camera_lines = None
        self.cameras_data = None
        self.cam_scale_val = 100
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        side_vbox = QVBoxLayout(sidebar)

        side_vbox.addWidget(QLabel("<h2>Structure from Motion</h2>"))
        side_vbox.addWidget(QLabel("Run COLMAP reconstruction locally using the generated averaged images."))
        
        self.chk_masks = QCheckBox("Use generated masks for feature extraction")
        self.chk_masks.setToolTip("Enable if dataset has tricky backgrounds. Usually disabled for robust generic SfM.")
        side_vbox.addWidget(self.chk_masks)
        
        self.btn_run_sfm = QPushButton("▶ Run COLMAP Reconstruction")
        self.btn_run_sfm.setStyleSheet("background-color: #0d6efd; color: white; font-weight: bold; padding: 12px;")
        self.btn_run_sfm.clicked.connect(lambda: self.start_reconstruction.emit(self.chk_masks.isChecked()))
        side_vbox.addWidget(self.btn_run_sfm)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        side_vbox.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Ready")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        # Tools
        side_vbox.addWidget(QLabel("<b>Render Options</b>"))
        self.chk_grid = QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(lambda c: self.grid_item.setVisible(c))
        side_vbox.addWidget(self.chk_grid)
        
        self.chk_cams = QCheckBox("Show Cameras")
        self.chk_cams.setChecked(True)
        self.chk_cams.toggled.connect(lambda c: self.camera_lines.setVisible(c) if self.camera_lines else None)
        side_vbox.addWidget(self.chk_cams)

        side_vbox.addSpacing(10)
        self.lbl_cam_scale = QLabel("Camera Scale: 1.0x")
        self.slider_cam = QSlider(Qt.Orientation.Horizontal)
        self.slider_cam.setRange(1, 500)
        self.slider_cam.setValue(100)
        self.slider_cam.valueChanged.connect(self._on_cam_slider)
        
        r_cam = QHBoxLayout()
        r_cam.addWidget(self.lbl_cam_scale)
        r_cam.addWidget(self.slider_cam)
        side_vbox.addLayout(r_cam)

        main_layout.addWidget(sidebar)

        # Viewport
        self.view_container = QWidget()
        view_lyt = QVBoxLayout(self.view_container)
        view_lyt.setContentsMargins(0,0,0,0)
        
        self.view = ArcballGLViewWidget()
        self.grid_item = gl.GLGridItem()
        self.view.addItem(self.grid_item)
        view_lyt.addWidget(self.view)
        
        main_layout.addWidget(self.view_container, 1)

    def set_progress(self, msg, val):
        self.lbl_status.setText(msg)
        self.progress_bar.setValue(val)

    def _on_cam_slider(self, val):
        self.cam_scale_val = val
        self.lbl_cam_scale.setText(f"Camera Scale: {val/100.0:.1f}x")
        if self.cameras_data:
            self.render_cameras(self.cameras_data, val / 100.0)

    def load_reconstruction_model(self, sparse_dir):
        if not os.path.exists(sparse_dir):
            QMessageBox.critical(self, "Error", f"Sparse directory does not exist: {sparse_dir}")
            return
            
        try:
            pts_file = os.path.join(sparse_dir, "points3D.txt")
            cams_file = os.path.join(sparse_dir, "images.txt")
            intrinsics_file = os.path.join(sparse_dir, "cameras.txt")

            if not os.path.exists(pts_file):
                QMessageBox.critical(self, "Error", f"Could not find points3D.txt in {sparse_dir}")
                return

            points, colors = self.parse_points3D_txt(pts_file)
            self.render_point_cloud(points, colors)
            
            if os.path.exists(cams_file) and os.path.exists(intrinsics_file):
                self.cameras_data = self.parse_images_txt(cams_file)
                
                # Attempt to calculate optimal frustum scale
                if self.cameras_data and self.scatter_plot:
                    cam_pos = np.array([c[0] for c in self.cameras_data])
                    pc_pos = self.scatter_plot.pos
                    # Median distance from cameras to point cloud
                    dists = []
                    # Sample some cameras to be fast
                    sample_indices = np.random.choice(len(cam_pos), min(len(cam_pos), 10), replace=False)
                    for i in sample_indices:
                        d = np.min(np.linalg.norm(pc_pos - cam_pos[i], axis=1))
                        dists.append(d)
                    
                    median_dist = np.median(dists) if dists else 1.0
                    # Heuristic: frustum size should be about 5% of distance to scene
                    optimal_scale = (median_dist * 0.05) / (self.scene_base_scale)
                    self.slider_cam.setValue(int(optimal_scale * 100))
                
                self.render_cameras(self.cameras_data)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to render 3D model:\\n{str(e)}")

    def parse_points3D_txt(self, file_path):
        points, colors = [], []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split()
                if len(parts) < 7: continue
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([float(parts[4])/255.0, float(parts[5])/255.0, float(parts[6])/255.0, 1.0])
        return np.array(points), np.array(colors)

    def parse_images_txt(self, file_path):
        camera_poses = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split()
                if len(parts) >= 10 and parts[-1].lower().endswith(('.jpg', '.jpeg', '.png')):
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    
                    q = QQuaternion(qw, qx, qy, qz)
                    q_inv = q.conjugated()
                    t = QVector3D(tx, ty, tz)
                    center = q_inv.rotatedVector(-t)
                    
                    camera_poses.append((np.array([center.x(), center.y(), center.z()]), q_inv))
        return camera_poses

    def render_point_cloud(self, points, colors):
        if self.scatter_plot is not None: self.view.removeItem(self.scatter_plot)
        if len(points) == 0: return

        self.scatter_plot = gl.GLScatterPlotItem(pos=points, color=colors, size=2.0, pxMode=True, glOptions='translucent')
        self.view.addItem(self.scatter_plot)
        
        max_dist = np.max(np.linalg.norm(points, axis=1)) if len(points) > 0 else 100
        self.scene_base_scale = max_dist / 30.0 
        self.view.opts['distance'] = max_dist * 1.5

    def render_cameras(self, cameras, scale_mult=None):
        if self.camera_lines is not None: self.view.removeItem(self.camera_lines)
        if not cameras: return

        if scale_mult is None:
            scale_mult = self.cam_scale_val / 100.0

        base_scale = getattr(self, 'scene_base_scale', 100.0 / 30.0)
        scale = base_scale * scale_mult
        
        w, h, f = 1.0 * scale, 0.75 * scale, 1.0 * scale
        lines, line_colors = [], []
        color = [1.0, 0.2, 0.0, 0.8]

        for center_arr, q_inv in cameras:
            center = QVector3D(*center_arr)
            pts_local = [
                QVector3D(0, 0, 0), QVector3D(-w, -h, f),
                QVector3D(w, -h, f), QVector3D(w, h, f),
                QVector3D(-w, h, f)
            ]
            pw = np.array([[(center + q_inv.rotatedVector(p)).x(), 
                            (center + q_inv.rotatedVector(p)).y(), 
                            (center + q_inv.rotatedVector(p)).z()] for p in pts_local])
            
            edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
            for e in edges:
                lines.extend([pw[e[0]], pw[e[1]]])
                line_colors.extend([color, color])

        self.camera_lines = gl.GLLinePlotItem(pos=np.array(lines), color=np.array(line_colors), mode='lines', antialias=True)
        self.view.addItem(self.camera_lines)
