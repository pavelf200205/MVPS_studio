import os
import json
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSizePolicy,
    QGroupBox, QFormLayout, QScrollArea, QColorDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QColor, QQuaternion, QVector3D, QMatrix4x4, QPixmap

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import OpenGL.GL as ogl

from mvps_studio.gui.widgets.settings_panel import SettingsPanel

# ==========================================
# 1. ARCBALL VIEWPORT
# ==========================================
class ArcballGLViewWidget(gl.GLViewWidget):
    gizmo_moved = Signal(QVector3D)
    gizmo_scale_delta = Signal(float)
    gizmo_scale_factor = Signal(float)
    camera_moved = Signal() 
    camera_picked = Signal(QPoint)
    camera_exited_via_interaction = Signal()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_rot = QQuaternion()
        self.mousePos = QPoint()
        self.last_arcball_vec = None
        self.opts['distance'] = 100.0
        self.opts['center'] = QVector3D(0, 0, 0)
        self.gizmo_active = False
        self.gizmo_pos = QVector3D(0, 0, 0)
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

        is_gizmo_interaction = self.gizmo_active and (
            (ev.buttons() == Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)) or
            (ev.buttons() == Qt.MouseButton.RightButton)
        )

        if ev.buttons() != Qt.MouseButton.NoButton and not is_gizmo_interaction:
            if self.camera_view_active and (abs(dx) > 2 or abs(dy) > 2):
                self.camera_view_active = False
                self.camera_exited_via_interaction.emit()

        if ev.buttons() == Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            if self.gizmo_active:
                d = self.opts.get('distance', 100.0)
                pan_x, pan_y = dx * (d / 1000.0), -dy * (d / 1000.0)
                rot_mat = QMatrix4x4()
                rot_mat.rotate(self.q_rot.conjugated())
                right = rot_mat.mapVector(QVector3D(1, 0, 0))
                up = rot_mat.mapVector(QVector3D(0, 1, 0))
                self.gizmo_pos += (right * pan_x) + (up * pan_y)
                self.gizmo_moved.emit(self.gizmo_pos)
            self.mousePos = current_pos
            self.update()

        elif ev.buttons() == Qt.MouseButton.LeftButton:
            current_arcball_vec = self._project_to_sphere(current_pos)
            if self.last_arcball_vec is not None:
                rot = QQuaternion.rotationTo(self.last_arcball_vec, current_arcball_vec)
                self.q_rot = rot * self.q_rot
            self.last_arcball_vec = current_arcball_vec
            self.mousePos = current_pos
            self.update()
            self.camera_moved.emit()
            
        elif ev.buttons() == Qt.MouseButton.MiddleButton:
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
            self.camera_moved.emit()

        elif ev.buttons() == Qt.MouseButton.RightButton:
            if self.gizmo_active:
                self.gizmo_scale_delta.emit(dy)
            self.mousePos = current_pos
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.gizmo_active:
                if delta > 0: self.gizmo_scale_factor.emit(1.05)
                elif delta < 0: self.gizmo_scale_factor.emit(0.95)
            ev.accept()
            return
            
        if self.camera_view_active:
            self.camera_view_active = False
            self.camera_exited_via_interaction.emit()
            
        if delta > 0: self.opts['distance'] *= 0.85
        elif delta < 0: self.opts['distance'] *= 1.15
        self.update()
        self.camera_moved.emit()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.camera_picked.emit(ev.position().toPoint())
        super().mouseDoubleClickEvent(ev)


class NormalizationScene(QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        
        self.cameras_data = None
        self.intrinsics_dict = None
        self.scatter_plot = None
        self.camera_lines = None
        self.sphere_item = None
        self.gizmo_item = None
        
        self.norm_offset = None
        self.norm_max_dist = None
        self.norm_ratio = 3.0
        
        self.cam_scale_val = 100
        self.pt_scale_val = 20
        self.sphere_color = [0.29, 0.0, 0.51] # Deep purple
        self.opacity_val = 30
        
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Left: Viewport ---
        self.view_container = QWidget()
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = ArcballGLViewWidget()
        self.view.setBackgroundColor('black')
        
        # Add basic grid
        gx = gl.GLGridItem()
        gx.scale(10, 10, 1)
        self.view.addItem(gx)
        
        view_layout.addWidget(self.view)
        main_layout.addWidget(self.view_container, 1)
        
        # --- Right: Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sidebar.setFixedWidth(300)
        sl = QVBoxLayout(self.sidebar)
        sl.addWidget(QLabel("<b>Normalization</b>"))
        
        # Controls
        self.btn_load = QPushButton("Load Active SfM Model")
        self.btn_load.clicked.connect(self.load_active_sfm)
        sl.addWidget(self.btn_load)
        
        self.btn_normalize = QPushButton("Compute Normalization")
        self.btn_normalize.clicked.connect(self.run_normalization)
        self.btn_normalize.setEnabled(False)
        sl.addWidget(self.btn_normalize)
        
        sl.addSpacing(10)
        
        # Sliders
        self.lbl_cam_scale = QLabel("Camera Scale: 1.0x")
        self.slider_cam = QSlider(Qt.Orientation.Horizontal)
        self.slider_cam.setRange(1, 300)
        self.slider_cam.setValue(100)
        self.slider_cam.valueChanged.connect(self._on_cam_slider)
        
        r1 = QHBoxLayout()
        r1.addWidget(self.lbl_cam_scale)
        r1.addWidget(self.slider_cam)
        sl.addLayout(r1)
        
        self.lbl_pt_scale = QLabel("Point Size: 2.0")
        self.slider_pt = QSlider(Qt.Orientation.Horizontal)
        self.slider_pt.setRange(1, 100)
        self.slider_pt.setValue(20)
        self.slider_pt.valueChanged.connect(self._on_pt_slider)
        
        r2 = QHBoxLayout()
        r2.addWidget(self.lbl_pt_scale)
        r2.addWidget(self.slider_pt)
        sl.addLayout(r2)
        
        sl.addSpacing(10)
        
        self.btn_export = QPushButton("Confirm Normalization")
        self.btn_export.clicked.connect(self.export_npz)
        self.btn_export.setEnabled(False)
        sl.addWidget(self.btn_export)
        sl.addStretch()
        
        main_layout.addWidget(self.sidebar)
        
        # Signals
        self.view.gizmo_moved.connect(self._on_gizmo_moved)
        self.view.gizmo_scale_delta.connect(self._on_gizmo_scale_delta)
        self.view.gizmo_scale_factor.connect(self._on_gizmo_scale_factor)
        self.view.camera_moved.connect(self.update_normalization_transforms)

    def _on_cam_slider(self, val):
        self.lbl_cam_scale.setText(f"Camera Scale: {val/100.0:.1f}x")
        self.cam_scale_val = val
        if self.cameras_data:
            self.render_cameras(self.cameras_data, val / 100.0)

    def _on_pt_slider(self, val):
        self.lbl_pt_scale.setText(f"Point Size: {val/10.0:.1f}")
        self.pt_scale_val = val
        if self.scatter_plot:
            self.scatter_plot.setData(size=val/10.0)

    def load_active_sfm(self):
        if not self.main_app.colmap_dir:
            QMessageBox.warning(self, "Warning", "No project loaded.")
            return
            
        sfm_dir = os.path.join(self.main_app.colmap_dir, "sparse", "0")
        
        pts_file = os.path.join(sfm_dir, "points3D.txt")
        cams_file = os.path.join(sfm_dir, "images.txt")
        intrinsics_file = os.path.join(sfm_dir, "cameras.txt")

        if not os.path.exists(pts_file):
            QMessageBox.critical(self, "Error", f"Could not find points3D.txt in {sfm_dir}. Please run SfM first.")
            return

        try:
            points, colors = self.parse_points3D(pts_file)
            self.render_point_cloud(points, colors)
            
            if os.path.exists(cams_file) and os.path.exists(intrinsics_file):
                self.intrinsics_dict = self.parse_intrinsics(intrinsics_file)
                self.cameras_data = self.parse_images(cams_file)
                
                self.render_cameras(self.cameras_data, self.cam_scale_val / 100.0)
                self.btn_normalize.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

    def parse_points3D(self, file_path):
        points = []
        colors = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                data = line.split()
                points.append([float(data[1]), float(data[2]), float(data[3])])
                colors.append([float(data[4])/255.0, float(data[5])/255.0, float(data[6])/255.0, 1.0])
        return np.array(points), np.array(colors)

    def parse_intrinsics(self, file_path):
        cameras = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                data = line.strip().split()
                cam_id = int(data[0])
                model = data[1]
                w, h = int(data[2]), int(data[3])
                if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"]:
                    f_val, cx, cy = map(float, data[4:7])
                    K = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]])
                    fov_y = 2 * np.degrees(np.arctan(h / (2 * f_val)))
                elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
                    fx, fy, cx, cy = map(float, data[4:8])
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    fov_y = 2 * np.degrees(np.arctan(h / (2 * fy)))
                cameras[cam_id] = (K, w, h, fov_y, cx, cy)
        return cameras

    def parse_images(self, file_path):
        cameras = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                line = lines[i].strip()
                if line.startswith('#') or not line: continue
                data = line.split()
                view_id = int(data[0])
                q = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
                t = np.array([float(data[5]), float(data[6]), float(data[7])])
                cam_id = int(data[8])
                filename = data[9]

                q_obj = QQuaternion(q[0], q[1], q[2], q[3])
                q_inv = q_obj.conjugated()
                
                R = np.array([
                    [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
                    [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
                    [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
                ])
                center = -np.dot(R.T, t)
                cameras.append((center, q_inv, R, t.reshape(3,1), view_id, filename, cam_id))
        return cameras

    def render_point_cloud(self, points, colors):
        if self.scatter_plot in self.view.items:
            self.view.removeItem(self.scatter_plot)
        self.original_points = points
        self.original_colors = colors
        self.scatter_plot = gl.GLScatterPlotItem(pos=points, color=colors, size=self.pt_scale_val/10.0, pxMode=True)
        self.view.addItem(self.scatter_plot)
        
        center = np.mean(points, axis=0)
        self.view.opts['center'] = QVector3D(*center)
        self.view.opts['distance'] = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))

    def render_cameras(self, cameras, scale_mult):
        if self.camera_lines in self.view.items:
            self.view.removeItem(self.camera_lines)
            
        points = []
        colors = []
        
        base_scale = 100.0 / 30.0
        scale = base_scale * scale_mult
        
        w_f, h_f, f_f = 1.0 * scale, 0.75 * scale, 1.0 * scale

        for c in cameras:
            center = QVector3D(*c[0])
            q_inv = c[1]
            
            pts_local = [
                QVector3D(0, 0, 0),                 
                QVector3D(-w_f, -h_f, f_f),         
                QVector3D(w_f, -h_f, f_f),          
                QVector3D(w_f, h_f, f_f),           
                QVector3D(-w_f, h_f, f_f)           
            ]
            
            pw = [(center + q_inv.rotatedVector(p)) for p in pts_local]
            
            lines = [
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (2, 3), (3, 4), (4, 1)
            ]
            
            for line in lines:
                p1 = pw[line[0]]
                p2 = pw[line[1]]
                points.extend([[p1.x(), p1.y(), p1.z()], [p2.x(), p2.y(), p2.z()]])
                colors.extend([[0, 1, 0, 1], [0, 1, 0, 1]]) 
                
        self.camera_lines = gl.GLLinePlotItem(pos=np.array(points), color=np.array(colors), mode='lines')
        self.view.addItem(self.camera_lines)

    def run_normalization(self):
        if not self.cameras_data: return
        
        A_camera_normalize = np.zeros((3, 3))
        b_camera_normalize = np.zeros(3)
        camera_center_list = []

        for center, _, R, _, _, _, _ in self.cameras_data:
            camera_center_list.append(center.reshape(3, 1))
            vi = R[2, :].reshape(3, 1) 
            Vi = vi @ vi.T
            A_camera_normalize += np.eye(3) - Vi
            b_camera_normalize += (center.reshape(3, 1).T @ (np.eye(3) - Vi)).flatten()

        offset = np.linalg.lstsq(A_camera_normalize, b_camera_normalize, rcond=None)[0]
        dists = [np.linalg.norm(c.flatten() - offset) for c in camera_center_list]
        self.norm_max_dist = np.max(dists)
        
        self.norm_offset = QVector3D(offset[0], offset[1], offset[2])
        self.view.gizmo_pos = self.norm_offset
        self.view.gizmo_active = True
        
        self.norm_ratio = 3.0
        self.rebuild_normalization_items()
        self.btn_export.setEnabled(True)

    def rebuild_normalization_items(self):
        if self.norm_offset is None or self.norm_max_dist is None: return

        if self.sphere_item in self.view.items: self.view.removeItem(self.sphere_item)
        if self.gizmo_item in self.view.items: self.view.removeItem(self.gizmo_item)

        md = gl.MeshData.sphere(rows=40, cols=40)
        alpha = self.opacity_val / 100.0
        color_rgba = tuple(self.sphere_color + [alpha])

        custom_opts = {
            ogl.GL_DEPTH_TEST: True,
            ogl.GL_BLEND: True,
            ogl.GL_ALPHA_TEST: False,
            ogl.GL_CULL_FACE: True, 
            'glBlendFunc': (ogl.GL_SRC_ALPHA, ogl.GL_ONE_MINUS_SRC_ALPHA),
            'glCullFace': (ogl.GL_BACK,) 
        }

        self.sphere_item = gl.GLMeshItem(
            meshdata=md, smooth=True, drawFaces=True, 
            drawEdges=False, color=color_rgba,
            edgeColor=(0,0,0,0),
            shader='balloon', glOptions=custom_opts
        )
        self.view.addItem(self.sphere_item)
        
        self.gizmo_item = gl.GLAxisItem()
        self.view.addItem(self.gizmo_item)

        self.update_normalization_transforms()

    def update_normalization_transforms(self):
        if self.norm_offset is None or self.norm_max_dist is None: return
        if self.sphere_item is None or self.gizmo_item is None: return
        
        radius = self.norm_max_dist / max(self.norm_ratio, 0.001)
        
        self.sphere_item.resetTransform()
        self.sphere_item.scale(radius, radius, radius)
        self.sphere_item.translate(self.norm_offset.x(), self.norm_offset.y(), self.norm_offset.z())
        
        self.gizmo_item.setSize(x=self.norm_max_dist/2, y=self.norm_max_dist/2, z=self.norm_max_dist/2)
        self.gizmo_item.resetTransform()
        self.gizmo_item.translate(self.norm_offset.x(), self.norm_offset.y(), self.norm_offset.z())
        self.btn_export.setText("Confirm Normalization")

    def _on_gizmo_moved(self, new_pos):
        self.norm_offset = new_pos
        self.update_normalization_transforms()

    def _on_gizmo_scale_delta(self, dy):
        sensitivity = 0.02
        self.norm_ratio = max(0.1, min(self.norm_ratio + (dy * sensitivity), 20.0))
        self.update_normalization_transforms()

    def _on_gizmo_scale_factor(self, factor):
        self.norm_ratio = max(0.1, min(self.norm_ratio / factor, 20.0))
        self.update_normalization_transforms()

    def export_npz(self):
        if not self.cameras_data or self.norm_offset is None: return
        
        out_dir = self.main_app.lino_scene.combo_runs.currentData()
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Warning", "No LINO run selected in Normals tab. Please go to Normals tab and select a run first.")
            return

        run_config_path = os.path.join(out_dir, "run_config.json")
        
        if not os.path.exists(run_config_path):
            QMessageBox.warning(self, "Warning", f"No run_config.json found in {out_dir}. Cannot apply LINO cropping to intrinsics.")
            return
            
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
            
        groups = run_config.get("groups", [])
        
        save_path = os.path.join(out_dir, "cameras_sphere.npz")

        try:
            camera_sphere = {}
            offset = np.array([self.norm_offset.x(), self.norm_offset.y(), self.norm_offset.z()])
            scale = self.norm_max_dist / max(self.norm_ratio, 0.001)
            
            scale_mat = np.eye(4)
            scale_mat[:3, :3] *= scale
            scale_mat[:3, 3] = offset
            
            # Map Colmap filenames to their data
            colmap_dict = {}
            for _, _, R, t, view_id, filename, cam_id in self.cameras_data:
                colmap_dict[os.path.basename(filename)] = (R, t, cam_id)
                
            for group in groups:
                g_idx = group.get("group_idx")
                sampled = group.get("sampled_images", [])
                if not sampled: continue
                
                # Try matching by max_group / mean_group names (used in SfM)
                possible_names = [
                    f"max_group_{g_idx}.jpg", f"mean_group_{g_idx}.jpg",
                    f"max_group_{g_idx}.png", f"mean_group_{g_idx}.png"
                ]
                
                ref_filename = None
                for name in possible_names:
                    if name in colmap_dict:
                        ref_filename = name
                        break
                        
                # Fallback to the original raw image name if SfM was run directly on raw views
                if ref_filename is None and sampled:
                    raw_name = os.path.basename(sampled[0])
                    if raw_name in colmap_dict:
                        ref_filename = raw_name
                        
                if not ref_filename:
                    continue
                    
                R, t, cam_id = colmap_dict[ref_filename]
                K_mat_local = self.intrinsics_dict[cam_id][0].copy()
                
                # Apply crop and resize
                y0, y1 = group.get("r_s", 0), group.get("r_e", 0)
                x0, x1 = group.get("c_s", 0), group.get("c_e", 0)
                w_new, h_new = group.get("target_w", 0), group.get("target_h", 0)
                
                orig_w = x1 - x0
                orig_h = y1 - y0
                
                if orig_w > 0 and orig_h > 0 and w_new > 0 and h_new > 0:
                    scale_x = w_new / orig_w
                    scale_y = h_new / orig_h
                    
                    K_mat_local[0, 2] -= x0
                    K_mat_local[1, 2] -= y0
                    
                    K_mat_local[0, 0] *= scale_x
                    K_mat_local[1, 1] *= scale_y
                    K_mat_local[0, 2] *= scale_x
                    K_mat_local[1, 2] *= scale_y

                K_4x4 = np.eye(4)
                K_4x4[:3, :3] = K_mat_local
                
                W2C = np.eye(4)
                W2C[:3, :3] = R
                W2C[:3, 3] = t.flatten()
                
                camera_sphere[f"world_mat_{g_idx}"] = K_4x4 @ W2C
                camera_sphere[f"scale_mat_{g_idx}"] = scale_mat
                
            np.savez(save_path, **camera_sphere)
            self.btn_export.setText("Confirmed!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export NPZ:\n{str(e)}")

