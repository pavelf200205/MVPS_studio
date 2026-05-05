import os
import json
import time
import textwrap

import numpy as np
import trimesh

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QProgressBar, QCheckBox, QTextEdit, QSpinBox, QMessageBox, QFileDialog, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QVector3D

import pyqtgraph.opengl as gl
import pyvista as pv
from pyvistaqt import QtInteractor

from mvps_studio.gui.widgets.settings_panel import SettingsPanel
from mvps_studio.modules.supernormal_worker import SuperNormalWorker

class SuperNormalScene(QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.worker = None
        self._setup_ui()
        self._setup_3d_view()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_resolution_limits()

    def update_resolution_limits(self):
        lino_dir = self.main_app.lino_scene.combo_runs.currentData()
        if not lino_dir or not os.path.isdir(lino_dir):
            return
            
        cfg_path = os.path.join(lino_dir, "run_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                    res = cfg.get("resolution", 1024)
                    self.slider_res.setMaximum(res)
                    if self.slider_res.value() > res:
                        self.slider_res.setValue(res)
            except Exception:
                pass

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter to allow resizing sidebar vs viewport
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # --- Left: Viewport and Logs ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = QtInteractor(self)
        self.view.set_background('black')
        left_layout.addWidget(self.view, 3) # Viewport gets more space
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("SuperNormal training logs will appear here...")
        left_layout.addWidget(self.log_console, 1) # Logs get less space
        
        self.splitter.addWidget(left_container)
        
        # --- Right: Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sl = QVBoxLayout(self.sidebar)
        sl.addWidget(QLabel("<b>SuperNormal Reconstruction</b>"))
        
        # Status Label
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #aaa;")
        sl.addWidget(self.lbl_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        sl.addWidget(self.progress_bar)
        
        self.lbl_it_s = QLabel("it/s: 0.00")
        self.lbl_spent = QLabel("Spent: 00:00:00")
        self.lbl_eta = QLabel("ETA: 00:00:00")
        self.lbl_total = QLabel("Total: 00:00:00")
        
        timing_lyt = QHBoxLayout()
        v1 = QVBoxLayout()
        v1.addWidget(self.lbl_it_s)
        v1.addWidget(self.lbl_spent)
        v2 = QVBoxLayout()
        v2.addWidget(self.lbl_eta)
        v2.addWidget(self.lbl_total)
        timing_lyt.addLayout(v1)
        timing_lyt.addLayout(v2)
        sl.addLayout(timing_lyt)
        
        sl.addSpacing(10)
        
        # Settings
        self.lbl_target_res = QLabel("Target Resolution: 1024px")
        self.slider_res = QSlider(Qt.Orientation.Horizontal)
        self.slider_res.setRange(256, 2048)
        self.slider_res.setValue(1024)
        self.slider_res.valueChanged.connect(self._on_res_slider)
        
        r1 = QHBoxLayout()
        r1.addWidget(self.lbl_target_res)
        r1.addWidget(self.slider_res)
        sl.addLayout(r1)
        
        self.lbl_iters = QLabel("Iterations:")
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(1000, 100000)
        self.spin_iters.setSingleStep(1000)
        self.spin_iters.setValue(30000)
        
        r2 = QHBoxLayout()
        r2.addWidget(self.lbl_iters)
        r2.addWidget(self.spin_iters)
        sl.addLayout(r2)
        
        self.chk_optimize_vram = QCheckBox("Enable VRAM Optimization (Dynamic V_inverse)")
        self.chk_optimize_vram.setChecked(False)
        self.chk_optimize_vram.setVisible(False)  # Hidden due to device mismatch bug
        # sl.addWidget(self.chk_optimize_vram)
        
        sl.addSpacing(10)
        
        # Controls
        self.btn_start = QPushButton("Start Reconstruction")
        self.btn_start.clicked.connect(self.start_reconstruction)
        sl.addWidget(self.btn_start)
        
        self.btn_export_ds = QPushButton("Export Dataset for Custom SuperNormal")
        self.btn_export_ds.clicked.connect(self.export_dataset_action)
        sl.addWidget(self.btn_export_ds)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_reconstruction)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #5a2020;")
        sl.addWidget(self.btn_stop)
        
        self.btn_export = QPushButton("Export Mesh")
        self.btn_export.clicked.connect(self.export_mesh_action)
        self.btn_export.setEnabled(False)
        sl.addWidget(self.btn_export)
        
        sl.addStretch()
        
        self.splitter.addWidget(self.sidebar)
        self.splitter.setSizes([800, 300])

    def _setup_3d_view(self):
        # Enable VTK's TrackballCamera interactor style — provides unconstrained
        # arcball rotation (identical to the normalization tab) with no up-vector lock.
        self.view.enable_trackball_style()
        
        # Disable the up-vector constraint so the camera can roll freely
        # (VTK cameras lock to Y-up by default; this removes that restriction)
        self.view.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        try:
            style = self.view.iren.GetInteractorStyle()
            style.SetAutoAdjustCameraClippingRange(True)
        except Exception:
            pass

    def _on_res_slider(self, val):
        self.lbl_target_res.setText(f"Target Resolution: {val}px")

    def _log(self, msg):
        self.log_console.append(msg)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def start_reconstruction(self):
        if not self.main_app.workspace_dir:
            QMessageBox.warning(self, "Warning", "No project loaded.")
            return
            
        lino_dir = self.main_app.lino_scene.combo_runs.currentData()
        if not lino_dir or not os.path.isdir(lino_dir):
            QMessageBox.warning(self, "Warning", "No valid LINO run selected. Go to Normals tab and select a run.")
            return
        
        cameras_npz = os.path.join(lino_dir, "cameras_sphere.npz")
        if not os.path.exists(cameras_npz):
            QMessageBox.warning(self, "Warning", "cameras_sphere.npz not found! Please run Camera Normalization first.")
            return
            
        import shutil
        import glob
        
        # Prepare SuperNormal dataset structure
        # Native Integration: we no longer copy files. SuperNormal's dataset_loader.py
        # now natively understands the MVPS Studio LINO directory structure!
        
        base_exp_dir = os.path.join(lino_dir, "supernormal").replace("\\", "/")
        safe_lino_dir = lino_dir.replace("\\", "/")

        # Create config
        conf_text = textwrap.dedent(f'''
        general {{
            dataset_class = models.dataset_loader.Dataset
            renderer_class = models.renderer.NeuSRenderer

            base_exp_dir = "{base_exp_dir}"
            recording = []
        }}

        dataset {{
            data_dir = "{safe_lino_dir}"
            normal_dir = "outputs"
            render_cameras_name = "cameras_sphere.npz"
            object_cameras_name = "cameras_sphere.npz"
            cameras_name = "cameras_sphere.npz"
            exclude_views = []
            upsample_factor = 1
            target_resolution = {self.slider_res.value()}
            optimize_vram = False
        }}

        train {{
            learning_rate = 5e-4
            learning_rate_alpha = 0.05
            end_iter = {self.spin_iters.value()}
            increase_bindwidth_every = 2000

            gradient_method = dfd

            batch_size = 2048
            patch_size = 2

            warm_up_end = 500
            use_white_bkgd = False

            loss_type = l2
            normal_weight = 1.0
            eikonal_weight = 1.0
            mask_weight = 1.0
        }}

        val {{
            save_freq = 10000
            val_normal_freq = 100001
            val_normal_resolution_level = 2
            gradient_method = dfd
            val_mesh_freq = 10000
            val_mesh_res = 1024
            report_freq = 100
            eval_metric_freq = 10000
        }}

        model {{
            sdf_network {{
                d_out = 1
                d_in = 3
                d_hidden = 64
                n_layers = 1
                skip_in = [-1]
                bias = 0.8
                geometric_init = True
                weight_norm = True
                input_concat = True
            }}

            variance_network {{
                init_val = 0.5
            }}

            ray_marching {{
                start_step_size = 1e-2
                end_step_size = 1e-3
                occ_threshold = 0.1
                occ_sigmoid_k = 80.0
                occ_resolution = 128
                occ_update_freq = 8
            }}

            encoding {{
                otype = HashGrid
                n_levels = 14
                n_features_per_level = 2
                log2_hashmap_size = 19
                base_resolution = 32
                per_level_scale = 1.3195079107728942
            }}
        }}
        ''')
        
        # Save conf just in case
        os.makedirs(os.path.join(lino_dir, "supernormal"), exist_ok=True)
        conf_path = os.path.join(lino_dir, "supernormal", "run.conf")
        with open(conf_path, "w") as f:
            f.write(conf_text)
            
        self._log(f"Starting SuperNormal training...")
        
        self.worker = SuperNormalWorker(conf_text, mode='train', is_continue=False)
        self.worker.log_msg.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.lbl_status.setText("Running...")
        self.progress_bar.setValue(0)
        self.start_time = time.time()
        
        self.worker.start()
        


    def _on_progress(self, current_iter, total_iter):
        if total_iter > 0:
            pct = int(current_iter * 100 / total_iter)
            self.progress_bar.setValue(pct)
            
            elapsed = time.time() - self.start_time
            if elapsed > 0 and current_iter > 0:
                it_s = current_iter / elapsed
                remaining = total_iter - current_iter
                eta = remaining / it_s
                total_time = elapsed + eta
                
                self.lbl_it_s.setText(f"it/s: {it_s:.2f}")
                self.lbl_spent.setText(f"Spent: {self._format_time(elapsed)}")
                self.lbl_eta.setText(f"ETA: {self._format_time(eta)}")
                self.lbl_total.setText(f"Total: {self._format_time(total_time)}")
            
            self.lbl_status.setText(f"Iteration {current_iter} / {total_iter}")

    def _format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def stop_reconstruction(self):
        if not self.worker: return
        self.worker.stop()
        self.lbl_status.setText("Stopping...")

    def _on_finished(self, success, msg_or_path):
        if success:
            self._log("Process finished successfully.")
            self.lbl_status.setText("Finished")
            self.progress_bar.setValue(100)
            self.load_latest_mesh(msg_or_path)
            self.btn_export.setEnabled(True)
        else:
            self._log(f"Process finished with error: {msg_or_path}")
            self.lbl_status.setText("Stopped/Failed")
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def load_latest_mesh(self, base_exp_dir):
        if not base_exp_dir:
            return
            
        meshes_dir = os.path.join(base_exp_dir, "meshes_validation")
        
        if not os.path.exists(meshes_dir): 
            self._log(f"Warning: Could not find meshes at {meshes_dir}")
            return
        
        ply_files = [os.path.join(meshes_dir, f) for f in os.listdir(meshes_dir) if f.endswith('.ply')]
        if not ply_files: 
            self._log("Warning: No .ply files found in mesh output.")
            return
        
        # Get the latest modified ply file
        latest_ply = max(ply_files, key=os.path.getmtime)
        self.latest_mesh_path = latest_ply
        self._log(f"Loading mesh: {os.path.basename(latest_ply)}")
        
        try:
            self.view.clear()
            mesh = pv.read(latest_ply)
            
            # Add mesh with smooth shading and high ambient so shadowed areas are still visible
            self.view.add_mesh(
                mesh,
                color='#ddd5c8',       # warm off-white clay look
                smooth_shading=True,
                specular=0.15,
                specular_power=10,
                ambient=0.45,          # high ambient prevents pitch-black shadows
                diffuse=0.55,
                pbr=False
            )
            
            # Use a single headlight (camera-attached) so it always illuminates
            # whichever face the user is currently looking at, regardless of rotation.
            self.view.remove_all_lights()
            headlight = pv.Light(light_type='headlight', intensity=1.0)
            self.view.add_light(headlight)
            
            self.view.reset_camera()
            
        except Exception as e:
            self._log(f"Failed to load mesh {latest_ply}: {str(e)}")

    def export_mesh_action(self):
        if not hasattr(self, 'latest_mesh_path') or not self.latest_mesh_path:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "reconstructed_mesh.ply", "PLY Files (*.ply)")
        if save_path:
            import shutil
            try:
                shutil.move(self.latest_mesh_path, save_path)
                self._log(f"Successfully exported mesh to {save_path}")
                self.btn_export.setEnabled(False)
            except Exception as e:
                self._log(f"Error moving mesh: {e}")

    def export_dataset_action(self):
        lino_dir = self.main_app.lino_scene.combo_runs.currentData()
        if not lino_dir or not os.path.isdir(lino_dir):
            QMessageBox.warning(self, "Warning", "No valid LINO run selected. Go to Normals tab and select a run.")
            return
        
        cameras_npz = os.path.join(lino_dir, "cameras_sphere.npz")
        if not os.path.exists(cameras_npz):
            QMessageBox.warning(self, "Warning", "cameras_sphere.npz not found! Please run Camera Normalization first.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not save_dir:
            return

        import shutil
        
        try:
            self._log(f"Exporting dataset to {save_dir}...")
            os.makedirs(os.path.join(save_dir, "mask"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "normal_world_space_sdmunips"), exist_ok=True)
            config_path = os.path.join(lino_dir, 'run_config.json')
            if not os.path.exists(config_path):
                raise Exception("run_config.json not found")
                
            with open(config_path, 'r') as f:
                run_config = json.load(f)
            groups = run_config.get("groups", [])
            
            inputs_dir = os.path.join(lino_dir, "inputs")
            outputs_dir = os.path.join(lino_dir, "outputs")
            
            import cv2
            import numpy as np
            import pyexr
            
            valid_views = []
            max_h, max_w = 0, 0
            
            self._log("Scanning images to determine padding requirements...")
            
            # Pass 1: Determine max dimensions
            for group in groups:
                idx = group.get("group_idx")
                src_mask = os.path.join(inputs_dir, f"view_{idx:02d}.data", "mask.png")
                src_normal = os.path.join(outputs_dir, f"{idx:02d}_normal_world.exr")
                
                if not os.path.exists(src_normal):
                    src_normal = os.path.join(outputs_dir, f"{idx:02d}_normal_world.npy")
                
                if os.path.exists(src_mask) and os.path.exists(src_normal):
                    mask_img = cv2.imread(src_mask, cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        h, w = mask_img.shape[:2]
                        max_h = max(max_h, h)
                        max_w = max(max_w, w)
                        valid_views.append((idx, src_mask, src_normal))
            
            # Calculate Scaling from Target Resolution
            target_res = self.slider_res.value()
            max_dim = max(max_h, max_w)
            scale_ratio = 1.0
            if max_dim > target_res:
                scale_ratio = target_res / float(max_dim)
            
            new_h = int(max_h * scale_ratio)
            new_w = int(max_w * scale_ratio)
            
            if scale_ratio < 1.0:
                self._log(f"Downscaling dataset from {max_w}x{max_h} to {new_w}x{new_h} (Ratio: {scale_ratio:.3f})")
            
            cameras_data = dict(np.load(cameras_npz))
            
            # Pass 2: Pad, Resize, and Export
            count = 0
            for idx, src_mask, src_normal in valid_views:
                dst_mask = os.path.join(save_dir, "mask", f"{idx:03d}.png")
                dst_normal = os.path.join(save_dir, "normal_world_space_sdmunips", f"{idx:03d}.exr")
                
                # Process Mask
                mask_img = cv2.imread(src_mask, cv2.IMREAD_UNCHANGED)
                pad_h = max_h - mask_img.shape[0]
                pad_w = max_w - mask_img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    if len(mask_img.shape) == 3:
                        mask_img = np.pad(mask_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                    else:
                        mask_img = np.pad(mask_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                        
                if scale_ratio < 1.0:
                    mask_img = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                cv2.imwrite(dst_mask, mask_img)
                
                # Process Normal
                if src_normal.endswith('.npy'):
                    norm_arr = np.load(src_normal)
                else:
                    norm_arr = pyexr.read(src_normal)
                    
                pad_h = max_h - norm_arr.shape[0]
                pad_w = max_w - norm_arr.shape[1]
                if pad_h > 0 or pad_w > 0:
                    norm_arr = np.pad(norm_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                
                if scale_ratio < 1.0:
                    norm_arr = cv2.resize(norm_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                pyexr.write(dst_normal, norm_arr)
                
                # Process Intrinsics
                if scale_ratio < 1.0:
                    w_key = f"world_mat_{idx}"
                    if w_key in cameras_data:
                        w_mat = cameras_data[w_key].copy()
                        # Scale the projection part of the matrix (top 2 rows)
                        w_mat[0, :] *= scale_ratio
                        w_mat[1, :] *= scale_ratio
                        cameras_data[w_key] = w_mat
                
                count += 1
                
            np.savez(os.path.join(save_dir, "cameras_sphere.npz"), **cameras_data)
                
            self._log(f"Dataset exported successfully! ({count} views, {new_w}x{new_h})")
            QMessageBox.information(self, "Success", f"Dataset exported successfully!\n{count} views processed and baked to {new_w}x{new_h}.")
        except Exception as e:
            self._log(f"Error exporting dataset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export dataset:\n{str(e)}")
