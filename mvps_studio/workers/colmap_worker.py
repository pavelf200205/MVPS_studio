import os
import shutil
import subprocess
from PySide6.QtCore import QThread, Signal

class ColmapWorker(QThread):
    progress = Signal(str, int)  # status_message, progress_percentage
    finished_reconstruction = Signal(bool, str)  # success, message

    def __init__(self, workspace_dir, use_masks=False):
        super().__init__()
        self.workspace_dir = workspace_dir
        self.use_masks = use_masks
        self.is_interrupted = False
        
        self.max_imgs_dir = os.path.join(workspace_dir, "max_stack")
        self.masks_dir = os.path.join(workspace_dir, "masks")
        self.colmap_dir = os.path.join(workspace_dir, "colmap")
        self.database_path = os.path.join(self.colmap_dir, "database.db")
        self.sparse_dir = os.path.join(self.colmap_dir, "sparse", "0")

    def run(self):
        try:
            # 1. Setup
            self.progress.emit("Setting up COLMAP environment...", 0)
            if os.path.exists(self.database_path):
                try: os.remove(self.database_path)
                except Exception: pass
            
            if os.path.exists(os.path.join(self.colmap_dir, "sparse")):
                try: shutil.rmtree(os.path.join(self.colmap_dir, "sparse"))
                except Exception: pass
            
            os.makedirs(self.sparse_dir, exist_ok=True)
            
            # ---- SYSTEM COLMAP CLI ROUTE ----
            self.progress.emit("Extracting features using System COLMAP (CUDA)...", 10)
            cmd_ext = [
                "colmap", "feature_extractor",
                "--database_path", self.database_path,
                "--image_path", self.max_imgs_dir,
                "--ImageReader.camera_model", "SIMPLE_RADIAL"
            ]
            if self.use_masks and os.path.exists(self.masks_dir):
                cmd_ext.extend(["--ImageReader.mask_path", self.masks_dir])
            
            try:
                cmd_str = " ".join(cmd_ext)
                subprocess.run(cmd_str, check=True, capture_output=True, shell=True)
            except FileNotFoundError:
                self.finished_reconstruction.emit(False, "System 'colmap' command not found! Please install COLMAP manually and add it to your Windows PATH.")
                return
            except subprocess.CalledProcessError as e:
                self.finished_reconstruction.emit(False, f"Feature extraction failed:\\n{e.stderr.decode('utf-8', errors='ignore')}")
                return

            if self.is_interrupted: return

            self.progress.emit("Matching features via System COLMAP...", 40)
            try:
                subprocess.run(f"colmap exhaustive_matcher --database_path {self.database_path}", check=True, capture_output=True, shell=True)
            except subprocess.CalledProcessError as e:
                self.finished_reconstruction.emit(False, f"Matcher failed:\\n{e.stderr.decode('utf-8', errors='ignore')}")
                return

            if self.is_interrupted: return

            self.progress.emit("Running mapping via System COLMAP...", 70)
            try:
                out_path = os.path.join(self.colmap_dir, "sparse")
                subprocess.run(f"colmap mapper --database_path {self.database_path} --image_path {self.max_imgs_dir} --output_path {out_path}", check=True, capture_output=True, shell=True)
            except subprocess.CalledProcessError as e:
                self.finished_reconstruction.emit(False, f"Mapper failed:\\n{e.stderr.decode('utf-8', errors='ignore')}")
                return
                    
            if self.is_interrupted: return
            
            self.progress.emit("Finalizing...", 90)
            
            # Subprocess might generate the model at 'sparse/0'
            sparse_0_path = os.path.join(self.colmap_dir, "sparse", "0")
            if os.path.exists(sparse_0_path):
                self.progress.emit("Converting model to TXT format...", 95)
                try:
                    subprocess.run(f"colmap model_converter --input_path {sparse_0_path} --output_path {sparse_0_path} --output_type TXT", check=True, capture_output=True, shell=True)
                except subprocess.CalledProcessError as e:
                    self.finished_reconstruction.emit(False, f"Model converter failed:\\n{e.stderr.decode('utf-8', errors='ignore')}")
                    return
                
                self.progress.emit("Done", 100)
                self.finished_reconstruction.emit(True, sparse_0_path)
            else:
                self.finished_reconstruction.emit(False, "Sparse model not found at expected path: sparse_0_path")

        except Exception as e:
            self.finished_reconstruction.emit(False, str(e))
