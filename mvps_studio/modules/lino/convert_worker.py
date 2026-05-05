"""
mvps_studio.modules.lino.convert_worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
QThread that converts LINO camera-frame EXR normals to world frame using
COLMAP camera poses (images.txt / cameras.txt in the sparse/0 folder).

Port of third_party/MVPS_Scripts/gather_and_convert_normal_map_colmap_multicam.py
"""
import os
import json
import numpy as np
from PySide6.QtCore import QThread, Signal


def _quat_to_R(qw, qx, qy, qz):
    """Quaternion → 3×3 rotation matrix (COLMAP convention)."""
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,  2*qy*qz + 2*qx*qw,  1 - 2*qx**2 - 2*qy**2],
    ])


class WorldFrameConverter(QThread):
    """Background worker: camera-frame → world-frame normal maps."""

    progress   = Signal(str, int)   # message, percent 0-100
    group_done = Signal(int, str)   # group_idx, world_normal_path
    finished   = Signal(bool, str)  # success, error_message

    def __init__(self, run_dir: str, colmap_sparse_dir: str):
        super().__init__()
        self.run_dir          = run_dir
        self.colmap_sparse_dir = colmap_sparse_dir
        self._cancel          = False

    def cancel(self):
        self._cancel = True

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._do_convert()
        except Exception:
            import traceback
            self.finished.emit(False, traceback.format_exc())

    def _do_convert(self):
        self.progress.emit("Reading run configuration …", 0)

        cfg_path = os.path.join(self.run_dir, "run_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        groups      = cfg.get("groups", [])
        outputs_dir = os.path.join(self.run_dir, "outputs")

        # ── Parse COLMAP images.txt ────────────────────────────────────
        self.progress.emit("Parsing COLMAP camera poses …", 5)
        images_txt = os.path.join(self.colmap_sparse_dir, "images.txt")
        pose_by_name = {}   # basename_no_ext → C2W rotation 3×3

        with open(images_txt) as f:
            raw = [l.strip() for l in f if not l.startswith('#') and l.strip()]

        # images.txt: header line, then 2D-points line (alternating)
        for i in range(0, len(raw), 2):
            parts = raw[i].split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz      = map(float, parts[5:8])
            name            = parts[9]
            name_noext      = os.path.splitext(name)[0]

            R_w2c = _quat_to_R(qw, qx, qy, qz)
            t     = np.array([tx, ty, tz]).reshape(3, 1)
            W2C   = np.eye(4)
            W2C[:3, :3]  = R_w2c
            W2C[:3, [3]] = t
            C2W   = np.linalg.inv(W2C)
            pose_by_name[name_noext] = C2W[:3, :3]  # C→W rotation

        try:
            import pyexr
            _has_pyexr = True
        except ImportError:
            _has_pyexr = False

        n = len(groups)
        for gi, g in enumerate(groups):
            if self._cancel:
                self.finished.emit(False, "Cancelled by user.")
                return

            idx = g["group_idx"]

            # Find C2W rotation via group index matching COLMAP max_stack filenames
            R = None
            keys_to_try = [f"max_group_{idx}", f"mean_group_{idx}"]
            for key in keys_to_try:
                if key in pose_by_name:
                    R = pose_by_name[key]
                    break

            if R is None:
                self.progress.emit(
                    f"Group {idx:02d}: no matching COLMAP pose — skipped",
                    int(gi / n * 100))
                continue

            # Find camera-frame normal
            cam_path = None
            for ext in (".exr", ".npy"):
                p = os.path.join(outputs_dir, f"{idx:02d}_normal{ext}")
                if os.path.exists(p):
                    cam_path = p
                    break

            if cam_path is None:
                self.progress.emit(
                    f"Group {idx:02d}: no camera-frame normal — skipped",
                    int(gi / n * 100))
                continue

            # Load
            if cam_path.endswith(".npy"):
                nml_cam = np.load(cam_path).copy()
            else:
                if not _has_pyexr:
                    self.finished.emit(False, "pyexr not installed; cannot read EXR files.")
                    return
                nml_cam = pyexr.read(cam_path).copy()

            # Flip Y & Z: match OpenCV convention (X right, Y down, Z front)
            nml_cam[..., [1, 2]] *= -1

            H, W = nml_cam.shape[:2]
            nml_world = (R @ nml_cam.reshape(-1, 3).T).T.reshape(H, W, 3)

            out_path = os.path.join(outputs_dir, f"{idx:02d}_normal_world.npy")
            np.save(out_path, nml_world.astype(np.float32))

            pct = int((gi + 1) / n * 100)
            self.progress.emit(f"Converted group {idx:02d}  ({gi+1}/{n})", pct)
            self.group_done.emit(idx, out_path)

        self.finished.emit(True, "")
