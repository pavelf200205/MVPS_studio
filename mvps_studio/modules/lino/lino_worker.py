"""
mvps_studio.modules.lino.lino_worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
QThread worker that crops / resizes / samples each group, writes the
intermediate data to disk in SDM-UniPS layout, runs LINO-UniPS inference,
and saves per-view normal maps as .exr (or .npy fallback).

Supports pause / resume / cancel.
"""

import os
import cv2
import json
import numpy as np
import random
import threading

from PySide6.QtCore import QThread, Signal


class LinoWorker(QThread):
    """Background thread for the full LINO-UniPS pipeline."""

    progress = Signal(str, int)
    # emitted after Phase 1 formatting is done and run_config.json is written
    formatting_done = Signal(str)               # run_dir
    # (group_idx, output_exr_path)  — emitted after each group finishes
    group_done = Signal(int, str)
    # (success, message, run_dir)
    finished_reconstruction = Signal(bool, str, str)

    def __init__(self, workspace_dir, run_name, groups, mask_paths,
                 resolution, max_images,
                 resume_run_dir=None, skip_group_indices=None):
        super().__init__()
        self.workspace_dir = workspace_dir
        self.run_name = run_name
        self.groups = groups          # list[list[str]]  image paths per group
        self.mask_paths = mask_paths  # list[str]        one mask per group
        self.resolution = resolution  # int              longest side px
        self.max_images = max_images  # int              cap per group
        # Resume mode: if set, skip Phase 1 and use this directory directly
        self.resume_run_dir = resume_run_dir
        self.skip_group_indices = set(skip_group_indices or [])

        self._cancel = False
        self._pause_event = threading.Event()
        self._pause_event.set()       # starts un-paused

    # ------------------------------------------------------------------
    # public control API  (called from the GUI thread)
    # ------------------------------------------------------------------
    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def cancel(self):
        self._cancel = True
        self._pause_event.set()       # unblock if paused so thread can exit

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_state(self):
        """Block while paused; return False if cancelled."""
        self._pause_event.wait()
        return not self._cancel

    @staticmethod
    def _largest_component_bbox(mask_gray):
        """Return (r_s, r_e, c_s, c_e) of the largest connected component."""
        _, bw = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            bw, connectivity=8
        )
        if n_labels <= 1:
            return 0, mask_gray.shape[0], 0, mask_gray.shape[1]

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x = stats[largest, cv2.CC_STAT_LEFT]
        y = stats[largest, cv2.CC_STAT_TOP]
        w = stats[largest, cv2.CC_STAT_WIDTH]
        h = stats[largest, cv2.CC_STAT_HEIGHT]
        return y, y + h, x, x + w

    # ------------------------------------------------------------------
    # main thread body
    # ------------------------------------------------------------------
    def run(self):
        try:
            self._do_run()
        except Exception:
            import traceback
            self.finished_reconstruction.emit(
                False, f"Error:\n{traceback.format_exc()}", "")

    def _do_run(self):
        self.progress.emit("Initializing LINO-UniPS …", 0)

        import torch
        try:
            import pyexr
            _has_pyexr = True
        except ImportError:
            _has_pyexr = False

        # ---- directory tree ------------------------------------------------
        if self.resume_run_dir:
            # ── RESUME MODE ── reuse existing run directory
            run_dir = self.resume_run_dir
            inputs_dir  = os.path.join(run_dir, "inputs")
            outputs_dir = os.path.join(run_dir, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)

            # Immediately surface already-completed outputs in the grid
            for gi in sorted(self.skip_group_indices):
                for ext in (".exr", ".npy"):
                    p = os.path.join(outputs_dir, f"{gi:02d}_normal{ext}")
                    if os.path.exists(p):
                        self.group_done.emit(gi, p)
                        break

            self.progress.emit("Resuming from existing run directory …", 40)
            self.formatting_done.emit(run_dir)

        else:
            # ── FRESH RUN ── full Phase 1 formatting
            run_dir = os.path.join(self.workspace_dir, "lino_runs", self.run_name)
            inputs_dir  = os.path.join(run_dir, "inputs")
            outputs_dir = os.path.join(run_dir, "outputs")
            os.makedirs(inputs_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)

            run_config = {
                "run_name":   self.run_name,
                "resolution": self.resolution,
                "max_images": self.max_images,
                "groups":     [],
            }
            total_groups = len(self.groups)

            # ============================================================
            # PHASE 1 — crop / resize / sample / save to disk
            # ============================================================
            self.progress.emit("Formatting dataset …", 5)

            for idx, group_imgs in enumerate(self.groups):
                if not self._check_state():
                    self.finished_reconstruction.emit(False, "Cancelled by user.", "")
                    return

                mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                r_s, r_e, c_s, c_e = self._largest_component_bbox(mask)
                if c_s >= c_e or r_s >= r_e:
                    r_s, r_e, c_s, c_e = 0, mask.shape[0], 0, mask.shape[1]

                if len(group_imgs) > self.max_images:
                    sampled = sorted(random.sample(group_imgs, self.max_images))
                else:
                    sampled = list(group_imgs)

                view_dir = os.path.join(inputs_dir, f"view_{idx:02d}.data")
                os.makedirs(view_dir, exist_ok=True)

                bbox_h, bbox_w = r_e - r_s, c_e - c_s
                max_side = max(bbox_h, bbox_w)
                scale = self.resolution / float(max_side) if max_side > 0 else 1.0
                tw = max(int(bbox_w * scale), 1)
                th = max(int(bbox_h * scale), 1)

                crop_mask = mask[r_s:r_e, c_s:c_e]
                cv2.imwrite(
                    os.path.join(view_dir, "mask.png"),
                    cv2.resize(crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST))

                saved_names = []
                for i, img_path in enumerate(sampled):
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    crop = img[r_s:r_e, c_s:c_e]
                    cv2.imwrite(
                        os.path.join(view_dir, f"L{i:02d}.jpg"),
                        cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
                    saved_names.append(img_path)

                run_config["groups"].append({
                    "group_idx":   idx,
                    "r_s": int(r_s), "r_e": int(r_e),
                    "c_s": int(c_s), "c_e": int(c_e),
                    "resize_scale":  float(scale),
                    "target_w": tw, "target_h": th,
                    "original_h": int(mask.shape[0]),
                    "original_w": int(mask.shape[1]),
                    "sampled_images": saved_names,
                })

                pct = 5 + int(35 * (idx + 1) / total_groups)
                self.progress.emit(f"Formatted group {idx + 1}/{total_groups}", pct)

            with open(os.path.join(run_dir, "run_config.json"), "w") as f:
                json.dump(run_config, f, indent=4)
            self.formatting_done.emit(run_dir)

        # ================================================================
        # PHASE 2 — load model via the bridge module
        # ================================================================
        if not self._check_state():
            self.finished_reconstruction.emit(False, "Cancelled by user.", "")
            return

        self.progress.emit("Loading LINO-UniPS model …", 42)

        from mvps_studio.modules.lino.model_loader import load_predictor
        predictor = load_predictor()

        # ================================================================
        # PHASE 3 — inference per view
        # ================================================================
        n_groups = len(self.groups)
        for idx in range(n_groups):
            if not self._check_state():
                self.finished_reconstruction.emit(False, "Cancelled by user.", "")
                return

            # Skip groups whose output already exists (resume mode)
            if idx in self.skip_group_indices:
                continue

            view_dir  = os.path.join(inputs_dir, f"view_{idx:02d}.data")
            mask_file = os.path.join(view_dir, "mask.png")
            if not os.path.exists(mask_file):
                continue

            img_files = sorted(
                f for f in os.listdir(view_dir)
                if f.startswith("L") and f.endswith(".jpg"))

            self.progress.emit(
                f"Inference on group {idx + 1}/{n_groups} "
                f"({len(img_files)} imgs) …",
                42 + int(53 * (idx + 1) / n_groups))

            input_images = []
            for fname in img_files:
                bgr = cv2.imread(os.path.join(view_dir, fname))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                input_images.append((rgb,))

            mask_bgr = cv2.imread(mask_file)
            if mask_bgr is None:
                g = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask_bgr = np.stack([g] * 3, axis=-1)

            try:
                normal_map = predictor.predict(input_images, mask_bgr)

                out_path = os.path.join(outputs_dir, f"{idx:02d}_normal.npy")
                np.save(out_path, normal_map)

                self.group_done.emit(idx, out_path)

            except Exception as e:
                self.finished_reconstruction.emit(
                    False, f"Inference failed on group {idx}: {e}", "")
                return

            torch.cuda.empty_cache()

        # ================================================================
        self.progress.emit("LINO reconstruction complete!", 100)
        self.finished_reconstruction.emit(True, "Success", run_dir)
