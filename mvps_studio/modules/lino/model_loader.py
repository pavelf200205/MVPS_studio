"""
mvps_studio.modules.lino.model_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single point of contact between MVPS Studio and the LINO-UniPS
model code that lives in the project's third_party/ tree.

Every other module in mvps_studio imports exclusively from HERE,
so if the LINO submodule ever moves, only this file needs to change.
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Locate the LINO-UniPS checkout relative to *this* file.
# This is the ONLY place in MVPS Studio that knows the physical path.
# ---------------------------------------------------------------------------
_LINO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 os.pardir, os.pardir, os.pardir,   # → project root
                 "third_party", "LINO_UniPS")
)
_WEIGHTS_DIR = os.path.join(_LINO_ROOT, "weights")


def _ensure_path():
    """Insert LINO-UniPS root into sys.path (idempotent)."""
    if _LINO_ROOT not in sys.path:
        sys.path.insert(0, _LINO_ROOT)


def _patch_predictor(predictor):
    """
    Monkey-patch ``Predictor.predict`` to fix a bfloat16 precision bug.

    The original code converts **every** numpy array in the batch to
    ``self.dtype`` (bfloat16 on Ampere+).  bfloat16 has only 7 mantissa
    bits, so integer ROI coordinates like 333 silently become 332,
    causing a shape mismatch between ``nout`` and ``mask_original`` in
    ``predict_step``.

    The patched version keeps ``roi`` as int32 and ``mask_original`` as
    float32 so their shapes stay exact, while image data remains bfloat16.
    """
    import torch
    from src.data import DemoData                     # noqa: E402

    def _safe_predict(self, input_imgs_list, input_mask):
        demodata = DemoData(input_imgs_list, input_mask)
        data = demodata[0]

        for key in data:
            v = data[key]
            if isinstance(v, np.ndarray):
                if key == "roi":
                    # Integer coordinates — must NOT be bfloat16
                    data[key] = torch.tensor(
                        v, device=self.device, dtype=torch.int32
                    )[None, ...]
                elif key == "mask_original":
                    # Shape-critical — keep float32
                    data[key] = torch.tensor(
                        v, device=self.device, dtype=torch.float32
                    )[None, ...]
                else:
                    data[key] = torch.tensor(
                        v, device=self.device, dtype=self.dtype
                    )[None, ...]
            elif isinstance(v, torch.Tensor):
                data[key] = v.to(self.device, dtype=self.dtype)[None, ...]
            elif v is None:
                data[key] = None
            else:
                raise TypeError(f"Unsupported data type: {type(v)}")

        with torch.no_grad():
            output = self.model(data)
        return output

    import types
    predictor.predict = types.MethodType(_safe_predict, predictor)
    return predictor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_predictor(weights_filename: str = "LINO-UniPS.pth"):
    """
    Return a ready-to-use LINO ``Predictor`` instance.

    The predictor exposes::

        predictor.predict(input_imgs_list, input_mask)
            input_imgs_list : list of (np.ndarray,) tuples
            input_mask      : np.ndarray [H, W, C]  (will use channel 0)

    Returns ``np.ndarray [H, W, 3]`` float32 (normals in –1 … 1).
    """
    _ensure_path()
    from hubconf import LINO                          # noqa: E402

    pth = os.path.join(_WEIGHTS_DIR, weights_filename)
    if not os.path.isfile(pth):
        # Try alternate naming conventions (hyphen vs underscore)
        alt_names = ["LINO-UniPS.pth", "LINO_UniPS.pth", "lino_unips.pth"]
        for alt in alt_names:
            alt_pth = os.path.join(_WEIGHTS_DIR, alt)
            if os.path.isfile(alt_pth):
                pth = alt_pth
                break
        else:
            avail = os.listdir(_WEIGHTS_DIR) if os.path.isdir(_WEIGHTS_DIR) else []
            raise FileNotFoundError(
                f"LINO-UniPS weights not found at:\n  {pth}\n"
                f"Available files in weights dir: {avail}\n"
                "Please download them and place in the weights/ folder."
            )

    predictor = LINO(local_file_path=pth, task_name="Real")
    return _patch_predictor(predictor)

