"""Background removal + face-centred crop, producing the sprite the game
actually draws.

Runs the u2netp segmentation model (Apache 2.0, from the original U-2-Net
repo -- verified by reading the source repo's own LICENSE, same discipline
as celeb-lookalike's LICENSES.md) directly through OpenCV's DNN backend
(`cv2.dnn`), the same backend YuNet already uses in face_detect.py -- not
through rembg + onnxruntime.

This used to go through rembg. rembg's own top-level module eagerly imports
pymatting, scipy, and scikit-image at `import rembg` time regardless of
whether their only consumer (the alpha_matting=True refinement path, which
this service never enables) is ever used -- so those dependencies could not
just be uninstalled after the fact without breaking the import outright.
Together with onnxruntime, that chain (pymatting + scipy + scikit-image +
numba + llvmlite + networkx + onnxruntime) measured at roughly 470MB of a
~900MB image, and that image size turned out to be the actual root cause of
a production outage: pulling it onto a fresh Render Free instance took
~6 minutes, consistently longer than Render's fixed 5-minute port-scan
timeout, so deploys and cold-starts intermittently registered as failed
before the app had even started. cv2.dnn needs none of that -- opencv is
already a hard dependency for YuNet, so running u2netp through it instead of
onnxruntime adds zero new dependencies and removes six.

The preprocessing/postprocessing below is deliberately a byte-for-byte port
of rembg's own U2netpSession.predict() and BaseSession.normalize() (read
directly from rembg's source to get this right, not reverse-engineered from
the model), so the actual cutout quality is unchanged -- only how the model
gets run changed.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import MODELS_DIR
from .model_download import ensure as ensure_model

log = logging.getLogger(__name__)

# Same weight file rembg itself downloads -- reusing its exact URL and MD5
# rather than sourcing the model elsewhere, so this is still the same
# already-verified Apache 2.0 U-2-Net weights, just fetched directly instead
# of through the rembg package. See LICENSES.md.
U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
U2NETP_FILE = "u2netp.onnx"
U2NETP_MD5 = "8e83ca70e441ab06c318d82300c84806"


def ensure_u2netp(models_dir: Path | None = None) -> Path:
    d = models_dir or MODELS_DIR
    return ensure_model(U2NETP_URL, d / U2NETP_FILE, md5=U2NETP_MD5)

#: How much wider than the raw face box the final crop is. >1 so the sprite
#: shows some hair/forehead/chin/neck rather than a tight eyes-nose-mouth
#: rectangle -- it needs to read as "a head", not "a face detection box".
_PAD_FACTOR = 2.4

#: u2netp's fixed model input size -- every image is resized to exactly this
#: for inference regardless of its original size, so this is not a tunable.
_MODEL_INPUT_SIZE = (320, 320)
_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)

#: Longest edge the *source* image is downscaled to before preprocessing.
#: The model itself always sees a 320x320 tensor either way, but PIL/numpy
#: still have to hold and resize the source array first -- a large
#: host-supplied photo (an uncompressed phone photo can easily be
#: 3000-4000px) costs real memory at that step even though the model input
#: is fixed, and the final output is only OUTPUT_SIZE square regardless, so
#: there is no quality reason to work from a higher resolution than this.
_MAX_SEGMENT_EDGE = 800


class CutoutMaker:
    def __init__(self) -> None:
        model_path = ensure_u2netp()
        log.info("loading u2netp via cv2.dnn (%s)...", model_path)
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        # cv2.dnn.Net is not documented as thread-safe; FastAPI runs sync
        # endpoints in a thread pool, so serialise the same way FaceCounter does.
        self._lock = threading.Lock()
        log.info("u2netp ready")

    def _segment_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Run u2netp on an RGB array, return a single-channel float32 mask
        in [0, 1] at the model's native 320x320 resolution."""
        resized = cv2.resize(rgb, _MODEL_INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
        arr = resized.astype(np.float64) / max(resized.max(), 1e-6)
        normalized = np.empty_like(arr)
        for c in range(3):
            normalized[:, :, c] = (arr[:, :, c] - _NORM_MEAN[c]) / _NORM_STD[c]
        blob = normalized.transpose(2, 0, 1)[None].astype(np.float32)

        with self._lock:
            self._net.setInput(blob)
            out = self._net.forward()

        pred = out[:, 0, :, :]
        lo, hi = pred.min(), pred.max()
        pred = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)
        return np.squeeze(pred).astype(np.float32)

    def _square_crop_box(
        self, cx: float, cy: float, side: float, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        side = min(side, img_w, img_h)
        half = side / 2
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + side, y1 + side
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > img_w:
            x1 -= x2 - img_w
            x2 = img_w
        if y2 > img_h:
            y1 -= y2 - img_h
            y2 = img_h
        x1, y1 = max(0.0, x1), max(0.0, y1)
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def make(
        self,
        bgr: np.ndarray,
        face_box: tuple[float, float, float, float],
        output_size: int,
    ) -> bytes:
        """Return PNG bytes: the person cut out of the background, cropped
        square around the face, resized to output_size x output_size."""
        h, w = bgr.shape[:2]
        scale = min(1.0, _MAX_SEGMENT_EDGE / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(
                bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            # face_box came from detect_boxes() on the *original*-resolution
            # image, so it has to scale down with it to stay pointing at the
            # same face rather than an arbitrary region of the resized one.
            face_box = tuple(v * scale for v in face_box)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]

        mask_small = self._segment_mask(rgb)
        mask = cv2.resize(mask_small, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

        # Same compositing rembg's naive_cutout() does: paste the source
        # image onto a transparent canvas everywhere the mask says
        # foreground, nothing everywhere it doesn't.
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        cutout = Image.fromarray(rgba, mode="RGBA")

        x1, y1, x2, y2 = face_box
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(fw, fh) * _PAD_FACTOR

        cx1, cy1, cx2, cy2 = self._square_crop_box(cx, cy, side, img_w, img_h)
        cropped = cutout.crop((cx1, cy1, cx2, cy2))
        cropped = cropped.resize((output_size, output_size), Image.LANCZOS)

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
