"""Background removal + face-centred crop, producing the sprite the game
actually draws.

rembg (MIT, https://github.com/danielgatis/rembg) driving the u2netp weights
(Apache 2.0, from the original U-2-Net repo -- both permit commercial use,
verified by reading the source repos rather than trusting PyPI/npm metadata,
same discipline as celeb-lookalike's LICENSES.md). u2netp over the full u2net
model on purpose: ~4.7MB vs ~176MB, and a comedic bird sprite does not need
matting-grade edge quality.
"""

from __future__ import annotations

import io
import logging
import threading

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

log = logging.getLogger(__name__)

#: How much wider than the raw face box the final crop is. >1 so the sprite
#: shows some hair/forehead/chin/neck rather than a tight eyes-nose-mouth
#: rectangle -- it needs to read as "a head", not "a face detection box".
_PAD_FACTOR = 2.4


class CutoutMaker:
    def __init__(self, model_name: str = "u2netp") -> None:
        log.info("loading rembg session (%s)...", model_name)
        self._session = new_session(model_name)
        # rembg's session is not documented as thread-safe; FastAPI runs sync
        # endpoints in a thread pool, so serialise the same way FaceCounter does.
        self._lock = threading.Lock()
        log.info("rembg ready")

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
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(rgb)

        with self._lock:
            # Run on the full frame, not a pre-crop -- rembg segments better
            # with torso/shoulder context than on a tight face-only patch.
            cutout = remove(source, session=self._session)

        x1, y1, x2, y2 = face_box
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(fw, fh) * _PAD_FACTOR

        img_w, img_h = cutout.size
        cx1, cy1, cx2, cy2 = self._square_crop_box(cx, cy, side, img_w, img_h)
        cropped = cutout.crop((cx1, cy1, cx2, cy2))
        cropped = cropped.resize((output_size, output_size), Image.LANCZOS)

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
