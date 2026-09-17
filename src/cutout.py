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
import onnxruntime as ort
from PIL import Image
from rembg import new_session, remove

log = logging.getLogger(__name__)

#: How much wider than the raw face box the final crop is. >1 so the sprite
#: shows some hair/forehead/chin/neck rather than a tight eyes-nose-mouth
#: rectangle -- it needs to read as "a head", not "a face detection box".
_PAD_FACTOR = 2.4

#: Longest edge fed to rembg -- same purpose as face_detect.py's
#: _DET_LONG_EDGE (celeb-lookalike's own convention). Without this, a large
#: host-supplied photo (an uncompressed phone photo can easily be
#: 3000-4000px) runs full-resolution through U2NETP's inference, and
#: measured RSS for even a small 396px test photo already peaked at ~437MB
#: on a fresh Render Free instance (512MB total) -- a several-times-larger
#: real photo would be a near-certain OOM kill. The final output is only
#: OUTPUT_SIZE (512px) square regardless, so there is no quality reason to
#: segment at a higher resolution than this in the first place.
_MAX_SEGMENT_EDGE = 800


class CutoutMaker:
    def __init__(self, model_name: str = "u2netp") -> None:
        log.info("loading rembg session (%s)...", model_name)
        # onnxruntime's default thread count is "auto", which means
        # os.cpu_count() -- inside a CPU-quota-limited container that reads
        # the *host's* full core count (12, in testing), not the actual quota
        # (0.1 of one core on Render Free). The result is a dozen threads
        # fighting over a tenth of a core's worth of real CPU time, which
        # measured at 33.8s for a single cutout that takes ~0.5s unconstrained
        # -- not a hang, just catastrophic scheduling thrash. Pinning both
        # thread counts to 1 matches the single request this process actually
        # serves at a time (--workers 1) and removes the thrash entirely.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        # The actual fix for the 512MB ceiling, found by testing -- with
        # onnxruntime's arena allocator on (the default), RSS climbed toward
        # the 512MB cgroup limit within 2-3 requests on the same long-lived
        # worker process and stayed there, because the arena grows to serve
        # peak demand and does not hand pages back to the OS between calls.
        # Disabling it trades a little per-call speed for actually releasing
        # memory after each request -- measured holding steady around
        # 340-390MB across repeated calls instead of climbing to the ceiling.
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
        self._session = new_session(model_name, sess_opts=sess_opts)
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
