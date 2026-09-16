"""Face counting via YuNet -- the same detector celeb-lookalike uses, MIT
licensed, driven through OpenCV's DNN backend with no extra dependency.

Only detection is needed here, not recognition: this service does not care
*whose* face it is, only whether there is exactly one. That is the whole rule
(0 -> no_face, 1 -> continue, 2+ -> multiple_faces), same semantics as
celeb-lookalike's face_validation.py but without the sharpness/brightness
quality gates that module also runs -- those exist there because a blurry
face makes for a bad *identity match*; here the same face just becomes a
slightly blurry bird, which is not worth rejecting a photo over.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

log = logging.getLogger(__name__)

# opencv_zoo keeps its ONNX files in git-lfs; raw.githubusercontent.com serves
# a ~130 byte pointer file instead of the model, media.githubusercontent.com
# resolves the actual LFS object. Same URL celeb-lookalike downloads from.
_LFS_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
YUNET_URL = f"{_LFS_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

# A git-lfs pointer is tiny; a real model is not -- cheap way to catch the
# wrong URL being fetched and cached as if it were valid.
_MIN_MODEL_BYTES = 50_000

#: Longest edge fed to the detector. YuNet degrades on very large inputs;
#: downscaling first (then scaling the box back) is what celeb-lookalike's
#: benchmark found necessary to get reliable detections.
_DET_LONG_EDGE = 640


def _download(url: str, dest: Path) -> None:
    import requests

    log.info("downloading %s ...", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        if tmp.stat().st_size < _MIN_MODEL_BYTES:
            head = tmp.read_bytes()[:64]
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"{dest.name} downloaded as only {len(head)} bytes -- this looks like a "
                f"git-lfs pointer, not a model. Head: {head!r}"
            )
        tmp.replace(dest)
    log.info("  saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def ensure_yunet(models_dir: Path | None = None) -> Path:
    d = models_dir or MODELS_DIR
    yunet = d / YUNET_FILE
    if not yunet.exists() or yunet.stat().st_size < _MIN_MODEL_BYTES:
        _download(YUNET_URL, yunet)
    return yunet


class FaceCounter:
    """Thin wrapper: how many faces, and where is the biggest one.

    A lock guards detect() the same way celeb-lookalike's does -- FaceDetectorYN
    carries mutable input-size state across calls, so it is not reentrant, and
    FastAPI runs sync endpoints in a thread pool.
    """

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3) -> None:
        yunet = ensure_yunet()
        log.info("loading YuNet...")
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), score_threshold, nms_threshold, 5000,
        )
        self._lock = threading.Lock()
        log.info("YuNet ready")

    def detect_boxes(self, bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Return every detected face box (x1, y1, x2, y2), largest first."""
        h, w = bgr.shape[:2]
        scale = _DET_LONG_EDGE / max(h, w)
        if scale < 1.0:
            small = cv2.resize(
                bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale, small = 1.0, bgr

        sh, sw = small.shape[:2]
        with self._lock:
            self._detector.setInputSize((sw, sh))
            _retval, raw = self._detector.detect(small)
        if raw is None or len(raw) == 0:
            return []

        inv = 1.0 / scale
        boxes = []
        for row in raw:
            x, y, fw, fh = (float(v) * inv for v in row[:4])
            boxes.append((x, y, x + fw, y + fh))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes
