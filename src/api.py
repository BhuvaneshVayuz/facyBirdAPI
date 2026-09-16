"""facy-bird-api: turns a user's profile photo into a bird sprite.

Stateless, like fame-battle-api -- no database, nothing persisted. One
endpoint: upload a photo, get back a background-removed, face-centred PNG
cutout, or a typed error if the photo has zero or multiple faces in it.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import settings
from .cutout import CutoutMaker
from .face_detect import FaceCounter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="facy-bird-api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Loaded once at import time (not per-request, not lazily on first call) so
# the Docker image's baked-in weights are read from disk exactly once and the
# first real request does not pay a cold-load penalty on top of a cold start.
_face_counter = FaceCounter()
_cutout_maker = CutoutMaker()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/face-cutout")
async def face_cutout(file: UploadFile):
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(413, detail={"error": "file_too_large"})

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, detail={"error": "unreadable_image"})

    boxes = _face_counter.detect_boxes(bgr)
    if not boxes:
        raise HTTPException(422, detail={"error": "no_face"})
    if len(boxes) > 1:
        raise HTTPException(422, detail={"error": "multiple_faces", "count": len(boxes)})

    png_bytes = _cutout_maker.make(bgr, boxes[0], settings.output_size)
    return Response(content=png_bytes, media_type="image/png")
