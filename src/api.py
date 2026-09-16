"""facy-bird-api: turns a user's profile photo into a bird sprite.

Stateless, like fame-battle-api -- no database, nothing persisted. One
endpoint: upload a photo, get back a background-removed, face-centred PNG
cutout, or a typed error if the photo has zero or multiple faces in it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded here, not at module level -- module-level loading blocks
    # uvicorn from binding its listen socket at all until both models finish
    # loading (Python can't hand uvicorn a usable `app` until the whole
    # module has finished executing). On Render's free tier (0.1 CPU) that
    # load takes close to a minute, which showed up as repeated "No open
    # ports detected" retries during deploy -- the deploy still succeeded
    # once a scan happened to land after loading finished, but there was no
    # guarantee it would before Render gave up. Loading inside lifespan lets
    # uvicorn bind the port in ~1-2s regardless; the ASGI lifespan contract
    # already holds incoming requests (including /health) until this
    # startup block completes, so nothing can hit face_counter/cutout_maker
    # before they exist -- no readiness flag needed on top of that.
    app.state.face_counter = FaceCounter()
    app.state.cutout_maker = CutoutMaker()
    yield


app = FastAPI(title="facy-bird-api", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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

    boxes = app.state.face_counter.detect_boxes(bgr)
    if not boxes:
        raise HTTPException(422, detail={"error": "no_face"})
    if len(boxes) > 1:
        raise HTTPException(422, detail={"error": "multiple_faces", "count": len(boxes)})

    png_bytes = app.state.cutout_maker.make(bgr, boxes[0], settings.output_size)
    return Response(content=png_bytes, media_type="image/png")
