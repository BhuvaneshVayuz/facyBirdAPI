"""facy-bird-api: turns a user's profile photo into a bird sprite.

One endpoint: give it a photo URL, get back a background-removed,
face-centred PNG cutout, or a typed error if the photo has zero or multiple
faces in it (or couldn't be fetched at all).

Results are memoised in Postgres against the photo URL (see cutout_cache.py)
so the same photo is never downloaded and segmented twice. That cache is
strictly an optimisation: with no DATABASE_URL configured, or with the
database unreachable, this service still answers every request by computing
the cutout from scratch, exactly as it did when it was stateless.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .config import settings
from .cutout import CutoutMaker
from .cutout_cache import CutoutCache, make_cache_key
from .face_detect import FaceCounter
from .photo_fetch import PhotoFetchError, fetch_photo_bytes

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
    # Constructed last and never allowed to raise (see CutoutCache) -- the
    # two models above are what this service genuinely cannot serve without,
    # and a cache that can't reach its database must not join them in being
    # able to abort startup.
    app.state.cutout_cache = CutoutCache()
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


class FaceCutoutRequest(BaseModel):
    photo_url: str


# Plain `def`, not `async def` -- everything in here (the download, cv2
# decode, YuNet, rembg) is blocking, synchronous work. FastAPI runs a sync
# endpoint in its own threadpool automatically; an async one would run
# straight on the event loop and stall every other in-flight request for as
# long as the download takes, up to fetch_timeout_seconds.
def _face_verdict_error(code: str, face_count: int | None) -> HTTPException:
    """The 422 for a photo that decoded fine but isn't usable as a sprite.

    Built in one place so a replayed cache hit is indistinguishable from a
    freshly computed verdict -- same status, same detail shape, `count`
    included for multiple_faces exactly as before.
    """
    detail: dict = {"error": code}
    if face_count is not None:
        detail["count"] = face_count
    return HTTPException(422, detail=detail)


@app.post("/face-cutout")
def face_cutout(payload: FaceCutoutRequest):
    # Checked before the download, not just before inference: on a hit this
    # skips the outbound fetch too, which is the larger and far more
    # variable half of the work.
    cache_key = make_cache_key(payload.photo_url, settings.output_size)
    cached = app.state.cutout_cache.get(cache_key)
    if cached is not None:
        if cached.error_code is not None:
            raise _face_verdict_error(cached.error_code, cached.face_count)
        return Response(content=cached.png, media_type="image/png")

    try:
        raw = fetch_photo_bytes(payload.photo_url)
    except PhotoFetchError as err:
        status = 413 if err.code == "file_too_large" else 400
        raise HTTPException(status, detail={"error": err.code}) from err

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        # Not cached: a truncated or interrupted download lands here too, and
        # that says nothing about the photo itself -- see CACHEABLE_ERRORS.
        raise HTTPException(400, detail={"error": "unreadable_image"})

    boxes = app.state.face_counter.detect_boxes(bgr)
    if not boxes:
        app.state.cutout_cache.put_error(cache_key, payload.photo_url, "no_face", None)
        raise _face_verdict_error("no_face", None)
    if len(boxes) > 1:
        app.state.cutout_cache.put_error(
            cache_key, payload.photo_url, "multiple_faces", len(boxes)
        )
        raise _face_verdict_error("multiple_faces", len(boxes))

    png_bytes = app.state.cutout_maker.make(bgr, boxes[0], settings.output_size)
    app.state.cutout_cache.put_png(cache_key, payload.photo_url, png_bytes)
    return Response(content=png_bytes, media_type="image/png")
