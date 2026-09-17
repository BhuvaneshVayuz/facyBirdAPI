# The deployed API. opencv-python-headless (not plain opencv-python, unlike
# celeb-lookalike's image) skips the GUI/highgui build, so this does not need
# libGL/libglib -- nothing here ever opens a window or even could.
#
# Both models run through OpenCV's DNN backend (cv2.dnn), not onnxruntime --
# see cutout.py's module docstring for why: onnxruntime + rembg's own
# dependency chain (pymatting/scipy/scikit-image/numba/llvmlite/networkx)
# added ~470MB to this image, which made pulling it onto a fresh Render Free
# instance take ~6 minutes -- longer than Render's fixed 5-minute port-scan
# deploy timeout, causing intermittent deploy/cold-start failures. Dropping
# that chain in favour of the same cv2.dnn backend YuNet already uses (zero
# new dependencies, since opencv is already required) is the actual fix.
#
# Two model weights get baked in at build time, same reasoning as
# celeb-lookalike's ensure_models() step: without this, the first request
# after every cold start (this runs on Render's free tier, which sleeps)
# would pay for a live download instead of just loading local bytes.
#   - YuNet (face_detect.py)  ~230KB, MIT
#   - u2netp (cutout.py)      ~4.7MB, Apache 2.0
# See LICENSES.md for how both were verified.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # OpenCV auto-detects thread count from the container's *visible* CPU
    # count (12 in testing on Render Free), not its actual cgroup quota (0.1
    # of one core) -- left unset, that's several threads thrashing over a
    # tenth of a core's worth of real CPU time rather than just running
    # single-threaded, which measured dramatically slower under that
    # constraint, not faster.
    OPENCV_NUM_THREADS=1

WORKDIR /app

# Dependencies first, against a stub package tree, so editing src/ does not
# reinstall OpenCV on every push.
#
# The install is editable, and that is load-bearing rather than a habit:
# config.MODELS_DIR is derived from src/'s own location, so a normal install
# would put the package in site-packages and MODELS_DIR would resolve there --
# somewhere the baked-in weights below do not exist.
COPY pyproject.toml ./
RUN mkdir -p src && touch src/__init__.py && pip install -e .

COPY src/ ./src/

RUN python -c "from src.face_detect import ensure_yunet; ensure_yunet()"
RUN python -c "from src.cutout import ensure_u2netp; ensure_u2netp()"

EXPOSE 8000

# Render supplies $PORT. One worker: each worker loads its own copy of both
# models.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
