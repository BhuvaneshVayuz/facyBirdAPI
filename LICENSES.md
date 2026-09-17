# Licenses

Every model weight this service loads, with its actual license and where it
came from -- same discipline as celeb-lookalike's LICENSES.md: verified by
reading each model's own source repository, not inferred from PyPI/npm
metadata (which has been wrong before, see that file's note on NudeNet).

> **Image policy.** Uploaded photos are processed in memory and never written
> to disk, and nothing about them is persisted -- this service has no
> database. The response is the cutout PNG; nothing else about the request is
> kept.

## Models

| Model | Role | License | Source | Notes |
|---|---|---|---|---|
| YuNet (`face_detection_yunet`) | face counting | **MIT** (Copyright 2020 Shiqi Yu) | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | Same model celeb-lookalike uses for detection, verified there by reading the model directory's own `LICENSE`. |
| u2netp (U²-Net, lightweight variant) | background removal | **Apache 2.0** | https://github.com/xuebinqin/U-2-Net | Verified by reading the source repo's own `LICENSE` (Apache 2.0), not rembg's PyPI page. Downloaded directly (`cutout.py`'s `ensure_u2netp()`) from the same pre-converted ONNX file rembg's own releases host (https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx, MD5-verified), run through OpenCV's DNN backend rather than through the `rembg` package -- this service does not depend on rembg's code at all, only reuses the same already-verified weight file it points at. |

**Why `u2netp` over the full `u2net`:** ~4.7MB vs ~176MB, meaningfully faster
inference, and this is a comedic game sprite rather than a professional
matting tool -- the full model's extra edge precision is not worth the size
or latency here.

**Why `cv2.dnn` and not the `rembg` package:** rembg's own top-level module
eagerly imports `pymatting`/`scipy`/`scikit-image` regardless of whether
their only consumer (an optional refinement mode this service never enables)
is ever used, and together with `onnxruntime` that chain measured at ~470MB
of a ~900MB image -- large enough that pulling it onto a fresh Render Free
instance exceeded Render's deploy timeout. `cv2.dnn` runs the identical
weight file with zero new dependencies (OpenCV is already required for
YuNet). See `cutout.py`'s module docstring for the full story.
