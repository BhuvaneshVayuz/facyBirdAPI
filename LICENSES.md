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
| u2netp (U²-Net, lightweight variant) | background removal | **Apache 2.0** | https://github.com/xuebinqin/U-2-Net | Verified by reading the source repo's own `LICENSE` (Apache 2.0), not rembg's PyPI page. Distributed by rembg (MIT-licensed code, https://github.com/danielgatis/rembg) as a pre-converted ONNX file; the code and the weights are two separate licenses, both permissive. |

**Why `u2netp` over the full `u2net`:** ~4.7MB vs ~176MB, meaningfully faster
inference, and this is a comedic game sprite rather than a professional
matting tool -- the full model's extra edge precision is not worth the size
or latency here.
