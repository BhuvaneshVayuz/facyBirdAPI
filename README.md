# facy-bird-api

Turns a user's profile photo into a bird sprite for flappy-bird-web's "Facy
Bird" mode: background removed, cropped square around the face. One
endpoint, no database, nothing persisted -- same stateless posture as
fame-battle-api, not leaderboard-api's.

## Why this exists as a backend, not client-side

The first version of this was going to run entirely in the browser via
MediaPipe's WASM runtime. That runtime turned out to be ~11-12MB regardless
of which specific task you use -- inherent to running real ML inference
in-browser, not something a lighter package avoids. Moving the processing
here keeps flappy-bird-web's bundle exactly as small as it's always been (no
WASM, no model download in the browser) at the cost of a network round-trip
with the actual photo bytes, which is the honest tradeoff: the photo leaves
the browser now, whereas a pure-client approach never would have sent it
anywhere. Nothing is stored here either way.

## Why the server fetches the photo, not the browser

The client originally fetched the host's `profileImage` URL itself and
uploaded the bytes here. That depended on whatever bucket/CDN hosts those
photos sending correct CORS headers for the actual host origin -- which
worked in testing and then broke in production once a real host page hit it
(the bucket didn't send `Access-Control-Allow-Origin` for that origin, for
reasons outside this project's control). Having this service download the
photo itself sidesteps CORS entirely -- server-to-server requests have no
CORS concept -- at the cost of this service now making outbound requests to
a URL supplied by the caller, so `src/photo_fetch.py` applies basic SSRF
guards: only `http`/`https`, and the hostname must resolve to a public
address (rejects loopback/private/link-local/reserved, which is what
catches both `localhost`-style URLs and the classic cloud metadata-endpoint
target `169.254.169.254`).

## `POST /face-cutout`

JSON body: `{"photo_url": "https://..."}`. Response is `image/png`
(transparent background) on success.

| Situation | Response |
|---|---|
| 0 faces found | `422 {"error": "no_face"}` |
| exactly 1 face | `200`, the cutout PNG |
| 2+ faces found | `422 {"error": "multiple_faces", "count": N}` |
| bad/non-image bytes at that URL | `400 {"error": "unreadable_image"}` |
| non-http(s) scheme, or resolves to a private/loopback/link-local address | `400 {"error": "invalid_url"}` |
| couldn't download it (timeout, connection error, non-2xx) | `400 {"error": "photo_unavailable"}` |
| download exceeded `MAX_UPLOAD_BYTES` | `413 {"error": "file_too_large"}` |

## How a request is processed

1. **Download** the photo from `photo_url` server-side (`src/photo_fetch.py`),
   enforcing the SSRF guards above, a `FETCH_TIMEOUT_SECONDS` timeout, and
   `MAX_UPLOAD_BYTES` as a streaming size cap.
2. **YuNet** (MIT, the same detector celeb-lookalike uses) counts faces. Not
   recognition -- this service does not care *whose* face it is, only how
   many. `0 -> no_face`, `2+ -> multiple_faces`, no "guess the subject"
   fallback for group photos, same principle as celeb-lookalike's
   `face_validation.py`.
3. **u2netp** (Apache 2.0, run directly through OpenCV's DNN backend --
   `cv2.dnn`, not the `rembg` package) removes the background from the full
   frame, downscaled first to at most 800px on the long edge
   (`_MAX_SEGMENT_EDGE` in `cutout.py`) -- run on the whole photo rather than
   a pre-crop, since the segmenter does better with torso/shoulder context
   than a tight face-only patch.
4. The result is cropped to a square centred on the detected face (2.4x the
   face box, so the sprite reads as "a head" with some hair/chin/shoulder
   room, not a tight eyes-nose-mouth rectangle) and resized to
   `OUTPUT_SIZE`.

The whole endpoint is a plain (non-async) function, not `async def` -- every
step above is blocking, synchronous work (the download included), and
FastAPI runs a sync endpoint in its own threadpool automatically. An async
def here would run straight on the event loop and stall every other
in-flight request for as long as the slowest step takes.

## Render Free tier: what actually broke and what fixed it

Four separate problems showed up in production, each looking like a
different failure until measured directly (reproduced locally with
`docker run --cpus=0.1 --memory=512m`, matching Render Free's actual
allocation):

1. **Deploys intermittently failed** ("Port scan timeout reached"). Cause:
   both models loaded as module-level code, which blocks uvicorn from
   binding its port at all until loading finishes -- fixed by moving loading
   into `lifespan` (`api.py`), so the port opens in ~1-2s regardless of how
   long model loading takes afterward.
2. **A single `/face-cutout` call could take 60+ seconds and never return**,
   eventually taking the whole service down (health checks started failing
   too). Cause (back when this ran through onnxruntime, before problem #4's
   fix below): onnxruntime's default thread count is "auto", which reads
   `os.cpu_count()` -- inside this container that's the *host's* full core
   count (12, in testing), not the actual cgroup quota (0.1 of one core). A
   dozen threads fighting over a tenth of a core's worth of real CPU time is
   catastrophic scheduling thrash, not a hang: one call measured at 33.8s for
   work that takes ~0.5s unconstrained.
3. **Memory climbed toward the 512MB limit within 2-3 requests on the same
   worker and stayed there**, one bad request away from an OOM kill. Cause
   (also onnxruntime, also since removed): its memory arena allocator grows
   to serve peak demand and does not hand pages back to the OS between
   calls -- a long-lived-process accumulation problem, not primarily an
   input-image-size one (a small 396px test photo alone already peaked at
   437MB on a fresh instance).
4. **Deploys and cold starts intermittently failed even when the app itself
   was fine** -- Render's platform-level 5-minute port-scan timeout was
   consistently shorter than the ~6 minutes it took to *pull* this image
   (~900MB at the time) onto a fresh instance, before the app had even
   started. This is what problems #2 and #3 turned out to trace back to:
   both were onnxruntime's auto-threading and memory-arena behaviour, pulled
   in as a transitive dependency of `rembg` -- and `rembg`'s own top-level
   module eagerly imports `pymatting`/`scipy`/`scikit-image` regardless of
   whether their only consumer (an optional refinement mode never enabled
   here) is ever used, so that whole chain
   (`onnxruntime`+`pymatting`+`scipy`+`scikit-image`+`numba`+`llvmlite`+
   `networkx`) could not just be uninstalled after the fact without breaking
   `import rembg` outright. **Fixed by dropping `rembg` and `onnxruntime`
   entirely** and running the same `u2netp` weights directly through
   `cv2.dnn` (`cutout.py`) -- OpenCV is already a hard dependency for YuNet,
   so this adds zero new dependencies. Image dropped from ~900MB to
   **~451MB**; problems #2 and #3 disappeared as a side effect, since
   OpenCV's DNN backend doesn't have onnxruntime's auto-threading or
   arena-growth behaviour to begin with. Measured after the rewrite: fresh
   container healthy in ~7s (was up to 60s), memory settling at a flat
   ~309MB and staying there across 10 consecutive real requests (was
   climbing to the 512MB ceiling by request 3).

The input downscale (`_MAX_SEGMENT_EDGE` in `cutout.py`) is a related but
separate, still-relevant mitigation -- it caps the memory a single very
large photo (an uncompressed phone photo can be 3000-4000px) demands at the
preprocessing step, independent of everything above.

See [LICENSES.md](LICENSES.md) for where both models come from and why
`cv2.dnn` replaced `rembg`.

## Running locally

```bash
python -m venv .venv
.venv/Scripts/activate    # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
uvicorn src.api:app --reload
```

First run downloads both model weights (~5MB combined) to `models/`;
subsequent runs load from that cache.

## Testing

```bash
pytest
```

The happy-path integration test needs a real single-face photo at
`tests/fixtures/single-face.jpg` (gitignored -- real photos of real people
do not belong in version control, same as celeb-lookalike's image corpus)
and skips itself when that file is absent. Every other test mocks the face
count, so the error-path coverage (no_face / multiple_faces / bad upload /
oversized upload) runs without any image fixture at all.

## Build

```bash
docker build -t facy-bird-api .
docker run --rm -p 8000:8000 facy-bird-api
curl localhost:8000/health
```

Image is ~451MB -- opencv (which drives both models, see "Render Free tier"
above for why rembg/onnxruntime aren't dependencies here) and the Python
3.11 base account for most of it.
