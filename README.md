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

## `POST /face-cutout`

Multipart upload, field name `file`. Response is `image/png` (transparent
background) on success.

| Faces found | Response |
|---|---|
| 0 | `422 {"error": "no_face"}` |
| 1 | `200`, the cutout PNG |
| 2+ | `422 {"error": "multiple_faces", "count": N}` |
| unreadable upload | `400 {"error": "unreadable_image"}` |
| over `MAX_UPLOAD_BYTES` | `413 {"error": "file_too_large"}` |

## How a request is processed

1. **YuNet** (MIT, the same detector celeb-lookalike uses) counts faces. Not
   recognition -- this service does not care *whose* face it is, only how
   many. `0 -> no_face`, `2+ -> multiple_faces`, no "guess the subject"
   fallback for group photos, same principle as celeb-lookalike's
   `face_validation.py`.
2. **rembg** (`u2netp` weights, Apache 2.0) removes the background from the
   full frame -- run on the whole photo rather than a pre-crop, since the
   segmenter does better with torso/shoulder context than a tight face-only
   patch.
3. The result is cropped to a square centred on the detected face (2.4x the
   face box, so the sprite reads as "a head" with some hair/chin/shoulder
   room, not a tight eyes-nose-mouth rectangle) and resized to
   `OUTPUT_SIZE`.

See [LICENSES.md](LICENSES.md) for where both models come from.

## Running locally

```bash
python -m venv .venv
.venv/Scripts/activate    # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
uvicorn src.api:app --reload
```

First run downloads both model weights (~5MB combined) to `models/` and
`~/.rembg/`; subsequent runs load from that cache.

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

Image is ~900MB -- opencv, onnxruntime, and rembg's matting dependencies
(scipy/scikit-image/numba, pulled in transitively) account for most of it.
Normal for an ML-serving container; nowhere near Render's free-tier limits.
