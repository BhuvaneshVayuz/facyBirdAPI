from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.photo_fetch import PhotoFetchError

FIXTURES = Path(__file__).parent / "fixtures"
#: A real single-face photo, gitignored -- same reasoning as celeb-lookalike's
#: image corpus: real photos of real people do not belong in version control.
#: Drop one at tests/fixtures/single-face.jpg to run the true integration
#: test; it skips (not fails) when absent, e.g. in CI.
SINGLE_FACE = FIXTURES / "single-face.jpg"


@pytest.fixture(scope="module")
def client():
    # Models load inside api.py's `lifespan`, not at module level (see that
    # file's comment on why) -- a bare `TestClient(app)` does NOT trigger
    # lifespan startup/shutdown, only `with TestClient(app) as c:` does.
    # Module-scoped so the ~3s local model load happens once for this file,
    # not once per test.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_unreadable_image_rejected(client, monkeypatch):
    monkeypatch.setattr("src.api.fetch_photo_bytes", lambda url: b"not an image")
    r = client.post("/face-cutout", json={"photo_url": "https://example.com/photo.jpg"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unreadable_image"


def test_no_face_detected(client, monkeypatch):
    monkeypatch.setattr("src.api.fetch_photo_bytes", lambda url: _tiny_valid_jpeg())
    monkeypatch.setattr(app.state.face_counter, "detect_boxes", lambda bgr: [])
    r = client.post("/face-cutout", json={"photo_url": "https://example.com/photo.jpg"})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_face"


def test_multiple_faces_detected(client, monkeypatch):
    monkeypatch.setattr("src.api.fetch_photo_bytes", lambda url: _tiny_valid_jpeg())
    monkeypatch.setattr(
        app.state.face_counter,
        "detect_boxes",
        lambda bgr: [(0, 0, 10, 10), (20, 20, 30, 30)],
    )
    r = client.post("/face-cutout", json={"photo_url": "https://example.com/photo.jpg"})
    assert r.status_code == 422
    body = r.json()["detail"]
    assert body["error"] == "multiple_faces"
    assert body["count"] == 2


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("invalid_url", 400),
        ("photo_unavailable", 400),
        ("file_too_large", 413),
    ],
)
def test_photo_fetch_errors_map_to_correct_status(client, monkeypatch, code, expected_status):
    def raise_it(url):
        raise PhotoFetchError(code)

    monkeypatch.setattr("src.api.fetch_photo_bytes", raise_it)
    r = client.post("/face-cutout", json={"photo_url": "https://example.com/photo.jpg"})
    assert r.status_code == expected_status
    assert r.json()["detail"]["error"] == code


@pytest.mark.skipif(not SINGLE_FACE.exists(), reason="no single-face fixture present")
def test_happy_path_real_photo(client, monkeypatch):
    # Mocks only the download step -- everything downstream (decode, YuNet,
    # rembg) runs for real against the real fixture bytes, same as before
    # this endpoint took a URL instead of an upload.
    photo_bytes = SINGLE_FACE.read_bytes()
    monkeypatch.setattr("src.api.fetch_photo_bytes", lambda url: photo_bytes)

    r = client.post("/face-cutout", json={"photo_url": "https://example.com/photo.jpg"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000  # not an empty/broken PNG
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def _tiny_valid_jpeg() -> bytes:
    """A minimal real JPEG (solid colour) -- enough for cv2.imdecode to
    succeed, so these tests exercise the face-count branch, not the
    unreadable-image branch."""
    import cv2
    import numpy as np

    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()
