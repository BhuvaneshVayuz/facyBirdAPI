"""Cache behaviour at the endpoint level, with a stand-in for the database.

No Postgres required -- app.state.cutout_cache is swapped for an in-memory
object with the same surface, which is enough to pin down everything that
could actually regress here: that a hit skips the download and the models,
that a replayed hit is indistinguishable from a fresh response, that only
deterministic verdicts get stored, and -- most importantly -- that a cache
which fails in any way leaves the endpoint behaving exactly as it did when
this service was stateless.

The SQL itself is deliberately not exercised here; keeping this suite
runnable with nothing but `pytest` is the same tradeoff test_api.py already
makes by skipping its real-photo test rather than committing a fixture.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import settings
from src.cutout_cache import CACHEABLE_ERRORS, CachedCutout, CutoutCache, make_cache_key

PHOTO_URL = "https://example.com/pfp/v1.jpg"


class FakeCache:
    """Same surface as CutoutCache, backed by a dict."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.rows: dict[str, CachedCutout] = {}

    def get(self, cache_key):
        return self.rows.get(cache_key) if self.enabled else None

    def put_png(self, cache_key, source_url, png):
        if self.enabled:
            self.rows[cache_key] = CachedCutout(png=png, error_code=None, face_count=None)

    def put_error(self, cache_key, source_url, error_code, face_count):
        if self.enabled and error_code in CACHEABLE_ERRORS:
            self.rows[cache_key] = CachedCutout(
                png=None, error_code=error_code, face_count=face_count
            )


def _cache_with_a_dead_database(monkeypatch) -> CutoutCache:
    """A *real* CutoutCache that connected fine at startup and whose database
    has since gone away.

    Deliberately the real class rather than a raising stand-in: swallowing
    these failures is CutoutCache's own contract, not something the endpoint
    re-checks, so a fake that raises would be testing a guarantee nothing
    makes. __new__ skips __init__ only because __init__'s job is to connect.
    """
    cache = CutoutCache.__new__(CutoutCache)
    cache.enabled = True

    def dead(*a, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(cache, "_connect", dead)
    return cache


def _jpeg() -> bytes:
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def cache(monkeypatch):
    fake = FakeCache()
    monkeypatch.setattr(app.state, "cutout_cache", fake)
    return fake


@pytest.fixture
def counting_fetch(monkeypatch):
    """Counts downloads, so 'did this actually skip the work' is an
    assertion rather than an inference."""
    calls = []

    def fetch(url):
        calls.append(url)
        return _jpeg()

    monkeypatch.setattr("src.api.fetch_photo_bytes", fetch)
    return calls


@pytest.fixture
def one_face(monkeypatch):
    monkeypatch.setattr(app.state.face_counter, "detect_boxes", lambda bgr: [(8.0, 8.0, 56.0, 56.0)])


def test_second_request_for_same_photo_skips_download_and_inference(
    client, cache, counting_fetch, one_face, monkeypatch
):
    segmented = []
    real_make = app.state.cutout_maker.make
    monkeypatch.setattr(
        app.state.cutout_maker,
        "make",
        lambda *a, **k: (segmented.append(1), real_make(*a, **k))[1],
    )

    first = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    second = client.post("/face-cutout", json={"photo_url": PHOTO_URL})

    assert first.status_code == second.status_code == 200
    assert second.headers["content-type"] == "image/png"
    # Byte-identical, not merely "also a PNG" -- a cached sprite the game
    # draws must be the same sprite it would have drawn before.
    assert second.content == first.content
    assert len(counting_fetch) == 1, "cached request re-downloaded the photo"
    assert len(segmented) == 1, "cached request re-ran segmentation"


def test_changed_photo_url_misses_and_recomputes(client, cache, counting_fetch, one_face):
    client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    client.post("/face-cutout", json={"photo_url": "https://example.com/pfp/v2.jpg"})
    # The host mints a new S3 key per upload, so a changed photo is a changed
    # URL -- this is the entire invalidation story and it needs to hold.
    assert len(counting_fetch) == 2


def test_cached_no_face_replays_identically(client, cache, counting_fetch, monkeypatch):
    monkeypatch.setattr(app.state.face_counter, "detect_boxes", lambda bgr: [])

    first = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    second = client.post("/face-cutout", json={"photo_url": PHOTO_URL})

    assert first.status_code == second.status_code == 422
    assert first.json() == second.json() == {"detail": {"error": "no_face"}}
    assert len(counting_fetch) == 1


def test_cached_multiple_faces_preserves_count(client, cache, counting_fetch, monkeypatch):
    monkeypatch.setattr(
        app.state.face_counter, "detect_boxes", lambda bgr: [(0, 0, 10, 10), (20, 20, 30, 30)]
    )

    first = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    second = client.post("/face-cutout", json={"photo_url": PHOTO_URL})

    assert first.status_code == second.status_code == 422
    assert second.json() == first.json()
    assert second.json()["detail"] == {"error": "multiple_faces", "count": 2}
    assert len(counting_fetch) == 1


@pytest.mark.parametrize("code", ["photo_unavailable", "invalid_url", "file_too_large"])
def test_transient_fetch_failures_are_never_cached(client, cache, monkeypatch, code):
    """A host having a bad minute must not permanently poison that URL."""
    from src.photo_fetch import PhotoFetchError

    def raise_it(url):
        raise PhotoFetchError(code)

    monkeypatch.setattr("src.api.fetch_photo_bytes", raise_it)
    client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    assert cache.rows == {}


def test_unreadable_image_is_never_cached(client, cache, monkeypatch):
    """A truncated download decodes to nothing too, and that says nothing
    about the real photo -- it has to stay retryable."""
    monkeypatch.setattr("src.api.fetch_photo_bytes", lambda url: b"not an image")
    r = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
    assert r.status_code == 400
    assert cache.rows == {}


def test_disabled_cache_behaves_like_the_stateless_service(
    client, monkeypatch, counting_fetch, one_face
):
    monkeypatch.setattr(app.state, "cutout_cache", FakeCache(enabled=False))
    for _ in range(2):
        r = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
    assert len(counting_fetch) == 2


def test_database_dying_after_startup_degrades_to_recomputing(
    client, monkeypatch, counting_fetch, one_face
):
    """Reads and writes both fail; the endpoint neither notices nor cares."""
    broken = _cache_with_a_dead_database(monkeypatch)
    assert broken.get("anything") is None
    broken.put_png("k", PHOTO_URL, b"bytes")  # must not raise
    broken.put_error("k", PHOTO_URL, "no_face", None)  # must not raise

    monkeypatch.setattr(app.state, "cutout_cache", broken)
    for _ in range(2):
        r = client.post("/face-cutout", json={"photo_url": PHOTO_URL})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
    assert len(counting_fetch) == 2, "degraded cache should simply recompute each time"


def test_cache_unreachable_at_startup_disables_itself(monkeypatch):
    """A database that is down at boot must not abort startup -- an
    exception escaping lifespan would take /face-cutout down entirely,
    which is strictly worse than never caching at all."""
    monkeypatch.setattr(settings, "database_url", "postgresql://nobody@127.0.0.1:1/nope")
    monkeypatch.setattr(settings, "db_connect_timeout_seconds", 1)
    cache = CutoutCache()  # must not raise
    assert cache.enabled is False
    assert cache.get(make_cache_key(PHOTO_URL, 512)) is None


def test_no_database_url_disables_the_cache(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    assert CutoutCache().enabled is False


def test_cache_key_covers_url_and_output_size():
    assert make_cache_key(PHOTO_URL, 512) == make_cache_key(PHOTO_URL, 512)
    assert make_cache_key(PHOTO_URL, 512) != make_cache_key("https://example.com/other.jpg", 512)
    # OUTPUT_SIZE is env-tunable; bumping it must not serve the old size.
    assert make_cache_key(PHOTO_URL, 512) != make_cache_key(PHOTO_URL, 256)


def test_transient_codes_are_excluded_from_the_cacheable_set():
    assert CACHEABLE_ERRORS == {"no_face", "multiple_faces"}
    assert not CACHEABLE_ERRORS & {"photo_unavailable", "invalid_url", "unreadable_image"}
    assert settings.output_size > 0
