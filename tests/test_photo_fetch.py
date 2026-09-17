import pytest

from src.photo_fetch import PhotoFetchError, fetch_photo_bytes


def test_rejects_non_http_scheme():
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("file:///etc/passwd")
    assert exc.value.code == "invalid_url"


def test_rejects_localhost_hostname():
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://localhost/photo.jpg")
    assert exc.value.code == "invalid_url"


def test_rejects_loopback_ip_literal():
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://127.0.0.1/photo.jpg")
    assert exc.value.code == "invalid_url"


def test_rejects_private_ip_literal():
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://192.168.1.1/photo.jpg")
    assert exc.value.code == "invalid_url"


def test_rejects_cloud_metadata_ip():
    # 169.254.169.254 -- the classic SSRF target (AWS/GCP/Azure instance
    # metadata endpoint, link-local so this is caught by is_link_local).
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://169.254.169.254/latest/meta-data/")
    assert exc.value.code == "invalid_url"


def test_allows_public_ip_literal(monkeypatch):
    # A literal public IP skips DNS entirely, so this doesn't depend on
    # network access the way a hostname-based happy-path test would.
    monkeypatch.setattr(
        "src.photo_fetch.requests.get",
        lambda *a, **k: _FakeResponse(b"fake bytes"),
    )
    result = fetch_photo_bytes("http://8.8.8.8/photo.jpg")
    assert result == b"fake bytes"


def test_download_error_maps_to_photo_unavailable(monkeypatch):
    import requests

    def raise_it(*a, **k):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("src.photo_fetch.requests.get", raise_it)
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://8.8.8.8/photo.jpg")
    assert exc.value.code == "photo_unavailable"


def test_non_ok_status_maps_to_photo_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.photo_fetch.requests.get",
        lambda *a, **k: _FakeResponse(b"", ok=False),
    )
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://8.8.8.8/photo.jpg")
    assert exc.value.code == "photo_unavailable"


def test_oversized_download_rejected(monkeypatch):
    from src.config import settings

    big_chunk = b"x" * (settings.max_upload_bytes + 1)
    monkeypatch.setattr(
        "src.photo_fetch.requests.get",
        lambda *a, **k: _FakeResponse(big_chunk),
    )
    with pytest.raises(PhotoFetchError) as exc:
        fetch_photo_bytes("http://8.8.8.8/photo.jpg")
    assert exc.value.code == "file_too_large"


class _FakeResponse:
    """Enough of requests.Response's interface for fetch_photo_bytes: used
    as a context manager (the real call is inside `with requests.get(...)`),
    iter_content, headers, and ok."""

    def __init__(self, body: bytes, ok: bool = True):
        self._body = body
        self.ok = ok
        self.headers = {}

    def iter_content(self, chunk_size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
