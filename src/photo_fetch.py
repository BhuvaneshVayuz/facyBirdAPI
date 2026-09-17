"""Server-side download of a host-supplied profile photo URL.

Moved here from the client on purpose: the browser fetching the photo
directly requires the photo's host (an arbitrary S3 bucket, in practice) to
have CORS configured correctly, which is out of this project's control and
did break in production. A server-to-server download has no CORS concept at
all, so this sidesteps that class of failure permanently -- the tradeoff is
that this server now makes outbound requests to a URL supplied by the host
page rather than a user directly, so it gets basic SSRF guards even though
the caller is semi-trusted (a host app's own stored profile photo URL, not
free-form public input): reject non-http(s) schemes, and reject hostnames
that resolve to a private/loopback/link-local address (the classic SSRF
target being a cloud metadata endpoint like 169.254.169.254).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests

from .config import settings


class PhotoFetchError(Exception):
    """Raised with a typed `code` matching the API's error vocabulary."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _resolves_to_public_address(hostname: str) -> bool:
    """True only if every address this hostname resolves to is a normal
    public address -- false for anything private/loopback/link-local/
    reserved, and false (safe default) if resolution fails outright."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch_photo_bytes(url: str) -> bytes:
    """Download `url` server-side, enforcing scheme/host/size limits.

    Raises PhotoFetchError with code 'invalid_url' (bad scheme, unresolvable
    or private host) or 'photo_unavailable' (timeout, connection error,
    non-2xx response, or the download exceeded max_upload_bytes).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PhotoFetchError("invalid_url")
    if not _resolves_to_public_address(parsed.hostname):
        raise PhotoFetchError("invalid_url")

    try:
        with requests.get(url, timeout=settings.fetch_timeout_seconds, stream=True) as resp:
            if not resp.ok:
                raise PhotoFetchError("photo_unavailable")
            content_length = resp.headers.get("content-length")
            if content_length is not None and int(content_length) > settings.max_upload_bytes:
                raise PhotoFetchError("file_too_large")

            chunks = []
            total = 0
            for chunk in resp.iter_content(1 << 16):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise PhotoFetchError("file_too_large")
                chunks.append(chunk)
            return b"".join(chunks)
    except PhotoFetchError:
        raise
    except requests.RequestException:
        raise PhotoFetchError("photo_unavailable")
