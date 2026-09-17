"""Shared model-download helper -- used by both face_detect.py (YuNet) and
cutout.py (u2netp), which each bake their model into the Docker image at
build time by calling their own ensure_*() the same way this module is used
for both."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# A truncated/redirect-to-an-HTML-error-page download is tiny; a real model
# is not -- cheap way to catch a bad URL being fetched and cached as if it
# were valid before anything tries to load it as a model.
_MIN_MODEL_BYTES = 50_000


def download(url: str, dest: Path, md5: str | None = None) -> None:
    import requests

    log.info("downloading %s ...", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        hasher = hashlib.md5() if md5 else None
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
                if hasher:
                    hasher.update(chunk)
        if tmp.stat().st_size < _MIN_MODEL_BYTES:
            head = tmp.read_bytes()[:64]
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"{dest.name} downloaded as only {len(head)} bytes -- this looks like an "
                f"error page, not a model. Head: {head!r}"
            )
        if hasher and hasher.hexdigest() != md5:
            got = hasher.hexdigest()
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{dest.name} MD5 mismatch: expected {md5}, got {got}")
        tmp.replace(dest)
    log.info("  saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def ensure(url: str, dest: Path, md5: str | None = None) -> Path:
    if not dest.exists() or dest.stat().st_size < _MIN_MODEL_BYTES:
        download(url, dest, md5)
    return dest
