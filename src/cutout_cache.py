"""Postgres-backed cache of already-computed cutouts, keyed on the photo URL.

The expensive part of /face-cutout is not the response, it's getting there:
a server-side download of the photo (up to fetch_timeout_seconds) followed by
YuNet + u2netp inference. Both are pure functions of the photo bytes, so
doing them twice for the same photo is wasted work -- and the host app mints
a *new* S3 key on every profile-photo upload, which means the URL itself
changes whenever the photo changes. That makes the URL a sound cache key and
makes invalidation free: a new photo is simply a new key that misses.

Keyed on the URL rather than on a user id on purpose. For "is this the same
image", the URL already *is* the image's identity -- a user id would add
nothing to the lookup while requiring the frontend to send a field it
currently doesn't, and `user_id` in this project comes from the host's
localStorage (see flappy-bird-web's readHostIdentity), so it's caller-
controlled and not something to key a shared cache on. The frontend contract
is therefore completely unchanged: already-deployed MF bundles get the
caching for free without being rebuilt.

Everything here is best-effort. A cache that is unreachable, misconfigured,
or outright absent must never turn a working /face-cutout into a failing one
-- so every operation swallows its own errors and the endpoint falls through
to computing the cutout exactly as it did before this module existed. With
no DATABASE_URL set at all the cache is simply disabled and the service
behaves byte-for-byte like the stateless version.

Storage lives in the same Neon database leaderboard-api uses, in its own
table with no foreign key to `users`: this endpoint gets called before a
player has ever finished a run, so that user row frequently doesn't exist
yet and an FK would reject the insert.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import psycopg

from .config import settings

log = logging.getLogger(__name__)

#: `png` and `error_code` are mutually exclusive and exactly one is always
#: set -- a row either caches a successful cutout or caches a verdict about
#: the photo that there's no point recomputing. The CHECK makes that
#: invariant the database's business rather than a convention to remember.
#:
#: last_used_at (not created_at) drives expiry, so a photo that's still being
#: played with stays cached indefinitely and only genuinely idle rows age
#: out. Every hit touches it -- see get().
SCHEMA = """
CREATE TABLE IF NOT EXISTS face_cutouts (
    cache_key    TEXT PRIMARY KEY,
    source_url   TEXT NOT NULL,
    png          BYTEA,
    error_code   TEXT,
    face_count   INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT face_cutouts_png_xor_error CHECK ((png IS NULL) <> (error_code IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_face_cutouts_last_used_at
    ON face_cutouts (last_used_at);
"""

#: Errors worth remembering: both are verdicts about a photo that decoded
#: fine, so they're deterministic for those bytes and replaying them is
#: exactly what a recompute would produce. Deliberately excludes
#: `photo_unavailable` (transient -- a host having a bad minute must not
#: poison a URL) and `unreadable_image` (a truncated or interrupted download
#: produces it too, and that says nothing about the real photo).
CACHEABLE_ERRORS = frozenset({"no_face", "multiple_faces"})


@dataclass(frozen=True)
class CachedCutout:
    """A cache hit: either `png` is set, or `error_code` is."""

    png: bytes | None
    error_code: str | None
    face_count: int | None


def make_cache_key(photo_url: str, output_size: int) -> str:
    """Hash the URL *and* the output size together.

    output_size is an env-tunable (OUTPUT_SIZE), and every cached PNG was
    rendered at whatever it was set to at the time. Folding it into the key
    means bumping it invalidates the old entries by construction instead of
    silently serving the previous dimensions forever.
    """
    return hashlib.sha256(f"{output_size}|{photo_url}".encode()).hexdigest()


class CutoutCache:
    """Best-effort cache handle. `enabled` is False when there's no database
    configured or it couldn't be reached at startup; every method is a safe
    no-op in that state."""

    def __init__(self) -> None:
        self.enabled = False

        if not settings.database_url:
            log.info("DATABASE_URL not set -- cutout cache disabled, every request recomputes")
            return

        # A database that's down at boot must not take the whole service
        # down with it: an exception escaping lifespan would abort startup
        # and leave /face-cutout returning nothing at all, which is strictly
        # worse than the uncached behaviour this is meant to improve on.
        try:
            with self._connect() as conn:
                conn.execute(SCHEMA)
                conn.commit()
        except Exception:
            log.exception("cutout cache unavailable at startup -- continuing without it")
            return

        self.enabled = True
        log.info("cutout cache ready")
        self._sweep_expired()

    def _connect(self) -> psycopg.Connection:
        # One connection per operation, no pool -- same call this project's
        # leaderboard-api makes, at the same casual scale. connect_timeout
        # matters more here than it does there: this cache sits in front of
        # a user-facing request, and a database that hangs rather than
        # refuses should cost a bounded couple of seconds before being
        # written off, not the whole request.
        return psycopg.connect(
            settings.database_url,
            connect_timeout=settings.db_connect_timeout_seconds,
        )

    def _sweep_expired(self) -> None:
        """Drop entries nothing has asked for in cache_ttl_days.

        Unbounded growth is real here rather than theoretical: a new photo
        means a new S3 key means a new row, so every profile-photo change a
        user ever makes leaves the previous row orphaned and unreachable
        forever. At a few hundred KB per PNG that reaches Neon's free-tier
        storage limit sooner than you'd like. Running this at startup (free
        tier restarts often enough to make a scheduled job unnecessary) is an
        indexed delete over a small table -- cheap, and idempotent.
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM face_cutouts WHERE last_used_at < now() - make_interval(days => %s)",
                    (settings.cache_ttl_days,),
                )
                conn.commit()
                if cur.rowcount:
                    log.info("cutout cache: swept %d expired entr(ies)", cur.rowcount)
        except Exception:
            log.exception("cutout cache sweep failed -- harmless, entries just stay")

    def get(self, cache_key: str) -> CachedCutout | None:
        """Return the cached entry for this key, or None on a miss (or on
        any failure at all -- the caller treats both the same way)."""
        if not self.enabled:
            return None
        # The whole body is inside the guard, not just the query: unpacking a
        # row is only safe while the table looks the way SCHEMA left it, and
        # a hand-altered or half-migrated table would otherwise turn a cache
        # lookup into a 500 on a request that could have been served by
        # simply recomputing.
        try:
            with self._connect() as conn:
                # UPDATE ... RETURNING rather than SELECT-then-UPDATE: the
                # read and the last_used_at touch that keeps a live entry
                # from being swept are one statement and one round trip, so
                # keeping expiry accurate costs nothing over a plain read.
                row = conn.execute(
                    """
                    UPDATE face_cutouts SET last_used_at = now()
                    WHERE cache_key = %s
                    RETURNING png, error_code, face_count
                    """,
                    (cache_key,),
                ).fetchone()
                conn.commit()

            if row is None:
                return None
            png, error_code, face_count = row
            # psycopg3 hands back `bytes` for bytea already; the cast is for
            # the memoryview some adapter configurations produce, and is free
            # when it's already bytes.
            return CachedCutout(
                png=bytes(png) if png is not None else None,
                error_code=error_code,
                face_count=face_count,
            )
        except Exception:
            log.exception("cutout cache read failed -- falling through to recompute")
            return None

    def put_png(self, cache_key: str, source_url: str, png: bytes) -> None:
        self._put(cache_key, source_url, png=png, error_code=None, face_count=None)

    def put_error(self, cache_key: str, source_url: str, error_code: str, face_count: int | None) -> None:
        if error_code not in CACHEABLE_ERRORS:
            return
        self._put(cache_key, source_url, png=None, error_code=error_code, face_count=face_count)

    def _put(
        self,
        cache_key: str,
        source_url: str,
        png: bytes | None,
        error_code: str | None,
        face_count: int | None,
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._connect() as conn:
                # DO NOTHING, not DO UPDATE: a conflict means two requests
                # for the same new photo raced and the other one already
                # stored an equivalent result (the key pins both the URL and
                # the output size, so "equivalent" is guaranteed, not hoped
                # for). Rewriting a few hundred KB to arrive at identical
                # bytes would be pure cost.
                conn.execute(
                    """
                    INSERT INTO face_cutouts (cache_key, source_url, png, error_code, face_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO NOTHING
                    """,
                    (cache_key, source_url, png, error_code, face_count),
                )
                conn.commit()
        except Exception:
            # The caller already has a perfectly good result to return --
            # failing to memoise it is not worth failing the request over.
            log.exception("cutout cache write failed -- result still returned, just not cached")
