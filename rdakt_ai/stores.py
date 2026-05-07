"""Session stores for persisting entity mappings."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import cast

try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


class SessionStore(ABC):
    """Abstract base for session stores."""

    @abstractmethod
    def save(self, session_id: str, mapping: dict[str, str]) -> None: ...

    @abstractmethod
    def load(self, session_id: str) -> dict[str, str] | None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MemoryStore(SessionStore):
    """Ephemeral in-memory session store. Thread-safe via lock."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def save(self, session_id: str, mapping: dict[str, str]) -> None:
        with self._lock:
            self._store[session_id] = dict(mapping)

    def load(self, session_id: str) -> dict[str, str] | None:
        with self._lock:
            data = self._store.get(session_id)
            return dict(data) if data is not None else None

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def close(self) -> None:
        with self._lock:
            self._store.clear()


class SQLiteStore(SessionStore):
    """Persistent session store backed by SQLite.

    Uses the standard library ``sqlite3`` module — no extra dependencies required.
    Thread-safe via sqlite3's serialized mode (``check_same_thread=False``).
    WAL journal mode is enabled for better concurrent read/write performance.
    """

    def __init__(self, path: str = "rdakt_sessions.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, "
            "mapping TEXT NOT NULL, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    def save(self, session_id: str, mapping: dict[str, str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, mapping, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, json.dumps(mapping)),
        )
        self._conn.commit()

    def load(self, session_id: str) -> dict[str, str] | None:
        row = self._conn.execute(
            "SELECT mapping FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class RedisStore(SessionStore):
    """Persistent session store backed by Redis.

    Requires the ``redis`` extra: ``pip install rdakt-ai[redis]``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl: int | None = None,
        prefix: str = "rdakt:",
    ) -> None:
        if redis is None:
            raise ImportError("Redis support requires the 'redis' extra. Install it with: pip install rdakt-ai[redis]")
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def save(self, session_id: str, mapping: dict[str, str]) -> None:
        data = json.dumps(mapping)
        if self._ttl is not None:
            self._client.setex(self._key(session_id), self._ttl, data)
        else:
            self._client.set(self._key(session_id), data)

    def load(self, session_id: str) -> dict[str, str] | None:
        data = cast(str | None, self._client.get(self._key(session_id)))
        if data is None:
            return None
        return cast(dict[str, str], json.loads(data))

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def close(self) -> None:
        self._client.close()
