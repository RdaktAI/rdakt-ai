"""Tests for session stores."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from rdakt_ai.stores import MemoryStore, RedisStore, SessionStore, SQLiteStore


class TestSessionStoreProtocol:
    def test_memory_store_is_session_store(self) -> None:
        store = MemoryStore()
        assert isinstance(store, SessionStore)


class TestMemoryStore:
    def test_save_and_load(self) -> None:
        store = MemoryStore()
        mapping = {"<PERSON_1>": "John"}
        store.save("session-1", mapping)
        loaded = store.load("session-1")
        assert loaded == mapping

    def test_load_nonexistent(self) -> None:
        store = MemoryStore()
        assert store.load("nonexistent") is None

    def test_update_existing(self) -> None:
        store = MemoryStore()
        store.save("s1", {"<PERSON_1>": "John"})
        store.save("s1", {"<PERSON_1>": "John", "<EMAIL_1>": "john@test.com"})
        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2

    def test_delete(self) -> None:
        store = MemoryStore()
        store.save("s1", {"<PERSON_1>": "John"})
        store.delete("s1")
        assert store.load("s1") is None

    def test_delete_nonexistent(self) -> None:
        store = MemoryStore()
        store.delete("nonexistent")  # should not raise

    def test_multiple_sessions(self) -> None:
        store = MemoryStore()
        store.save("s1", {"<PERSON_1>": "John"})
        store.save("s2", {"<PERSON_1>": "Jane"})
        assert store.load("s1") == {"<PERSON_1>": "John"}
        assert store.load("s2") == {"<PERSON_1>": "Jane"}


class TestSQLiteStore:
    def test_is_session_store(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        assert isinstance(store, SessionStore)

    def test_save_and_load(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        mapping = {"<PERSON_1>": "John", "<EMAIL_1>": "john@test.com"}
        store.save("session-1", mapping)
        loaded = store.load("session-1")
        assert loaded == mapping

    def test_load_nonexistent(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        assert store.load("nonexistent") is None

    def test_update_existing(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.save("s1", {"<PERSON_1>": "John"})
        store.save("s1", {"<PERSON_1>": "John", "<EMAIL_1>": "john@test.com"})
        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2

    def test_delete(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.save("s1", {"<PERSON_1>": "John"})
        store.delete("s1")
        assert store.load("s1") is None

    def test_delete_nonexistent(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.delete("nonexistent")  # should not raise

    def test_multiple_sessions(self, tmp_path) -> None:
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.save("s1", {"<PERSON_1>": "John"})
        store.save("s2", {"<PERSON_1>": "Jane"})
        assert store.load("s1") == {"<PERSON_1>": "John"}
        assert store.load("s2") == {"<PERSON_1>": "Jane"}

    def test_table_auto_created(self, tmp_path) -> None:
        db_path = tmp_path / "fresh.db"
        assert not db_path.exists()
        store = SQLiteStore(path=str(db_path))
        store.save("s1", {"<PERSON_1>": "John"})
        assert db_path.exists()
        assert store.load("s1") == {"<PERSON_1>": "John"}

    def test_persistence_across_instances(self, tmp_path) -> None:
        db_path = str(tmp_path / "persist.db")
        store1 = SQLiteStore(path=db_path)
        store1.save("s1", {"<PERSON_1>": "John"})

        store2 = SQLiteStore(path=db_path)
        assert store2.load("s1") == {"<PERSON_1>": "John"}


class TestRedisStore:
    def test_is_session_store(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_redis.Redis.from_url.return_value = MagicMock()
            store = RedisStore()
            assert isinstance(store, SessionStore)

    def test_save_sets_key(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore(url="redis://localhost:6379")
            store.save("session-1", {"<EMAIL_1>": "john@test.com"})
            mock_client.set.assert_called_once()
            call_args = mock_client.set.call_args
            assert call_args[0][0] == "rdakt:session-1"
            stored_json = call_args[0][1]
            assert json.loads(stored_json) == {"<EMAIL_1>": "john@test.com"}

    def test_save_with_ttl(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore(url="redis://localhost:6379", ttl=3600)
            store.save("s1", {"<PERSON_1>": "John"})
            mock_client.setex.assert_called_once()
            call_args = mock_client.setex.call_args
            assert call_args[0][0] == "rdakt:s1"
            assert call_args[0][1] == 3600

    def test_save_without_ttl_uses_set(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore(url="redis://localhost:6379")
            store.save("s1", {"<PERSON_1>": "John"})
            mock_client.set.assert_called_once()
            mock_client.setex.assert_not_called()

    def test_load_existing_key(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            mock_client.get.return_value = '{"<EMAIL_1>": "john@test.com"}'
            store = RedisStore()
            loaded = store.load("session-1")
            assert loaded == {"<EMAIL_1>": "john@test.com"}
            mock_client.get.assert_called_once_with("rdakt:session-1")

    def test_load_nonexistent_returns_none(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            mock_client.get.return_value = None
            store = RedisStore()
            assert store.load("nonexistent") is None

    def test_delete(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore()
            store.delete("s1")
            mock_client.delete.assert_called_once_with("rdakt:s1")

    def test_custom_prefix(self) -> None:
        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore(prefix="myapp:")
            store.save("s1", {"<PERSON_1>": "John"})
            call_args = mock_client.set.call_args
            assert call_args[0][0] == "myapp:s1"

    def test_import_error_without_redis(self) -> None:
        with patch("rdakt_ai.stores.redis", None), pytest.raises(ImportError, match="redis"):
            RedisStore()


class TestSQLiteStoreHardening:
    def test_wal_mode_enabled(self, tmp_path):
        """SQLiteStore enables WAL journal mode."""

        store = SQLiteStore(path=str(tmp_path / "test.db"))
        result = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        store.close()

    def test_close(self, tmp_path):
        """SQLiteStore.close() closes the connection."""

        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.save("s1", {"k": "v"})
        store.close()
        # After close, operations should raise
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            store.save("s2", {"k": "v"})

    def test_context_manager(self, tmp_path):
        """SQLiteStore can be used as a context manager."""

        with SQLiteStore(path=str(tmp_path / "test.db")) as store:
            store.save("s1", {"k": "v"})
            assert store.load("s1") == {"k": "v"}
        # After exiting, connection is closed
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            store.save("s2", {"k": "v"})


class TestRedisStoreClose:
    def test_close(self):
        """RedisStore.close() closes the client."""

        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.from_url.return_value = mock_client
            store = RedisStore(url="redis://localhost:6379")
            store.close()
            mock_client.close.assert_called_once()


class TestMemoryStoreClose:
    def test_close_clears_data(self):
        """MemoryStore.close() clears all data."""
        store = MemoryStore()
        store.save("s1", {"k": "v"})
        store.close()
        assert store.load("s1") is None


class TestSQLiteStoreEdgeCases:
    def test_save_empty_mapping(self, tmp_path):
        """Saving an empty mapping works."""
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        store.save("s1", {})
        assert store.load("s1") == {}
        store.close()

    def test_save_large_mapping(self, tmp_path):
        """Saving a large mapping works."""
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        mapping = {f"<EMAIL_{i}>": f"user{i}@example.com" for i in range(1000)}
        store.save("s1", mapping)
        loaded = store.load("s1")
        assert loaded == mapping
        store.close()

    def test_unicode_values(self, tmp_path):
        """Unicode values in mappings are preserved."""
        store = SQLiteStore(path=str(tmp_path / "test.db"))
        mapping = {"<PERSON_1>": "M\u00fcller"}
        store.save("s1", mapping)
        assert store.load("s1") == mapping
        store.close()
