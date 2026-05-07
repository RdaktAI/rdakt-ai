"""RedisStore: persist session mappings to Redis.

This example shows store factory usage but uses MemoryStore and SQLiteStore
for the actual run (to avoid requiring a Redis server).
"""

from __future__ import annotations

import tempfile

from rdakt_ai import MemoryStore, RedisStore, SQLiteStore, create_store
from rdakt_ai.config import RdaktConfig

# Show that all stores can be imported
print(f"Available stores: {MemoryStore.__name__}, {SQLiteStore.__name__}, {RedisStore.__name__}")

# Demonstrate create_store factory with memory
config_memory = RdaktConfig(session_store="memory")
store = create_store(config_memory)
print(f"Created: {type(store).__name__}")
store.close()

# Demonstrate create_store factory with sqlite
config_sqlite = RdaktConfig(session_store="sqlite", store_options={"path": tempfile.mktemp(suffix=".db")})
store2 = create_store(config_sqlite)
print(f"Created: {type(store2).__name__}")
store2.close()

print("Done.")
