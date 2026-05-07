"""SQLiteStore: persist session mappings to SQLite.

Demonstrates multi-turn anonymization with persistent storage —
the mapping survives across requests.
"""

from __future__ import annotations

import tempfile

from rdakt_ai import Anonymizer, Entity, RdaktSession, SQLiteStore


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    with SQLiteStore(path=db_path) as store:
        # First turn: detect, anonymize, save
        session = RdaktSession(session_id="conv-1")
        entities = [Entity(value="john@example.com", type="EMAIL", start=12, end=28)]
        anonymizer = Anonymizer()
        text = "Contact me: john@example.com"
        anon_text, mapping = anonymizer.anonymize(text, entities)
        session.add_mappings(mapping)
        store.save(session.session_id, session.entity_map)
        print(f"Anonymized: {anon_text}")

        # Second turn: load existing mapping, deanonymize
        loaded = store.load("conv-1")
        assert loaded is not None
        session2 = RdaktSession(session_id="conv-1")
        session2.add_mappings(loaded)
        restored = session2.deanonymize(f"I sent an email to {list(mapping.keys())[0]}")
        print(f"Restored:   {restored}")

    print("Done.")


if __name__ == "__main__":
    main()
