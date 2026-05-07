"""Streaming deanonymization — chunk-by-chunk token replacement."""

from rdakt_ai import RdaktSession

print("=== Streaming Deanonymization ===")
print()

session = RdaktSession(
    session_id="demo",
    entity_map={
        "<EMAIL_1>": "john@example.com",
        "<PERSON_1>": "John Smith",
    },
)

# Simulate streaming chunks from an LLM
chunks = [
    "Hello ",
    "<PERS",  # partial token split across chunks
    "ON_1>, ",
    "your email is <EMAIL_1>.",
]

print("Chunks received and deanonymized:")
for chunk in chunks:
    fragments = session.deanonymize_chunk(chunk)
    output = "".join(fragments)
    print(f"  chunk={chunk!r:30s} -> output={output!r}")

# Flush any remaining buffer
remaining = session.flush()
if remaining:
    print(f"  flush -> {''.join(remaining)!r}")
