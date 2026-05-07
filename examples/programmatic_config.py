"""Programmatic configuration with Pydantic models.

Shows how to build RdaktConfig in code using EntityConfig and SessionConfig.
"""

from __future__ import annotations

import tempfile

from rdakt_ai import EntityConfig, RdaktConfig, SessionConfig, create_store

# --- 1) Minimal defaults ---
default_config = RdaktConfig()
print(f"Default mode:     {default_config.mode}")
print(f"Default pipeline: {default_config.pipeline}")
print(f"Default store:    {default_config.session_store}")

# --- 2) Full config with nested Pydantic models ---
with tempfile.TemporaryDirectory() as tmpdir:
    config = RdaktConfig(
        mode="active",
        on_error="block",
        pipeline=["regex", {"ner": {"model": "en_core_web_sm"}}],
        entities={
            "EMAIL": EntityConfig(strategy="token"),
            "PERSON": EntityConfig(strategy="synthetic"),
            "EMPLOYEE_ID": EntityConfig(pattern=r"EMP-\d{6}", strategy="token"),
        },
        session=SessionConfig(store="sqlite", path=f"{tmpdir}/sessions.db"),
    )

    print(f"\nFull config mode:     {config.mode}")
    print(f"Full config error:    {config.on_error}")
    print(f"Full config pipeline: {config.pipeline}")
    print(f"Full config store:    {config.session_store}")
    print(f"Store options:        {config.store_options}")
    print(f"Entity strategies:    {config.entity_strategies}")
    print(f"Custom patterns:      {config.custom_patterns}")

    store = create_store(config)
    print(f"Store type:           {type(store).__name__}")
    store.close()

# --- 3) Pydantic validation catches errors early ---
try:
    RdaktConfig(mode="stealth")  # type: ignore[arg-type]
except ValueError as e:
    print(f"\nValidation error (bad mode): {e!s:.80s}...")

try:
    RdaktConfig(pipeline=["unknown_detector"])
except ValueError as e:
    print(f"Validation error (bad detector): {e!s:.80s}...")

print("\nDone.")
