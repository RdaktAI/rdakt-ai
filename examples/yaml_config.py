"""YAML-driven pipeline configuration.

Shows how to load a RdaktConfig from YAML and use it to build middleware.
See examples/configs/ for ready-to-use YAML config files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rdakt_ai import create_store, load_config

# Write a sample YAML config
yaml_content = """\
mode: active
on_error: warn_and_forward

pipeline:
  - regex

session:
  store: sqlite
  path: {db_path}

entities:
  EMAIL:
    strategy: token
  PERSON:
    strategy: synthetic
  EMPLOYEE_ID:
    pattern: "EMP-\\\\d{{6}}"
    strategy: token
"""

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = f"{tmpdir}/sessions.db"
    config_path = Path(tmpdir) / "rdakt.yaml"
    config_path.write_text(yaml_content.format(db_path=db_path))

    config = load_config(config_path)
    print(f"Mode:       {config.mode}")
    print(f"Pipeline:   {config.pipeline}")
    print(f"Store:      {config.session_store}")
    print(f"Strategies: {config.entity_strategies}")
    print(f"Patterns:   {config.custom_patterns}")

    store = create_store(config)
    print(f"Store type: {type(store).__name__}")
    store.close()

print("Done.")
