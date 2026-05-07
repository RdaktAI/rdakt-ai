"""Onboarding CLI — what `rdakt-ai validate` and `rdakt-ai show` do under the hood.

The wizard itself is interactive (intentionally — it's the "first five
minutes" UX), so this example exercises the two non-interactive commands
on a synthetic in-memory sample. Both reuse the same OntologyApplier and
RegexDetector that the runtime middleware uses, so what you see here is
exactly what production would do.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rdakt_ai.cli_helpers import (
    analyze_sample,
    resolved_config_dict,
    summarise_hits,
    validate_paths,
)
from rdakt_ai.config import (
    FieldRule,
    OntologyConfig,
    RdaktConfig,
    SyntheticReplacement,
)

# A typical OpenAI-style chat request body, with one structured field
# (`metadata.user_id`) layered on top of the free-text content.
sample_body = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Hi, my email is alice@example.com",
            "metadata": {"user_id": "internal-9f8a3c"},
        }
    ],
}

# ---- 1. analyze_sample — what the wizard shows after you point it at a JSON sample
print("=== analyze_sample ===")
hits = analyze_sample(sample_body)
for (entity_type, path), count in sorted(summarise_hits(hits).items()):
    print(f"  {entity_type}: {count} instance(s) at {path}")
print()

# ---- 2. validate_paths — exactly what `rdakt-ai validate` does
print("=== validate_paths ===")
config = RdaktConfig(
    ontology=OntologyConfig(
        fields=[
            FieldRule(
                path="$.messages[*].metadata.user_id",
                replace_with_synthetic=SyntheticReplacement(
                    type="USER_ID", format="user-{n:06d}"
                ),
            )
        ]
    )
)

with tempfile.TemporaryDirectory() as tmpdir:
    sample_file = Path(tmpdir) / "sample.json"
    sample_file.write_text(json.dumps(sample_body))

    report = validate_paths([sample_file], config)
    for file_report in report.files:
        print(f"  {file_report.path.name}")
        for row in file_report.redactions:
            print(f"    {row.type:<10} {row.original!r} -> {row.replacement!r} at {row.path}")
    print(
        f"  → {report.total_redactions} redaction(s), {report.error_count} error(s)"
    )
print()

# ---- 3. resolved_config_dict — what `rdakt-ai show` prints
print("=== resolved_config_dict ===")
resolved = resolved_config_dict(None)  # no path → defaults
print(f"  mode:               {resolved['mode']}")
print(f"  pipeline:           {resolved['pipeline']}")
print(f"  placeholders.template: {resolved['placeholders']['template']}")
print(f"  session.store:      {resolved['session']['store']}")
