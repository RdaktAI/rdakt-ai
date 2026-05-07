"""Per-field PII ontology — JSONPath-keyed rules.

Shows the three rule shapes side-by-side on a structured request body:

* `detect` — restrict the global detector pipeline to a subset of types
  for a specific path (skip cost and false positives elsewhere).
* `replace_with_synthetic` — swap the *whole* field value with a generated
  token (use this for opaque internal IDs that should never reach the LLM).
* `detect_via_regex` + `mask_strategy: hash` — inline regex for a
  domain-specific identifier, hashed deterministically with a per-session
  salt so cross-session correlation isn't possible.

The ontology pass runs in-place over the JSON body. The session map
captures every replacement so the LLM response is restored to real values.
"""

from __future__ import annotations

import json

from rdakt_ai.anonymizer import Anonymizer
from rdakt_ai.config import (
    FieldRule,
    OntologyConfig,
    SyntheticReplacement,
)
from rdakt_ai.ontology import OntologyApplier
from rdakt_ai.session import RdaktSession

# A typical OpenAI-style chat request body, with a few structured fields
# layered on top of the free-text message content.
request_body = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Hi, my email is alice@example.com",
            "metadata": {"user_id": "internal-id-9f8a3c"},
            "context": {"case_number": "CASE-2026-000001"},
        },
        {
            "role": "user",
            "content": "Also reachable on 555-100-2000",
            "metadata": {"user_id": "internal-id-9f8a3c"},  # repeated
            "context": {"case_number": "CASE-2026-000002"},
        },
    ],
}

ontology = OntologyConfig(
    fields=[
        FieldRule(
            path="$.messages[*].content",
            detect=["EMAIL", "PHONE"],
        ),
        FieldRule(
            path="$.messages[*].metadata.user_id",
            replace_with_synthetic=SyntheticReplacement(
                type="USER_ID",
                format="user-{n:06d}",
            ),
        ),
        FieldRule(
            path="$.messages[*].context.case_number",
            detect_via_regex=r"\bCASE-\d{4}-\d{6}\b",
            detect_via_regex_as="CASE_NUMBER",
            mask_strategy="hash",
        ),
    ]
)

session = RdaktSession(session_id="demo-session")
applier = OntologyApplier(ontology, anonymizer=Anonymizer(), session=session)

owned = applier.apply(request_body)

print("=== After ontology pass ===")
print(json.dumps(request_body, indent=2))
print()
print(f"Ontology owned {len(owned)} field positions")
print()

# Repeated user_id reuses the same synthetic token within the session
u0 = request_body["messages"][0]["metadata"]["user_id"]
u1 = request_body["messages"][1]["metadata"]["user_id"]
assert u0 == u1, "repeated value should reuse the same token"
print(f"Repeated user_id stayed consistent: both rendered as {u0!r}")

# Round-trip through deanonymize: the LLM response would be a string with
# tokens in it; the session restores them to the original values.
fake_llm_response = (
    f"Confirming for {u0}: the case {request_body['messages'][0]['context']['case_number']} "
    f"is on file."
)
print()
print("=== Round-trip ===")
print(f"LLM sees:      {fake_llm_response}")
print(f"Deanonymized:  {session.deanonymize(fake_llm_response)}")
