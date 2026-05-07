"""LangChain integration examples — three tiers of Rdakt AI usage."""

from __future__ import annotations

import os
import sys

try:
    import langchain_openai  # noqa: F401
except ImportError:
    print("langchain-openai not installed — skipping example")
    sys.exit(0)

os.environ.setdefault("OPENAI_API_KEY", "sk-rdakt-demo")

from langchain_openai import ChatOpenAI
from rdakt_ai import RdaktConfig

config = RdaktConfig()

# ── Tier 1: Low-level — create_http_clients ──────────────────────────────
#
# Returns wired (sync, async) httpx clients. You construct the model yourself.

from rdakt_ai.integrations.langchain import create_http_clients

sync_client, async_client = create_http_clients(config=config, session_key="conv-1")
model_tier1 = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=sync_client,
    http_async_client=async_client,
)
print("Tier 1 (create_http_clients): model created")

# ── Tier 2: High-level — protect_chat_model ──────────────────────────────
#
# Wraps an existing ChatOpenAI model. Uses Pydantic v2 model_copy internally.

from rdakt_ai.integrations.langchain import protect_chat_model

model = ChatOpenAI(model="gpt-4o-mini")
model_tier2 = protect_chat_model(model, config=config, session_key="conv-2")
print("Tier 2 (protect_chat_model): model created")

# ── Tier 3: Message-level — RdaktLangChainMiddleware ────────────────────
#
# Native LangChain agent middleware. Only works with create_agent().

try:
    from langchain.agents import create_agent
    from rdakt_ai.integrations.langchain import RdaktLangChainMiddleware

    agent_tier3 = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[],
        middleware=[RdaktLangChainMiddleware(config=config, session_key="conv-3")],
    )
    print("Tier 3 (RdaktLangChainMiddleware): agent created")
except ImportError:
    print("Tier 3: langchain package not installed (only langchain-openai)")

print("\nAll tiers ready. Run with a real API key to see anonymization in action.")
