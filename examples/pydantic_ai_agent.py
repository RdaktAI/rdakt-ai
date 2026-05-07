"""PydanticAI integration examples — three tiers of Rdakt AI usage."""

from __future__ import annotations

import os
import sys

try:
    import pydantic_ai  # noqa: F401
except ImportError:
    print("pydantic-ai not installed — skipping example")
    sys.exit(0)

os.environ.setdefault("OPENAI_API_KEY", "sk-rdakt-demo")

from rdakt_ai import RdaktConfig

config = RdaktConfig()

# ── Tier 1: Low-level — create_http_client ──────────────────────────────
#
# Returns a wired httpx.AsyncClient. You construct the provider yourself.

from rdakt_ai.integrations.pydantic_ai import create_http_client

http_client = create_http_client(config=config, session_key="conv-1")

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

agent_tier1 = Agent(
    OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(http_client=http_client)),
    system_prompt="You are a helpful assistant.",
)
print("Tier 1 (create_http_client): agent created")

# ── Tier 2: High-level — protect_agent ───────────────────────────────────
#
# Wraps an existing agent. Uses public provider constructor APIs internally.

from rdakt_ai.integrations.pydantic_ai import protect_agent

agent = Agent("openai:gpt-4o-mini", system_prompt="You are a helpful assistant.")
agent_tier2 = protect_agent(agent, config=config, session_key="conv-2")
print("Tier 2 (protect_agent): agent created")

# ── Tier 3: Message-level — RdaktCapability ─────────────────────────────
#
# Native PydanticAI capability. Intercepts at the message layer.

from rdakt_ai.integrations.pydantic_ai import RdaktCapability

agent_tier3 = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
    capabilities=[RdaktCapability(config=config, session_key="conv-3")],
)
print("Tier 3 (RdaktCapability): agent created")

print("\nAll three tiers ready. Run with a real API key to see anonymization in action.")
