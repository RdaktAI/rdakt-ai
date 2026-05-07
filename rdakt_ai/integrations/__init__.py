"""Framework integration connectors for Rdakt AI."""

from __future__ import annotations


class NotSupportedError(Exception):
    """Raised when a framework object cannot be protected.

    This typically means the framework class does not expose a public
    parameter for injecting a custom httpx client.
    """
