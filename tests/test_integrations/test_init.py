"""Tests for rdakt_ai.integrations package."""

from rdakt_ai.integrations import NotSupportedError


class TestNotSupportedError:
    def test_is_exception(self) -> None:
        err = NotSupportedError("test")
        assert isinstance(err, Exception)

    def test_message(self) -> None:
        err = NotSupportedError("ChatAnthropic is not supported")
        assert "ChatAnthropic" in str(err)

    def test_importable_from_root(self) -> None:
        from rdakt_ai import NotSupportedError as RootError

        assert RootError is NotSupportedError
