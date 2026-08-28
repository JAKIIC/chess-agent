import logging
from pathlib import Path

from xiangqi_agent.diagnostics.logging import configure_logging, redact


def test_redact_removes_multiple_secret_forms() -> None:
    text = 'Authorization: bearer sk-first, API_KEY: "sk-second", "api_key": "third"'
    result = redact(text)
    assert "sk-first" not in result
    assert "sk-second" not in result
    assert "third" not in result
    assert "Authorization" in result
    assert redact("ordinary message") == "ordinary message"


def test_logger_redacts_arguments_and_direct_messages(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    logger.warning("Authorization: Bearer %s", "sk-argument")
    logger.error('payload {"API_KEY":"sk-direct"}')
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    assert "sk-argument" not in content
    assert "sk-direct" not in content


def test_logger_is_non_propagating_bounded_and_deduplicated(tmp_path: Path) -> None:
    first = configure_logging(tmp_path)
    second = configure_logging(tmp_path)
    assert first is second
    assert first.propagate is False
    assert len(first.handlers) == 1
    assert isinstance(first.handlers[0], logging.handlers.RotatingFileHandler)
