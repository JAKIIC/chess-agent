import logging
from pathlib import Path

from xiangqi_agent.diagnostics.logging import configure_logging, redact, shutdown_logging


def test_redact_removes_multiple_secret_forms() -> None:
    text = 'Authorization: bearer sk-first, API_KEY: "sk-second", "api_key": "third"'
    result = redact(text)
    assert "sk-first" not in result
    assert "sk-second" not in result
    assert "third" not in result
    assert "Authorization" in result
    assert redact("ordinary message") == "ordinary message"


def test_redact_api_key_values_of_all_json_forms_without_leaking_nested_values() -> None:
    text = '{"api_key": ["nested-secret", {"x": "value"}], "API_KEY": null, "Api_Key": 42}'
    result = redact(text)
    assert "nested-secret" not in result
    assert '"api_key": "[REDACTED]"' in result
    assert '"API_KEY": "[REDACTED]"' in result
    assert '"Api_Key": "[REDACTED]"' in result


def test_logger_redacts_arguments_and_direct_messages(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    logger.warning("Authorization: Bearer %s", "sk-argument")
    logger.error('payload {"API_KEY":"sk-direct"}')
    try:
        raise RuntimeError("Bearer sk-trace and {\"api_key\":\"trace-secret\"}")
    except RuntimeError:
        logger.exception("operation failed")
    logger.error("operation failed", exc_info=(RuntimeError, RuntimeError("sk-exc"), None))
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    assert "sk-argument" not in content
    assert "sk-direct" not in content
    assert "sk-trace" not in content
    assert "trace-secret" not in content
    assert "sk-exc" not in content


def test_logger_is_non_propagating_bounded_and_deduplicated(tmp_path: Path) -> None:
    first = configure_logging(tmp_path)
    second = configure_logging(tmp_path)
    assert first is second
    assert first.propagate is False
    assert len(first.handlers) == 1
    assert isinstance(first.handlers[0], logging.handlers.RotatingFileHandler)


def test_shutdown_closes_owned_handlers_and_allows_reconfigure(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    handler = logger.handlers[0]
    shutdown_logging(tmp_path)
    assert logger.handlers == []
    assert handler.stream is None
    reconfigured = configure_logging(tmp_path)
    assert len(reconfigured.handlers) == 1
