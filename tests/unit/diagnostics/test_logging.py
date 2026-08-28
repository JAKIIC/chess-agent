import io
import json
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
    assert '"api_key":"[REDACTED]"' in result
    assert '"API_KEY":"[REDACTED]"' in result
    assert '"Api_Key":"[REDACTED]"' in result


def test_redact_decodes_unicode_escaped_json_keys_and_bearer_values() -> None:
    text = (
        r'{"api\u005fkey":"encoded-secret","\u0061pi_key":"encoded-prefix-secret",'
        r'"Authorization":"Bearer\u0020sk-escaped-bearer",'
        r'"outer":"{\"api_key\":\"embedded-secret\"}"}'
    )
    result = redact(text)
    for secret in ("encoded-secret", "encoded-prefix-secret", "sk-escaped-bearer", "embedded-secret"):
        assert secret not in result


def test_redact_decodes_top_level_and_embedded_json_string_literals() -> None:
    messages = (
        r'"{\"api\u005fkey\":\"encoded-secret\"}"',
        r'payload "{\"\u0061pi_key\":\"encoded-prefix-secret\"}"',
        r'"{\"Authorization\":\"Bearer\u0020sk-escaped-bearer\"}"',
    )
    for message in messages:
        result = redact(message)
        assert all(secret not in result for secret in ("encoded-secret", "encoded-prefix-secret", "sk-escaped-bearer"))


def test_redact_masks_non_sk_bearer_and_preserves_json_parseability() -> None:
    direct = redact(r'{"Authorization":"Bearer\u0020plain-token"}')
    assert json.loads(direct)["Authorization"] == "Bearer [REDACTED]"
    encoded = redact(r'"{\"Authorization\":\"Bearer\u0020plain-token\"}"')
    assert json.loads(json.loads(encoded))["Authorization"] == "Bearer [REDACTED]"


def test_logger_redacts_non_sk_bearer_in_exception(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    try:
        raise RuntimeError(r'{"Authorization":"Bearer\u0020exception-token"}')
    except RuntimeError:
        logger.exception("bearer failure")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    assert "exception-token" not in content


def test_redact_masks_malformed_bearer_and_api_key_fallbacks() -> None:
    text = '''Authorization: Bearer "quoted-plain-token"
Authorization: Bearer {braced-plain-token}
{"api_key": "truncated-plain-secret
{"api_key":unquoted-plain-secret
valid {"api_key":"[REDACTED]"}'''
    result = redact(text)
    for secret in ("quoted-plain-token", "braced-plain-token", "truncated-plain-secret", "unquoted-plain-secret"):
        assert secret not in result
    assert 'valid {"api_key":"[REDACTED]"}' in result
    assert 'Bearer "[REDACTED]"' == redact('Bearer "quoted-token"')
    assert "Bearer {[REDACTED]}" == redact("Bearer {braced-token}")


def test_redact_masks_secrets_after_redacted_marker_prefix() -> None:
    for text in ('{"api_key":"[REDACTED]actual-secret', '{"api_key":[REDACTED]actual-secret'):
        assert "actual-secret" not in redact(text)


def test_redact_scans_unicode_malformed_keys_and_bearer_values() -> None:
    cases = (
        r'{"api\u005fkey":unquoted-unicode-secret}',
        r'{"Authorization":"Bearer\u0020truncated-bearer-secret}',
        r'Authorization: Bearer "prefix\\"escaped-quote-secret',
        r'{"\u0061\u0070\u0069\u005f\u006b\u0065\u0079":fully-escaped-secret}',
    )
    result = redact("\n".join(cases))
    for secret in ("unquoted-unicode-secret", "truncated-bearer-secret", "escaped-quote-secret", "fully-escaped-secret"):
        assert secret not in result


def test_logger_exception_scans_unicode_malformed_values(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    try:
        raise RuntimeError(r'{"api\u005fkey":exception-unicode-secret}')
    except RuntimeError:
        logger.exception("unicode malformed value")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    assert "exception-unicode-secret" not in content


def test_external_handler_cannot_leak_after_shutdown_and_reconfigure(tmp_path: Path) -> None:
    first = configure_logging(tmp_path)
    stream = io.StringIO()
    external = logging.StreamHandler(stream)
    first.addHandler(external)
    shutdown_logging(tmp_path)
    logger = configure_logging(tmp_path)
    logger.error("Authorization: Bearer external-secret {\"api_key\":\"api-secret\"}")
    try:
        raise RuntimeError("Bearer " + "exception-" + "secret {\"api_key\":\"exception-" + "api" + "\"}")
    except RuntimeError:
        logger.exception("failure")
    assert "external-secret" not in stream.getvalue()
    assert "api-secret" not in stream.getvalue()
    assert "exception-secret" not in stream.getvalue()
    assert "exception-api" not in stream.getvalue()
    logger.removeHandler(external)
    external.close()


def test_repeated_configuration_does_not_duplicate_logger_filters(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    filters = list(logger.filters)
    configure_logging(tmp_path)
    assert logger.filters == filters


def test_redact_scans_unicode_escaped_bearer_word() -> None:
    text = r'Authorization: \u0042\u0065\u0061\u0072\u0065\u0072\u0020unicode-word-secret'
    assert "unicode-word-secret" not in redact(text)


def test_logger_exception_scans_unicode_escaped_bearer_word(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    try:
        raise RuntimeError(r'Authorization: \u0042\u0065\u0061\u0072\u0065\u0072\u0020exception-unicode-word')
    except RuntimeError:
        logger.exception("escaped bearer")
    for handler in logger.handlers:
        handler.flush()
    assert "exception-unicode-word" not in (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")


def test_redact_masks_malformed_quoted_api_key_suffix() -> None:
    assert "quoted-secret-suffix" not in redact('{"api_key":"[REDACTED]quoted-secret-suffix')


def test_logger_exception_redacts_malformed_fallbacks(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    try:
        raise RuntimeError(
            'Authorization: Bearer "exception-quoted"\n'
            'Authorization: Bearer {exception-braced}\n'
            '{"api_key":exception-api-secret}'
        )
    except RuntimeError:
        logger.exception("malformed security values")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    for secret in ("exception-quoted", "exception-braced", "exception-api-secret"):
        assert secret not in content


def test_logger_redacts_double_encoded_json_in_direct_and_exception_messages(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    logger.error(r'"{\"api\u005fkey\":\"direct-double-encoded\"}"')
    try:
        raise RuntimeError(r'"{\"Authorization\":\"Bearer\u0020sk-exception-double\"}"')
    except RuntimeError:
        logger.exception("double-encoded failure")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "xiangqi-agent.log").read_text(encoding="utf-8")
    assert "direct-double-encoded" not in content
    assert "sk-exception-double" not in content


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


def test_shutdown_leaves_caller_owned_handler_attached(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)
    external = logging.StreamHandler()
    logger.addHandler(external)
    shutdown_logging(tmp_path)
    assert external in logger.handlers
    external.close()
    logger.removeHandler(external)
