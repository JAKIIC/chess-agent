from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import re
from pathlib import Path
from typing import Any

_BEARER = re.compile(r'(\bBearer\s+)[^\s,;"\'{}\[\]]+', re.IGNORECASE)
_SK_SECRET = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._~+/=-]*")
_JSON_API_KEY = re.compile(r'(?i)(["\']api_key["\']\s*:)')
_ESCAPED_JSON_STRING = re.compile(r'(?i)((?:\\)?["\']api_key(?:\\)?["\']\s*:\s*)((?:\\)?["\'])(.*?)(?:\\)?["\']')


def redact(text: str) -> str:
    """Replace credential values while preserving surrounding message structure."""
    result = _redact_json_fragments(text)
    result = _ESCAPED_JSON_STRING.sub(r"\1\2[REDACTED]\2", result)
    return _redact_surface(result)


def _redact_surface(text: str) -> str:
    result = _BEARER.sub(r"\1[REDACTED]", text)
    return _SK_SECRET.sub("[REDACTED]", result)


def _sanitize_json(value: Any, depth: int = 0) -> Any:
    if depth > 20:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.casefold() == "api_key" else _sanitize_json(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json(item, depth + 1) for item in value]
    if isinstance(value, str):
        return _redact_surface(_redact_json_fragments(value, depth + 1))
    return value


def _redact_json_fragments(text: str, depth: int = 0) -> str:
    if depth > 20:
        return "[REDACTED]"
    decoder = json.JSONDecoder()
    result: list[str] = []
    cursor = 0
    index = 0
    while index < len(text):
        if text[index] not in "[{\"":
            index += 1
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        sanitized = _sanitize_json(value, depth)
        serialized = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        should_replace = sanitized != value or (
            isinstance(value, str) and value.lstrip().startswith(("{", "["))
        )
        if should_replace:
            result.append(text[cursor:index])
            result.append(serialized)
            cursor = index + consumed
            index = cursor
        else:
            index += 1
    result.append(text[cursor:])
    return "".join(result)


class _RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


_configured: dict[Path, tuple[logging.Logger, logging.Handler]] = {}


def configure_logging(log_dir: Path) -> logging.Logger:
    target_dir = Path(log_dir).resolve()
    existing = _configured.get(target_dir)
    if existing is not None:
        return existing[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"xiangqi_agent.diagnostics.{target_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        target_dir / "xiangqi-agent.log", maxBytes=1_048_576, backupCount=3, encoding="utf-8"
    )
    handler.addFilter(_RedactionFilter())
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    _configured[target_dir] = (logger, handler)
    return logger


def shutdown_logging(log_dir: Path | None = None) -> None:
    targets = [Path(log_dir).resolve()] if log_dir is not None else list(_configured)
    for target_dir in targets:
        owned = _configured.pop(target_dir, None)
        if owned is None:
            continue
        logger, handler = owned
        if handler in logger.handlers:
            logger.removeHandler(handler)
        handler.close()


atexit.register(shutdown_logging)
