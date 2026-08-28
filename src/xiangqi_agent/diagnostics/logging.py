from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import re
from pathlib import Path
from typing import Any

_BEARER_WORD = re.compile(
    r"(?i)(?:b|\\u0042)(?:e|\\u0065)(?:a|\\u0061)(?:r|\\u0072)(?:e|\\u0065)(?:r|\\u0072)"
)
_SK_SECRET = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._~+/=-]*")
_API_KEY_CHAR = r"(?:a|\\u0061)(?:p|\\u0070)(?:i|\\u0069)(?:_|\\u005f)(?:k|\\u006b)(?:e|\\u0065)(?:y|\\u0079)"
_API_KEY_FALLBACK = re.compile(
    rf'(?i)((?:["\']|\\u0022|\\u0027)?(?:api_key|{_API_KEY_CHAR})(?:["\']|\\u0022|\\u0027)?\s*:\s*)([^\r\n]*)'
)
_API_KEY_LITERAL_FALLBACK = re.compile(r'(?i)(["\']api_key["\']\s*:\s*)([^\r\n]*)')


def redact(text: str) -> str:
    """Replace credential values while preserving surrounding message structure."""
    return _redact_surface(_redact_json_fragments(text))


def _redact_surface(text: str) -> str:
    result = _redact_bearer_values(text)
    for _ in range(2):
        result = _API_KEY_FALLBACK.sub(_mask_api_key_fallback, result)
        result = _API_KEY_LITERAL_FALLBACK.sub(_mask_api_key_fallback, result)
    return _SK_SECRET.sub("[REDACTED]", result)


def _redact_bearer_values(text: str) -> str:
    result: list[str] = []
    cursor = 0
    for match in _BEARER_WORD.finditer(text):
        if match.start() < cursor:
            continue
        position = match.end()
        while position < len(text):
            if text[position].isspace():
                position += 1
            elif text.startswith("\\u", position) and re.fullmatch(r"[0-9a-fA-F]{4}", text[position + 2 : position + 6]):
                position += 6
            elif text.startswith("\\t", position) or text.startswith("\\r", position) or text.startswith("\\n", position):
                position += 2
            else:
                break
        if position == match.end():
            continue
        if text.startswith("[REDACTED]", position):
            continue
        end = position
        opening = text[position:position + 1]
        if opening in ('"', "'"):
            end += 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                elif text[end] == opening:
                    end += 1
                    if end < len(text) and text[end] not in " \t\r\n,;}]":
                        end = text.find("\n", end)
                        end = len(text) if end < 0 else end
                    break
                elif text[end] in "\r\n":
                    break
                else:
                    end += 1
        elif opening in "[{":
            closing = "]" if opening == "[" else "}"
            close = text.find(closing, end + 1)
            end = len(text) if close < 0 else close + 1
        else:
            while end < len(text) and text[end] not in "\r\n,;":
                end += 1
            while end > position and text[end - 1].isspace():
                end -= 1
        result.append(text[cursor:position])
        if opening in ('"', "'") and end <= len(text) and end > position and text[end - 1] == opening:
            result.append(opening + "[REDACTED]" + opening)
        elif opening in "[{" and end <= len(text) and end > position and text[end - 1] == ("]" if opening == "[" else "}"):
            result.append(opening + "[REDACTED]" + text[end - 1])
        else:
            result.append("[REDACTED]")
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def _mask_api_key_fallback(match: re.Match[str]) -> str:
    value = match.group(2).strip()
    if re.match(r'^"?\[REDACTED\]"?(?:\s*[,}\]]|\s*$)', value):
        return match.group(0)
    return match.group(1) + "[REDACTED]"


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
        if record.exc_info is not None:
            exc_type, exc_value, traceback = record.exc_info
            if exc_type is not None and exc_value is not None:
                safe = _RedactedException(redact(str(exc_value))).with_traceback(traceback)
                record.exc_info = (exc_type, safe, traceback)
            record.exc_text = None
        return True


class _RedactedException(Exception):
    """Exception wrapper retaining the original traceback without its message."""


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
    if not any(isinstance(item, _RedactionFilter) for item in logger.filters):
        logger.addFilter(_RedactionFilter())
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
