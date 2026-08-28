from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_BEARER = re.compile(r"(\bBearer\s+)[^\s,;]+", re.IGNORECASE)
_SK_SECRET = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._~+/=-]*")
_JSON_API_KEY = re.compile(
    r'(?i)(["\']api_key["\']\s*:\s*)(["\'])(.*?)(\2)'
)


def redact(text: str) -> str:
    """Replace credential values while preserving surrounding message structure."""
    result = _JSON_API_KEY.sub(r"\1\2[REDACTED]\4", text)
    result = _BEARER.sub(r"\1[REDACTED]", result)
    return _SK_SECRET.sub("[REDACTED]", result)


class _RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


_configured: dict[Path, logging.Logger] = {}


def configure_logging(log_dir: Path) -> logging.Logger:
    target_dir = Path(log_dir).resolve()
    existing = _configured.get(target_dir)
    if existing is not None:
        return existing
    target_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"xiangqi_agent.diagnostics.{target_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        target_dir / "xiangqi-agent.log", maxBytes=1_048_576, backupCount=3, encoding="utf-8"
    )
    handler.addFilter(_RedactionFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    _configured[target_dir] = logger
    return logger
