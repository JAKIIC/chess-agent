from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

SERVICE = "xiangqi-learning-agent"
USERNAME = "deepseek-api-key"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    capture_fps: int = Field(default=2, gt=0)
    animation_fps: int = Field(default=10, gt=0)
    stable_frames: int = Field(default=3, gt=0)
    stable_window_ms: int = Field(default=600, gt=0)
    engine_movetime_fast_ms: int = Field(default=500, gt=0)
    engine_movetime_deep_ms: int = Field(default=3000, gt=0)
    diagnostic_retention_days: int = Field(default=7, ge=0)
    deepseek_timeout_seconds: int = Field(default=12, gt=0)
    save_diagnostic_images: bool = False
    deepseek_model: str = "deepseek-v4-flash"
    settings_path: Path = Field(default=Path("settings.json"), exclude=True, repr=False)

    _CONFIG_DIR_NAME: ClassVar[str] = "xiangqi-learning-agent"

    @field_validator("deepseek_model")
    @classmethod
    def _model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("deepseek_model must not be blank")
        return value

    @classmethod
    def default(cls, base_dir: Path) -> AppSettings:
        return cls(settings_path=Path(base_dir) / "settings.json")

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        if path is None:
            from platformdirs import user_config_dir

            path = Path(user_config_dir(cls._CONFIG_DIR_NAME)) / "settings.json"
        path = Path(path)
        if not path.exists():
            return cls.default(path.parent)
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("settings JSON must be an object")
            return cls.model_validate({**raw, "settings_path": path})
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid settings file {path.name}: {exc}") from exc

    def save(self, path: Path | None = None) -> None:
        target = Path(path) if path is not None else self.settings_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude={"settings_path"})
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, prefix=".settings-", suffix=".tmp", delete=False
            ) as handle:
                temp_name = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
            temp_name = None
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass


class SecretStore:
    SERVICE = SERVICE
    USERNAME = USERNAME

    def get_deepseek_key(self) -> str | None:
        import keyring

        return keyring.get_password(self.SERVICE, self.USERNAME)

    def set_deepseek_key(self, value: str) -> None:
        if not value.strip():
            raise ValueError("DeepSeek API key must not be empty")
        import keyring

        keyring.set_password(self.SERVICE, self.USERNAME, value)

    def delete_deepseek_key(self) -> None:
        import keyring

        keyring.delete_password(self.SERVICE, self.USERNAME)
