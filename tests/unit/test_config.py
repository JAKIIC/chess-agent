import json
from pathlib import Path

import pytest

from xiangqi_agent.config import AppSettings, SecretStore


def test_settings_have_safe_defaults(tmp_path: Path) -> None:
    settings = AppSettings.default(tmp_path)
    assert settings.capture_fps == 2
    assert settings.animation_fps == 10
    assert settings.stable_frames == 3
    assert settings.stable_window_ms == 600
    assert settings.engine_movetime_fast_ms == 500
    assert settings.engine_movetime_deep_ms == 3000
    assert settings.diagnostic_retention_days == 7
    assert settings.deepseek_timeout_seconds == 12
    assert settings.save_diagnostic_images is False
    assert settings.deepseek_model == "deepseek-v4-flash"


def test_load_missing_file_and_round_trip_non_sensitive_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings.load(path)
    assert settings.capture_fps == 2
    settings.save()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["capture_fps"] == 2
    assert "api_key" not in json.dumps(data).lower()
    assert AppSettings.load(path).deepseek_model == "deepseek-v4-flash"


def test_invalid_json_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="settings"):
        AppSettings.load(path)


def test_validation_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AppSettings.model_validate({"capture_fps": 0})
    with pytest.raises(ValueError):
        AppSettings.model_validate({"diagnostic_retention_days": -1})
    for payload in (
        {"capture_fps": "5"},
        {"save_diagnostic_images": 1},
        {"save_diagnostic_images": "false"},
    ):
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="settings"):
            AppSettings.load(path)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"api_key": "never-store"}', encoding="utf-8")
    with pytest.raises(ValueError, match="settings"):
        AppSettings.load(path)


def test_save_uses_atomic_replace_and_cleans_failed_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = AppSettings.default(tmp_path)
    target = tmp_path / "settings.json"
    target.write_text('{"capture_fps": 99}\n', encoding="utf-8")
    real_replace = __import__("os").replace

    def fail_replace(source: str, destination: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("xiangqi_agent.config.os.replace", fail_replace)
    with pytest.raises(OSError):
        settings.save()
    assert list(tmp_path.glob(".settings-*.tmp")) == []
    assert target.read_text(encoding="utf-8") == '{"capture_fps": 99}\n'
    monkeypatch.setattr("xiangqi_agent.config.os.replace", real_replace)


def test_secret_store_only_delegates_to_monkeypatched_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | None]] = []
    fake = type("FakeKeyring", (), {
        "get_password": staticmethod(lambda service, username: calls.append((service, username, None)) or "test-key"),
        "set_password": staticmethod(lambda service, username, value: calls.append((service, username, value))),
        "delete_password": staticmethod(lambda service, username: calls.append((service, username, "deleted"))),
    })
    monkeypatch.setitem(__import__("sys").modules, "keyring", fake)
    store = SecretStore()
    assert store.get_deepseek_key() == "test-key"
    store.set_deepseek_key("new-test-key")
    store.delete_deepseek_key()
    assert calls == [
        ("xiangqi-learning-agent", "deepseek-api-key", None),
        ("xiangqi-learning-agent", "deepseek-api-key", "new-test-key"),
        ("xiangqi-learning-agent", "deepseek-api-key", "deleted"),
    ]
    with pytest.raises(ValueError):
        store.set_deepseek_key("  ")
