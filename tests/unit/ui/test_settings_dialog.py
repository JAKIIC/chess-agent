from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from xiangqi_agent.ui.settings_dialog import DeepSeekSettingsDialog


class FakeSecretStore:
    def __init__(self, value: str | None) -> None:
        self.value = value

    def get_deepseek_key(self) -> str | None:
        return self.value

    def set_deepseek_key(self, value: str) -> None:
        self.value = value

    def delete_deepseek_key(self) -> None:
        self.value = None


def test_settings_never_echo_existing_key_and_clear_input_after_save(qtbot: object) -> None:
    store = FakeSecretStore("existing-secret")
    dialog = DeepSeekSettingsDialog(store)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    assert "已配置" in dialog.status_label.text()
    assert dialog.key_input.text() == ""
    assert dialog.key_input.echoMode() == QLineEdit.EchoMode.Password

    dialog.key_input.setText("new-secret")
    qtbot.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert store.value == "new-secret"
    assert dialog.key_input.text() == ""


def test_settings_can_remove_key_without_revealing_it(qtbot: object) -> None:
    store = FakeSecretStore("existing-secret")
    dialog = DeepSeekSettingsDialog(store)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    qtbot.mouseClick(dialog.clear_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert store.value is None
    assert "未配置" in dialog.status_label.text()
