from __future__ import annotations

from typing import Protocol

from keyring.errors import KeyringError
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DeepSeekSecretStore(Protocol):
    def get_deepseek_key(self) -> str | None: ...

    def set_deepseek_key(self, value: str) -> None: ...

    def delete_deepseek_key(self) -> None: ...


class DeepSeekSettingsDialog(QDialog):
    def __init__(
        self, store: DeepSeekSecretStore, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("DeepSeek 设置")
        self.setModal(True)
        self.setMinimumWidth(460)

        self.status_label = QLabel()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("粘贴新的 DeepSeek API Key")
        self.save_button = QPushButton("安全保存")
        self.clear_button = QPushButton("删除已保存 Key")
        self.cancel_button = QPushButton("取消")
        self.save_button.clicked.connect(self._save)
        self.clear_button.clicked.connect(self._clear)
        self.cancel_button.clicked.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("API Key 只保存到 Windows Credential Manager，界面不会回显。"))
        root.addWidget(self.status_label)
        root.addWidget(self.key_input)
        buttons = QHBoxLayout()
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)
        root.addLayout(buttons)
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            configured = bool(self._store.get_deepseek_key())
        except KeyringError:
            self.status_label.setText("Credential Manager 暂不可用")
            return
        self.status_label.setText("状态：已配置" if configured else "状态：未配置")

    def _save(self) -> None:
        value = self.key_input.text().strip()
        if not value:
            self.status_label.setText("请输入新的 API Key")
            return
        try:
            self._store.set_deepseek_key(value)
        except KeyringError:
            self.status_label.setText("保存失败：Credential Manager 暂不可用")
            return
        finally:
            self.key_input.clear()
        self._refresh_status()
        self.accept()

    def _clear(self) -> None:
        try:
            self._store.delete_deepseek_key()
        except KeyringError:
            self.status_label.setText("删除失败或当前没有已保存的 Key")
            return
        self.key_input.clear()
        self._refresh_status()
        self.accept()
