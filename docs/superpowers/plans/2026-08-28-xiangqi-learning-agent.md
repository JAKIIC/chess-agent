# 天天象棋学习助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 构建一个 Windows 独立桌面学习助手，实时同步微信天天象棋棋盘，使用本地 Pikafish 给出可靠分析，并使用 DeepSeek V4 提供有证据约束、可追问的中文教学解释。

**Architecture:** 应用按“窗口捕获 → 本地视觉识别 → 合法走法同步 → Pikafish 分析 → 结构化证据 → DeepSeek 解释 → PySide6 展示”分层。所有异步结果携带 position_id，只有与当前局面一致的结果才能进入 UI；识别不确定时保持最后一个已确认局面。

**Tech Stack:** Python 3.12、PySide6、windows-capture 2.0.1、OpenCV、NumPy、ONNX Runtime、Pikafish 2026-01-02、HTTPX、Pydantic、SQLite、keyring、pytest、pytest-qt、ruff、mypy、PyInstaller。

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

## Global Constraints

- 只支持 Windows 10 1903+ 与 Windows 11 64 位；Python 必须为 >=3.12,<3.13。
- 仅用于人机练习、残局训练和复盘；不提供真人在线对局辅助。
- 不模拟点击、不自动落子、不注入微信、不读取微信内存、不代理微信流量。
- 不向 DeepSeek 上传截图；只发送经过规则层和 Pikafish 验证的文本证据。
- 默认 DeepSeek 模型为 deepseek-v4-flash 非思考模式；深度复盘才使用 deepseek-v4-pro。
- Pikafish 固定为 2026-01-02，并作为独立 UCI 进程运行；默认 Threads=2、Hash=256、MultiPV=3。
- 捕获稳定采样率为 5 FPS，动画期间最高 10 FPS；连续 3 帧一致且跨度不超过 600 ms 才确认观察。
- API Key 只进入 Windows Credential Manager；日志不得记录密钥、完整截图或完整 DeepSeek 请求。
- SQLite 与诊断数据只保存在本地；诊断截图默认关闭，开启时只保存棋盘裁图并保留 7 天。
- 每个实现任务遵循 TDD：先写失败测试、确认失败、写最小实现、确认通过、再提交。

---

## Planned File Structure

~~~text
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ THIRD_PARTY_NOTICES.md
├─ assets.lock.json
├─ src/xiangqi_agent/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ bootstrap.py
│  ├─ config.py
│  ├─ application/
│  │  ├─ __init__.py
│  │  └─ controller.py
│  ├─ domain/
│  │  ├─ board.py
│  │  ├─ fen.py
│  │  ├─ rules.py
│  │  ├─ notation.py
│  │  ├─ analysis.py
│  │  └─ coach.py
│  ├─ platform/
│  │  ├─ shortcut.py
│  │  └─ windows.py
│  ├─ capture/
│  │  ├─ protocol.py
│  │  ├─ fake.py
│  │  └─ windows_capture_source.py
│  ├─ vision/
│  │  ├─ types.py
│  │  ├─ geometry.py
│  │  ├─ locator.py
│  │  ├─ model.py
│  │  ├─ recognizer.py
│  │  └─ temporal.py
│  ├─ sync/
│  │  ├─ state_machine.py
│  │  └─ service.py
│  ├─ engine/
│  │  ├─ protocol.py
│  │  ├─ process.py
│  │  └─ service.py
│  ├─ coach/
│  │  ├─ evidence.py
│  │  ├─ prompts.py
│  │  ├─ client.py
│  │  └─ service.py
│  ├─ storage/
│  │  ├─ db.py
│  │  └─ repository.py
│  ├─ diagnostics/
│  │  ├─ logging.py
│  │  └─ snapshots.py
│  └─ ui/
│     ├─ main_window.py
│     ├─ board_widget.py
│     ├─ analysis_panel.py
│     ├─ coach_panel.py
│     ├─ review_panel.py
│     └─ dialogs/
│        ├─ connect_dialog.py
│        ├─ calibration_dialog.py
│        ├─ correction_dialog.py
│        └─ settings_dialog.py
├─ scripts/
│  ├─ evaluate_models.py
│  ├─ collect_intersections.py
│  ├─ train_classifier.py
│  ├─ download_pikafish.py
│  ├─ smoke_capture.py
│  ├─ run_replay.py
│  └─ run_acceptance.py
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ ui/
│  ├─ replay/
│  └─ fixtures/
│     ├─ frames/
│     ├─ engine/
│     └─ deepseek/
└─ packaging/
   └─ xiangqi_agent.spec
~~~

文件按职责拆分：domain 不依赖 Qt、OpenCV、Windows 或网络；capture 只产出帧；vision 只产出观察；sync 是唯一可以确认局面的层；engine 与 coach 都只消费已确认的 BoardState；ui 通过信号订阅服务。

### Task 1: 初始化仓库、依赖和质量门

**Files:**
- Create: .gitignore
- Create: pyproject.toml
- Create: src/xiangqi_agent/__init__.py
- Create: src/xiangqi_agent/__main__.py
- Create: src/xiangqi_agent/bootstrap.py
- Create: tests/unit/test_package_smoke.py

**Interfaces:**
- Produces: xiangqi_agent.__version__: str
- Produces: xiangqi_agent.__main__.main() -> int
- Consumes: 无

- [ ] **Step 1: 初始化 Git 并写失败测试**

Run:

~~~powershell
git init -b main
New-Item -ItemType Directory -Force -Path tests\unit,src\xiangqi_agent | Out-Null
~~~

在 tests/unit/test_package_smoke.py 写入：

~~~python
from xiangqi_agent import __version__
from xiangqi_agent.__main__ import main


def test_package_has_version_and_entrypoint() -> None:
    assert __version__ == "0.1.0"
    assert main(["--check"]) == 0
~~~

- [ ] **Step 2: 确认测试因包不存在而失败**

Run:

~~~powershell
python -m pytest tests\unit\test_package_smoke.py -q
~~~

Expected: FAIL，错误包含 ModuleNotFoundError: No module named 'xiangqi_agent'。

- [ ] **Step 3: 写项目配置和最小入口**

pyproject.toml 必须包含：

~~~toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "xiangqi-learning-agent"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "PySide6>=6.8,<7",
  "windows-capture==2.0.1",
  "opencv-python>=4.10,<5",
  "numpy>=2,<3",
  "onnxruntime>=1.20,<2",
  "httpx>=0.28,<1",
  "pydantic>=2.9,<3",
  "keyring>=25,<26",
  "platformdirs>=4,<5",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
  "pytest-qt>=4,<5",
  "pytest-cov>=6,<7",
  "ruff>=0.9,<1",
  "mypy>=1.14,<2",
  "PyInstaller>=6,<7",
]
training = ["torch>=2.5,<3", "torchvision>=0.20,<1"]

[project.scripts]
xiangqi-agent = "xiangqi_agent.__main__:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "windows: requires a real Windows desktop session",
  "assets: requires downloaded model or engine assets",
  "acceptance: requires manual real-application verification",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
packages = ["xiangqi_agent"]
~~~

src/xiangqi_agent/__init__.py：

~~~python
__version__ = "0.1.0"
~~~

src/xiangqi_agent/__main__.py：

~~~python
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])
    if args == ["--check"]:
        return 0
    from xiangqi_agent.bootstrap import run
    return run()


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
~~~

同时创建 src/xiangqi_agent/bootstrap.py，提供返回 0 的 run()，并在 .gitignore 忽略 .venv、__pycache__、.pytest_cache、.mypy_cache、.ruff_cache、build、dist、用户数据库、日志、模型和引擎二进制。

- [ ] **Step 4: 建立虚拟环境并运行完整质量门**

Run:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy src
~~~

Expected: 三条检查全部通过。

- [ ] **Step 5: 提交基础工程**

~~~powershell
git add .gitignore pyproject.toml src tests
git commit -m "build: initialize xiangqi learning agent"
~~~

### Task 2: 定义棋盘模型、FEN 和 position_id

**Files:**
- Create: src/xiangqi_agent/domain/__init__.py
- Create: src/xiangqi_agent/domain/board.py
- Create: src/xiangqi_agent/domain/fen.py
- Create: tests/unit/domain/test_fen.py

**Interfaces:**
- Produces: Orientation(Enum), Move(dataclass), BoardState(dataclass)
- Produces: BoardState.fen: str、BoardState.position_id: str
- Produces: parse_fen(text: str) -> BoardState
- Produces: to_fen(board: BoardState) -> str
- Consumes: 无

- [ ] **Step 1: 写 FEN 往返和哈希稳定性测试**

~~~python
from xiangqi_agent.domain.fen import parse_fen, to_fen

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_fen_round_trip_and_position_id() -> None:
    board = parse_fen(START)
    assert len(board.pieces) == 90
    assert board.side_to_move == "w"
    assert to_fen(board) == START
    assert board.position_id == parse_fen(START + " - - 0 1").position_id
~~~

- [ ] **Step 2: 运行测试并确认失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\domain\test_fen.py -q

Expected: FAIL，domain.fen 尚不存在。

- [ ] **Step 3: 实现不可变数据模型和严格 FEN 解析**

board.py 定义：

~~~python
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Side = Literal["w", "b"]
VALID_PIECES = frozenset("KABNRCPkabnrcp.")


class Orientation(StrEnum):
    RED_BOTTOM = "red_bottom"
    BLACK_BOTTOM = "black_bottom"


@dataclass(frozen=True, slots=True)
class Move:
    uci: str
    from_index: int
    to_index: int
    captured: str | None = None


@dataclass(frozen=True, slots=True)
class BoardState:
    pieces: tuple[str, ...]
    side_to_move: Side
    orientation: Orientation = Orientation.RED_BOTTOM
    ply: int = 0

    def __post_init__(self) -> None:
        if len(self.pieces) != 90 or any(p not in VALID_PIECES for p in self.pieces):
            raise ValueError("board must contain exactly 90 valid intersections")

    @property
    def fen(self) -> str:
        from xiangqi_agent.domain.fen import to_fen
        return to_fen(self)

    @property
    def position_id(self) -> str:
        from hashlib import sha256
        return sha256(self.fen.encode("ascii")).hexdigest()[:32]
~~~

fen.py 必须把缺省域归一化为“棋盘 + 行棋方”，严格拒绝行数不是 10、每行展开不是 9、未知棋子和非法行棋方。内部空点统一为句点。

- [ ] **Step 4: 运行领域测试和静态检查**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\domain\test_fen.py -q
.\.venv\Scripts\ruff check src\xiangqi_agent\domain tests\unit\domain
.\.venv\Scripts\mypy src
~~~

Expected: 全部通过。

- [ ] **Step 5: 提交棋盘模型**

~~~powershell
git add src\xiangqi_agent\domain tests\unit\domain
git commit -m "feat: add immutable board and FEN model"
~~~

### Task 3: 实现规则、走法应用、局面差分和中文记谱

**Files:**
- Create: src/xiangqi_agent/domain/rules.py
- Create: src/xiangqi_agent/domain/notation.py
- Create: tests/unit/domain/test_rules.py
- Create: tests/unit/domain/test_notation.py

**Interfaces:**
- Consumes: BoardState, Move
- Produces: legal_moves(board: BoardState) -> tuple[Move, ...]
- Produces: apply_move(board: BoardState, move: Move) -> BoardState
- Produces: detect_unique_move(before: BoardState, after: BoardState) -> Move
- Produces: to_chinese(board: BoardState, move: Move) -> str

- [ ] **Step 1: 写关键规则失败测试**

测试至少覆盖：马腿阻挡、炮必须隔一子吃、象不过河、仕帅不出九宫、兵过河后可横走、将帅照面、自陷将军、吃子差分、红黑记谱和同路前后车。

~~~python
import pytest
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import detect_unique_move, legal_moves


def test_horse_leg_blocks_move() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/4P4/3NK4 w")
    assert "d0c2" not in {move.uci for move in legal_moves(board)}


def test_detect_unique_capture() -> None:
    before = parse_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w")
    after = parse_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 b")
    assert detect_unique_move(before, after).uci == "e4e5"


def test_rejects_two_moves_between_frames() -> None:
    before = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    after = parse_fen("3k5/9/9/9/9/9/9/9/9/5K3 b")
    with pytest.raises(ValueError, match="unique legal move"):
        detect_unique_move(before, after)
~~~

- [ ] **Step 2: 确认测试因规则模块不存在而失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\domain\test_rules.py tests\unit\domain\test_notation.py -q

Expected: FAIL，rules 或 notation 模块不存在。

- [ ] **Step 3: 实现确定性的规则内核**

rules.py 使用行列整数进行几何判断，先生成伪合法走法，再通过 apply_move 后的 is_in_check 过滤。公开函数签名固定为：

~~~python
def legal_moves(board: BoardState) -> tuple[Move, ...]: ...
def apply_move(board: BoardState, move: Move) -> BoardState: ...
def is_in_check(board: BoardState, side: Side) -> bool: ...
def detect_unique_move(before: BoardState, after: BoardState) -> Move: ...
~~~

实现要求：

- UCI 文件 a-i 映射列 0-8，秩 0-9 映射红方底线到黑方底线；内部数组行需要显式转换。
- detect_unique_move 遍历 before 的 legal_moves，应用后只比较 90 点棋子和行棋方；匹配数必须恰好为 1。
- apply_move 验证 move 在 legal_moves 中，并翻转 side_to_move、ply 加 1、保留 orientation。
- notation.py 根据移动方决定中文/阿拉伯数字，正确处理进退平和同路同类棋子的前中后缀。

- [ ] **Step 4: 运行全部领域测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\domain -q
.\.venv\Scripts\mypy src
~~~

Expected: 全部通过；非法局面与多步变化均明确抛出 ValueError。

- [ ] **Step 5: 提交规则内核**

~~~powershell
git add src\xiangqi_agent\domain tests\unit\domain
git commit -m "feat: add xiangqi rules and notation"
~~~

### Task 4: 配置、路径、日志脱敏和 API Key 安全存储

**Files:**
- Create: src/xiangqi_agent/config.py
- Create: src/xiangqi_agent/diagnostics/__init__.py
- Create: src/xiangqi_agent/diagnostics/logging.py
- Create: tests/unit/test_config.py
- Create: tests/unit/diagnostics/test_logging.py

**Interfaces:**
- Produces: AppSettings.load() -> AppSettings
- Produces: SecretStore.get_deepseek_key() -> str | None
- Produces: configure_logging(log_dir: Path) -> logging.Logger
- Consumes: platformdirs、keyring

- [ ] **Step 1: 写默认值和密钥脱敏测试**

~~~python
from pathlib import Path
from xiangqi_agent.config import AppSettings
from xiangqi_agent.diagnostics.logging import redact


def test_settings_have_safe_defaults(tmp_path: Path) -> None:
    settings = AppSettings.default(tmp_path)
    assert settings.capture_fps == 5
    assert settings.animation_fps == 10
    assert settings.save_diagnostic_images is False
    assert settings.deepseek_model == "deepseek-v4-flash"


def test_redact_removes_api_keys() -> None:
    text = "Authorization: Bearer sk-secret-value"
    assert "sk-secret-value" not in redact(text)
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\test_config.py tests\unit\diagnostics\test_logging.py -q

Expected: FAIL，AppSettings 和 redact 尚不存在。

- [ ] **Step 3: 实现设置与安全存储**

AppSettings 使用 Pydantic，固定包含 capture_fps=5、animation_fps=10、stable_frames=3、stable_window_ms=600、engine_movetime_fast_ms=500、engine_movetime_deep_ms=3000、diagnostic_retention_days=7、deepseek_timeout_seconds=12。

SecretStore 使用以下服务名和用户名：

~~~python
SERVICE = "xiangqi-learning-agent"
USERNAME = "deepseek-api-key"


class SecretStore:
    def get_deepseek_key(self) -> str | None:
        import keyring
        return keyring.get_password(SERVICE, USERNAME)

    def set_deepseek_key(self, value: str) -> None:
        import keyring
        keyring.set_password(SERVICE, USERNAME, value)
~~~

日志过滤器替换 Bearer token、sk- 开头的密钥和名为 api_key 的 JSON 字段。设置 JSON 只保存非敏感字段，采用临时文件 + os.replace 原子写入。

- [ ] **Step 4: 运行测试与类型检查**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\test_config.py tests\unit\diagnostics -q
.\.venv\Scripts\mypy src
~~~

Expected: 全部通过。

- [ ] **Step 5: 提交配置与日志**

~~~powershell
git add src\xiangqi_agent\config.py src\xiangqi_agent\diagnostics tests\unit
git commit -m "feat: add secure settings and redacted logging"
~~~

### Task 5: 建立 PySide6 独立窗口骨架

**Files:**
- Modify: src/xiangqi_agent/bootstrap.py
- Create: src/xiangqi_agent/ui/__init__.py
- Create: src/xiangqi_agent/ui/main_window.py
- Create: src/xiangqi_agent/ui/board_widget.py
- Create: src/xiangqi_agent/ui/analysis_panel.py
- Create: src/xiangqi_agent/ui/coach_panel.py
- Create: tests/ui/test_main_window.py

**Interfaces:**
- Consumes: BoardState
- Produces: MainWindow.set_board(board: BoardState) -> None
- Produces: MainWindow.set_sync_status(text: str, confidence: float | None) -> None
- Produces: BoardWidget.move_selected(str) Qt signal

- [ ] **Step 1: 写窗口布局失败测试**

~~~python
from PySide6.QtCore import Qt
from xiangqi_agent.ui.main_window import MainWindow


def test_main_window_contains_learning_panels(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.objectName() == "main_window"
    assert window.board_widget.objectName() == "mirror_board"
    assert window.analysis_panel.objectName() == "analysis_panel"
    assert window.coach_panel.objectName() == "coach_panel"
    assert window.windowTitle() == "天天象棋学习助手"
    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
~~~

- [ ] **Step 2: 确认 UI 测试失败**

Run: .\.venv\Scripts\python -m pytest tests\ui\test_main_window.py -q

Expected: FAIL，MainWindow 尚不存在。

- [ ] **Step 3: 实现最小但完整的窗口结构**

MainWindow 使用 QSplitter：左侧 BoardWidget 占 58%，右侧 QTabWidget 包含“分析”和“教练”；底部状态栏显示连接、行棋方、识别、引擎和 DeepSeek 五个标签。BoardWidget 使用 QWidget 绘制 9×10 交点棋盘并公开：

~~~python
class BoardWidget(QWidget):
    move_selected = Signal(str)

    def set_position(self, board: BoardState) -> None:
        self._board = board
        self.update()

    def set_last_move(self, move: Move | None) -> None:
        self._last_move = move
        self.update()

    def set_candidates(self, moves: tuple[Move, ...]) -> None:
        self._candidates = moves
        self.update()
~~~

bootstrap.run() 创建 QApplication、MainWindow、显示窗口并返回 app.exec()。不在本任务连接真实服务。

- [ ] **Step 4: 运行 UI 与无头领域测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\ui\test_main_window.py tests\unit -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 全部通过。

- [ ] **Step 5: 提交桌面窗口骨架**

~~~powershell
git add src\xiangqi_agent\bootstrap.py src\xiangqi_agent\ui tests\ui
git commit -m "feat: add independent desktop window shell"
~~~

### Task 6: 启动快捷方式并发现微信窗口

**Files:**
- Create: src/xiangqi_agent/platform/__init__.py
- Create: src/xiangqi_agent/platform/shortcut.py
- Create: src/xiangqi_agent/platform/windows.py
- Create: src/xiangqi_agent/ui/dialogs/__init__.py
- Create: src/xiangqi_agent/ui/dialogs/connect_dialog.py
- Create: tests/unit/platform/test_shortcut.py
- Create: tests/unit/platform/test_windows.py
- Create: tests/ui/test_connect_dialog.py

**Interfaces:**
- Produces: launch_shortcut(path: Path) -> None
- Produces: WindowInfo(hwnd: int, title: str, process_name: str, client_size: tuple[int, int])
- Produces: WindowsWindowCatalog.list_candidates() -> tuple[WindowInfo, ...]
- Consumes: 已知快捷方式路径和 Win32 EnumWindows

- [ ] **Step 1: 写筛选与学习模式确认测试**

~~~python
from xiangqi_agent.platform.windows import WindowInfo, filter_candidates


def test_filter_candidates_keeps_visible_wechat_windows() -> None:
    windows = (
        WindowInfo(1, "天天象棋", "Weixin.exe", (900, 1200)),
        WindowInfo(2, "", "Weixin.exe", (0, 0)),
        WindowInfo(3, "记事本", "notepad.exe", (800, 600)),
    )
    assert [w.hwnd for w in filter_candidates(windows)] == [1]
~~~

ConnectDialog 的 UI 测试必须确认未勾选“当前为人机练习、残局或复盘”时连接按钮禁用。

- [ ] **Step 2: 运行并确认失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\platform tests\ui\test_connect_dialog.py -q

Expected: FAIL，platform 模块尚不存在。

- [ ] **Step 3: 实现快捷方式启动和窗口目录**

launch_shortcut 使用 os.startfile(str(path))，仅允许 .lnk 且文件必须存在。WindowsWindowCatalog 通过 ctypes 调用 EnumWindows、IsWindowVisible、GetWindowTextW、GetClientRect 和 GetWindowThreadProcessId；进程名通过 psutil 不引入新依赖，改用 QueryFullProcessImageNameW。

筛选条件为：可见、非零客户区、标题非空，且进程名包含 Weixin、WeChat、xwechat 或窗口标题包含“天天象棋”。ConnectDialog 每 2 秒刷新列表，显示标题、进程和尺寸，不自动选择隐藏窗口。

- [ ] **Step 4: 运行测试和 Windows 手工冒烟**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\platform tests\ui\test_connect_dialog.py -q
.\.venv\Scripts\python -c "from pathlib import Path; from xiangqi_agent.platform.shortcut import validate_shortcut; print(validate_shortcut(Path(r'C:\Users\lenovo\Desktop\天天象棋.lnk')))"
~~~

Expected: 自动测试通过，第二条输出该快捷方式的绝对路径且不启动程序。

- [ ] **Step 5: 提交连接入口**

~~~powershell
git add src\xiangqi_agent\platform src\xiangqi_agent\ui\dialogs tests
git commit -m "feat: add shortcut launch and window selection"
~~~

### Task 7: 抽象帧源并接入 Windows Graphics Capture

**Files:**
- Create: src/xiangqi_agent/capture/__init__.py
- Create: src/xiangqi_agent/capture/protocol.py
- Create: src/xiangqi_agent/capture/fake.py
- Create: src/xiangqi_agent/capture/windows_capture_source.py
- Create: tests/unit/capture/test_fake_source.py
- Create: tests/integration/capture/test_windows_capture.py
- Create: scripts/smoke_capture.py

**Interfaces:**
- Consumes: WindowInfo
- Produces: CaptureFrame(timestamp_ns: int, hwnd: int, bgra: NDArray[uint8])
- Produces: FrameSource.start(on_frame: Callable[[CaptureFrame], None]) -> None
- Produces: FrameSource.stop() -> None

- [ ] **Step 1: 写可控假帧源测试**

~~~python
import numpy as np
from xiangqi_agent.capture.fake import FakeFrameSource


def test_fake_frame_source_emits_owned_bgra_frames() -> None:
    received = []
    source = FakeFrameSource(hwnd=9)
    source.start(received.append)
    source.push(np.zeros((20, 30, 4), dtype=np.uint8), timestamp_ns=123)
    source.stop()
    assert received[0].hwnd == 9
    assert received[0].bgra.shape == (20, 30, 4)
    assert received[0].timestamp_ns == 123
    assert received[0].bgra.flags["OWNDATA"]
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\capture\test_fake_source.py -q

Expected: FAIL，capture 模块不存在。

- [ ] **Step 3: 实现协议、假源和真实捕获适配器**

protocol.py 定义冻结 dataclass CaptureFrame 和 Protocol FrameSource。WindowsCaptureSource 将 windows-capture 的回调帧立刻转换为自有 BGRA NumPy 数组，并通过长度 2 的 Queue 把最新帧交给处理线程；队列满时丢弃旧帧而不是阻塞捕获线程。start 与 stop 必须幂等，窗口关闭时触发 closed 回调。

scripts/smoke_capture.py 接受 --hwnd、--seconds 和 --output-dir，只保存 1 张首帧与 1 张末帧，用于人工验证；不包含任何点击功能。

- [ ] **Step 4: 运行单元测试和真实窗口冒烟**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\capture -q
$targetHwnd = .\.venv\Scripts\python -c "from xiangqi_agent.platform.windows import WindowsWindowCatalog; print(WindowsWindowCatalog().list_candidates()[0].hwnd)"
.\.venv\Scripts\python scripts\smoke_capture.py --hwnd $targetHwnd --seconds 3 --output-dir .local\capture-smoke
~~~

Expected: 天天象棋窗口已打开时，单元测试通过；真实冒烟生成两张仅包含目标窗口的 BGRA 转 PNG 图片。没有候选窗口时命令明确失败且不产生图片。

- [ ] **Step 5: 提交捕获层**

~~~powershell
git add src\xiangqi_agent\capture scripts\smoke_capture.py tests
git commit -m "feat: add window frame capture abstraction"
~~~

### Task 8: 棋盘几何、归一化标定和 90 点裁切

**Files:**
- Create: src/xiangqi_agent/vision/__init__.py
- Create: src/xiangqi_agent/vision/types.py
- Create: src/xiangqi_agent/vision/geometry.py
- Create: src/xiangqi_agent/vision/locator.py
- Create: src/xiangqi_agent/ui/dialogs/calibration_dialog.py
- Create: tests/unit/vision/test_geometry.py
- Create: tests/unit/vision/test_locator.py
- Create: tests/ui/test_calibration_dialog.py

**Interfaces:**
- Consumes: CaptureFrame
- Produces: NormalizedQuad(points: tuple[tuple[float, float], ...])
- Produces: BoardGeometry.from_quad(quad, frame_size) -> BoardGeometry
- Produces: BoardGeometry.grid_points() -> tuple[tuple[float, float], ...]，长度 90
- Produces: BoardGeometry.crop_intersections(frame, size=48) -> tuple[NDArray, ...]
- Produces: BoardLocator.locate(frame: CaptureFrame) -> NormalizedQuad | None

- [ ] **Step 1: 写缩放不变性和网格顺序测试**

~~~python
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad


def test_normalized_calibration_survives_window_resize() -> None:
    quad = NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
    small = BoardGeometry.from_quad(quad, (1000, 800))
    large = BoardGeometry.from_quad(quad, (2000, 1600))
    assert len(small.grid_points()) == 90
    assert large.grid_points()[89] == tuple(v * 2 for v in small.grid_points()[89])


def test_locator_recovers_synthetic_warped_grid(synthetic_board_frame) -> None:
    from xiangqi_agent.vision.locator import BoardLocator
    quad = BoardLocator().locate(synthetic_board_frame)
    assert quad is not None
    assert synthetic_board_frame.corner_error(quad) <= 4.0
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\vision\test_geometry.py tests\unit\vision\test_locator.py tests\ui\test_calibration_dialog.py -q

Expected: FAIL，vision.geometry 尚不存在。

- [ ] **Step 3: 实现透视几何和四角标定**

NormalizedQuad 按左上、右上、右下、左下固定顺序校验坐标在 0..1 且凸四边形面积大于窗口面积的 10%。BoardGeometry 使用 cv2.getPerspectiveTransform 把棋盘映射到固定 480×540 画布，网格交点顺序为内部标准棋盘从黑方顶部到红方底部逐行排列。每个交点裁 48×48 补丁，越界直接抛 GeometryError。

BoardLocator 对灰度图执行自适应阈值和 HoughLinesP，分别聚类近水平与近垂直线，寻找 10 条秩线和 9 条路线的最大一致子集；相邻间隔变异系数必须 <=0.08，四角重投影误差必须 <=4 像素。几何门不通过时返回 None，由 UI 转四角标定，不能返回低质量猜测。

CalibrationDialog 连续收集四次左键点击，实时画出四边形和 90 点预览；只有几何校验通过时“确认”按钮可用。

- [ ] **Step 4: 运行几何和 UI 测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\unit\vision\test_geometry.py tests\unit\vision\test_locator.py tests\ui\test_calibration_dialog.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 全部通过，90 点顺序和缩放行为固定。

- [ ] **Step 5: 提交标定层**

~~~powershell
git add src\xiangqi_agent\vision src\xiangqi_agent\ui\dialogs\calibration_dialog.py tests
git commit -m "feat: add normalized board calibration"
~~~

### Task 9: 建立识别资产评估门和许可证记录

**Files:**
- Create: scripts/evaluate_models.py
- Create: scripts/collect_intersections.py
- Create: scripts/train_classifier.py
- Create: tests/unit/vision/test_model_selection.py
- Create: tests/fixtures/frames/manifest.schema.json
- Create: THIRD_PARTY_NOTICES.md
- Create: assets.lock.json

**Interfaces:**
- Produces: ModelMetrics(board_accuracy: float, macro_f1: float, p95_ms: float)
- Produces: select_candidate(metrics: dict[str, ModelMetrics]) -> str | None
- Produces: assets.lock.json 中每个资产的 url、commit、sha256、license、local_path
- Consumes: 两个已审计候选模型和本机标注帧

- [ ] **Step 1: 写固定门槛测试**

~~~python
from scripts.evaluate_models import ModelMetrics, select_candidate


def test_selects_only_candidate_meeting_all_gates() -> None:
    metrics = {
        "wechat_15_class": ModelMetrics(0.991, 0.996, 72.0),
        "two_stage": ModelMetrics(0.995, 0.994, 61.0),
    }
    assert select_candidate(metrics) == "wechat_15_class"


def test_returns_none_when_retraining_is_required() -> None:
    metrics = {"weak": ModelMetrics(0.97, 0.98, 40.0)}
    assert select_candidate(metrics) is None
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\vision\test_model_selection.py -q

Expected: FAIL，evaluate_models 尚不存在。

- [ ] **Step 3: 实现可复现评估和资产锁**

select_candidate 只接受 board_accuracy>=0.99、macro_f1>=0.995、p95_ms<=100 的候选，并按 board_accuracy、macro_f1、速度依次排序。evaluate_models.py 读取 manifest 中的截图路径、真实 FEN、方向、DPI 和主题，输出 JSON 与混淆矩阵 CSV。

assets.lock.json 由脚本写入实际 SHA-256，不允许空哈希。THIRD_PARTY_NOTICES.md 分别记录候选仓库 URL、固定提交、MIT 许可证文本位置、模型来源，以及 Pikafish GPLv3 的独立进程使用方式。

- [ ] **Step 4: 采集并执行模型选择**

执行者使用 collect_intersections.py 从本机微信天天象棋采集不少于 200 张稳定帧，覆盖红黑在下、100%/125%/150% DPI、开中残局和选中高亮，并人工校验 manifest 的 FEN。

Run:

~~~powershell
.\.venv\Scripts\python scripts\evaluate_models.py --manifest tests\fixtures\frames\manifest.json --output .local\model-eval
~~~

Expected: 输出 report.json。若 select_candidate 返回模型名，将该模型写入 assets.lock.json；若返回 null，则运行 train_classifier.py 训练至少 2,000 个已标注交点补丁，导出 ONNX 后重新运行同一评估，直到满足三项门槛。不得降低门槛来通过任务。

- [ ] **Step 5: 运行测试并提交可审计资产元数据**

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\vision\test_model_selection.py -q
git add scripts tests\fixtures\frames\manifest.schema.json THIRD_PARTY_NOTICES.md assets.lock.json
git commit -m "build: add licensed recognition asset gate"
~~~

### Task 10: ONNX 棋子分类和完整局面识别

**Files:**
- Create: src/xiangqi_agent/vision/model.py
- Create: src/xiangqi_agent/vision/recognizer.py
- Create: tests/unit/vision/test_model.py
- Create: tests/unit/vision/test_recognizer.py
- Create: tests/replay/test_recognition_manifest.py

**Interfaces:**
- Consumes: BoardGeometry、90 个交点补丁、assets.lock.json
- Produces: PiecePrediction(piece: str, confidence: float)
- Produces: ObservedPosition(pieces, confidences, orientation, timestamp_ns, geometry)
- Produces: PositionRecognizer.recognize(frame: CaptureFrame, geometry: BoardGeometry) -> ObservedPosition
- Produces: ObservedPosition.to_board_state(side_to_move: Side) -> BoardState

- [ ] **Step 1: 写假 ONNX 会话识别测试**

~~~python
import numpy as np
from xiangqi_agent.vision.model import PieceClassifier


class FakeSession:
    def run(self, output_names, inputs):
        batch = next(iter(inputs.values()))
        logits = np.zeros((batch.shape[0], 15), dtype=np.float32)
        logits[:, 14] = 8.0
        return [logits]


def test_classifier_returns_empty_with_confidence() -> None:
    classifier = PieceClassifier(FakeSession())
    patches = tuple(np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(90))
    result = classifier.predict(patches)
    assert len(result) == 90
    assert all(item.piece == "." and item.confidence > 0.99 for item in result)
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\vision\test_model.py tests\unit\vision\test_recognizer.py -q

Expected: FAIL，PieceClassifier 尚不存在。

- [ ] **Step 3: 实现模型适配和方向归一化**

PieceClassifier 从 assets.lock.json 加载被选模型，核验文件 SHA-256 后创建 ONNX Runtime CPU session；输入统一为 48×48 RGB float32，按被选资产记录的均值和方差归一化。softmax 在 NumPy 中稳定计算。

PositionRecognizer 根据帅/将所在九宫推断方向；黑方在下时将观察数组旋转 180 度，使内部 BoardState 始终红方在下。必须校验双方各恰好一个将帅、士相位置范围和棋子数量上限；违反时返回 RecognizerError，不自行改写高置信棋子。

- [ ] **Step 4: 运行单元和离线回放测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\vision tests\replay\test_recognition_manifest.py -q
~~~

Expected: 单元测试全部通过；带 assets 标记的回放达到 Task 9 固定门槛。

- [ ] **Step 5: 提交识别管线**

~~~powershell
git add src\xiangqi_agent\vision tests
git commit -m "feat: add ONNX board position recognition"
~~~

### Task 11: 多帧稳定器和可靠同步状态机

**Files:**
- Create: src/xiangqi_agent/vision/temporal.py
- Create: src/xiangqi_agent/sync/__init__.py
- Create: src/xiangqi_agent/sync/state_machine.py
- Create: src/xiangqi_agent/sync/service.py
- Create: tests/unit/vision/test_temporal.py
- Create: tests/unit/sync/test_state_machine.py
- Create: tests/integration/sync/test_sync_service.py

**Interfaces:**
- Consumes: ObservedPosition、BoardState、detect_unique_move
- Produces: TemporalFilter.push(observation) -> ObservedPosition | None
- Produces: SyncState(StrEnum)
- Produces: MoveEvent(before_position_id, after_position_id, move, chinese, side, is_capture, is_check, confidence, timestamp_ns)
- Produces: SyncUpdate(state, board, move_event, confidence, reason)
- Produces: SyncService.on_observation(observation) -> SyncUpdate

- [ ] **Step 1: 写动画不更新、合法一步更新和错误局面锁闭测试**

~~~python
from xiangqi_agent.sync.state_machine import SyncState


def test_requires_three_identical_observations(sync_harness) -> None:
    for timestamp in (0, 200_000_000):
        update = sync_harness.observe_start_position(timestamp)
        assert update.state is SyncState.CONFIRMING_INITIAL_POSITION
    update = sync_harness.observe_start_position(400_000_000)
    assert update.state is SyncState.READY_FOR_CONFIRMATION


def test_animation_frame_never_replaces_confirmed_board(sync_harness) -> None:
    sync_harness.confirm_start()
    sync_harness.observe_impossible_partial_frame()
    assert sync_harness.confirmed_board == sync_harness.start_board
    assert sync_harness.last_update.state is SyncState.SYNC_WARNING


def test_three_stable_frames_confirm_one_legal_move(sync_harness) -> None:
    sync_harness.confirm_start()
    update = sync_harness.observe_move_three_times("h2e2")
    assert update.state is SyncState.SYNCED
    assert update.move_event is not None
    assert update.move_event.move.uci == "h2e2"
~~~

- [ ] **Step 2: 运行测试并确认失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\vision\test_temporal.py tests\unit\sync tests\integration\sync -q

Expected: FAIL，TemporalFilter 和 SyncService 尚不存在。

- [ ] **Step 3: 实现状态机和失败闭锁**

SyncState 固定包含 DISCONNECTED、CALIBRATING、CONFIRMING_INITIAL_POSITION、READY_FOR_CONFIRMATION、WATCHING、WAITING_FOR_STABLE_FRAMES、VALIDATING_MOVE、SYNCED、ANALYZING、SYNC_WARNING、PAUSED。

TemporalFilter 以标准化 90 点棋子元组为键，保存最近观察；只有最后 3 个键一致、首尾时间差 <=600 ms、将帅置信度均 >=0.85 时返回稳定观察。SyncService 的核心逻辑必须等价于：

~~~python
def on_stable_observation(self, observed: ObservedPosition) -> SyncUpdate:
    candidate = observed.to_board_state(side_to_move=self.expected_side)
    if self.confirmed is None:
        self.pending_initial = candidate
        return SyncUpdate.ready_for_confirmation(candidate, observed.minimum_confidence)
    try:
        move = detect_unique_move(self.confirmed, candidate)
    except ValueError as exc:
        self.failed_rescans += 1
        return SyncUpdate.warning(self.confirmed, str(exc))
    before = self.confirmed
    self.confirmed = candidate
    self.failed_rescans = 0
    event = MoveEvent.from_confirmed_positions(
        before=before,
        after=candidate,
        move=move,
        confidence=observed.minimum_confidence,
        timestamp_ns=observed.timestamp_ns,
    )
    return SyncUpdate.synced(candidate, event, observed.minimum_confidence)
~~~

初始局面只有经用户 confirm_initial(board) 才进入 WATCHING。失败 3 次后发出 requires_manual_confirmation=True，但继续保留旧局面。

- [ ] **Step 4: 运行同步测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\vision\test_temporal.py tests\unit\sync tests\integration\sync -q
.\.venv\Scripts\mypy src
~~~

Expected: 动画帧、非法变化和多步跳变全部不会修改 confirmed_board；合法一步只产生一次 MoveEvent。

- [ ] **Step 5: 提交同步内核**

~~~powershell
git add src\xiangqi_agent\vision\temporal.py src\xiangqi_agent\sync tests
git commit -m "feat: add fail-closed board synchronization"
~~~

### Task 12: 连接流程、初始确认、镜像棋盘与手工纠正

**Files:**
- Create: src/xiangqi_agent/application/__init__.py
- Create: src/xiangqi_agent/application/controller.py
- Modify: src/xiangqi_agent/bootstrap.py
- Modify: src/xiangqi_agent/ui/main_window.py
- Modify: src/xiangqi_agent/ui/board_widget.py
- Modify: src/xiangqi_agent/ui/dialogs/connect_dialog.py
- Create: src/xiangqi_agent/ui/dialogs/correction_dialog.py
- Create: tests/ui/test_sync_workflow.py
- Create: tests/integration/test_application_controller.py

**Interfaces:**
- Consumes: FrameSource、PositionRecognizer、SyncService、MainWindow
- Produces: ApplicationController.connect(window: WindowInfo) -> None
- Produces: ApplicationController.confirm_initial_position(side_to_move: Side) -> None
- Produces: ApplicationController.apply_manual_correction(board: BoardState) -> None

- [ ] **Step 1: 写端到端假源同步 UI 测试**

~~~python
def test_fake_move_updates_mirror_and_status(qtbot, app_harness) -> None:
    app_harness.connect_fake_window()
    app_harness.push_start_position_three_times()
    app_harness.confirm_initial_position(side_to_move="w")
    app_harness.push_move_three_times("h2e2")
    qtbot.waitUntil(lambda: app_harness.window.board_widget.last_move_uci == "h2e2")
    assert app_harness.window.sync_label.text().startswith("已同步")
    assert app_harness.window.board_widget.board.side_to_move == "b"
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\ui\test_sync_workflow.py tests\integration\test_application_controller.py -q

Expected: FAIL，ApplicationController 尚不存在。

- [ ] **Step 3: 实现组合根和 UI 工作流**

ApplicationController 只负责订阅事件和跨层编排，不写识别、规则或绘制算法。连接后依次启动 FrameSource、运行标定/识别、弹出初始确认；确认框必须让用户选择“红方行棋”或“黑方行棋”，不能从静态布局猜测。确认后每个 SyncUpdate.SYNCED 调用 BoardWidget.set_position、set_last_move，并发出 position_confirmed(position_id)；SyncService 在每个合法半回合后自动翻转行棋方。

CorrectionDialog 显示可点击的 9×10 棋盘和 15 类棋子选择器，保存前使用 BoardState、将帅数量和 rules.is_in_check 进行基本校验。手工修正通过 SyncService.replace_confirmed(board, reason="manual") 成为新的 confirmed_board，原因写入同步事件元数据，但不伪造 MoveEvent。

- [ ] **Step 4: 运行 UI 和集成测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\ui tests\integration\test_application_controller.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 连接、初始确认、一步同步、警告和手工纠正五条流程全部通过。

- [ ] **Step 5: 提交镜像同步 UI**

~~~powershell
git add src\xiangqi_agent\application src\xiangqi_agent\bootstrap.py src\xiangqi_agent\ui tests
git commit -m "feat: connect synchronized mirror board workflow"
~~~

### Task 13: Pikafish 下载校验、UCI 协议和受管进程

**Files:**
- Create: src/xiangqi_agent/domain/analysis.py
- Create: src/xiangqi_agent/engine/__init__.py
- Create: src/xiangqi_agent/engine/protocol.py
- Create: src/xiangqi_agent/engine/process.py
- Create: scripts/download_pikafish.py
- Create: tests/unit/engine/test_protocol.py
- Create: tests/integration/engine/test_process.py
- Create: tests/fixtures/engine/sample_uci_output.txt

**Interfaces:**
- Produces: EngineLine、EngineAnalysis
- Produces: parse_info_line(text: str, position_id: str) -> EngineLine | None
- Produces: PikafishProcess.start()、analyse(board, movetime_ms, multipv) 、stop()、close()
- Consumes: BoardState FEN、Pikafish 可执行文件和 NNUE

- [ ] **Step 1: 写 UCI MultiPV 和将杀解析测试**

~~~python
from xiangqi_agent.engine.protocol import parse_info_line


def test_parse_multipv_cp_line() -> None:
    line = parse_info_line(
        "info depth 18 multipv 2 score cp -43 nodes 900 nps 1000 pv h2e2 h9g7",
        position_id="abc",
    )
    assert line is not None
    assert line.position_id == "abc"
    assert line.depth == 18
    assert line.multipv == 2
    assert line.score_cp == -43
    assert line.pv == ("h2e2", "h9g7")


def test_parse_mate_separately_from_cp() -> None:
    line = parse_info_line("info depth 20 score mate 3 pv e4e9", "abc")
    assert line is not None and line.mate_in == 3 and line.score_cp is None
~~~

- [ ] **Step 2: 确认解析测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\engine\test_protocol.py -q

Expected: FAIL，engine.protocol 尚不存在。

- [ ] **Step 3: 实现严格 UCI 解析和进程生命周期**

PikafishProcess.start() 启动子进程后发送 uci，等待 uciok；设置 Threads、Hash、MultiPV 和 EvalFile，再发送 isready 并等待 readyok。analyse() 发送 position fen 与 go movetime，持续收集每个 multipv 的最深结果，收到 bestmove 后返回 EngineAnalysis。close() 依次发送 stop、quit，2 秒未退出再 terminate，仍未退出才 kill。

download_pikafish.py 只从 official-pikafish/Pikafish 的固定 2026-01-02 Release 下载，解压后选择当前 CPU 可运行的 Windows 二进制，运行 uci 自检，将实际 URL 和 SHA-256 写入 assets.lock.json；任何哈希与已有锁不一致时中止。

- [ ] **Step 4: 运行解析测试和真实引擎集成测试**

Run:

~~~powershell
.\.venv\Scripts\python scripts\download_pikafish.py --version 2026-01-02 --destination .local\pikafish
.\.venv\Scripts\python -m pytest tests\unit\engine -q
$env:PIKAFISH_PATH=(Resolve-Path '.local\pikafish\pikafish.exe').Path
.\.venv\Scripts\python -m pytest tests\integration\engine\test_process.py -q -m assets
Remove-Item Env:PIKAFISH_PATH
~~~

Expected: 协议测试通过；真实引擎能分析初始局面并在 close 后不留下进程。

- [ ] **Step 5: 提交引擎适配器**

~~~powershell
git add src\xiangqi_agent\domain\analysis.py src\xiangqi_agent\engine scripts\download_pikafish.py tests assets.lock.json THIRD_PARTY_NOTICES.md
git commit -m "feat: add managed Pikafish UCI analysis"
~~~

### Task 14: 两阶段分析、取消过期任务和分析面板

**Files:**
- Create: src/xiangqi_agent/engine/service.py
- Modify: src/xiangqi_agent/ui/analysis_panel.py
- Modify: src/xiangqi_agent/application/controller.py
- Create: tests/unit/engine/test_analysis_service.py
- Create: tests/ui/test_analysis_panel.py
- Create: tests/integration/test_stale_analysis.py

**Interfaces:**
- Consumes: BoardState、PikafishProcess
- Produces: AnalysisService.submit(board: BoardState) -> None
- Produces: quick_ready(EngineAnalysis)、deep_ready(EngineAnalysis)、failed(str) Qt signals
- Produces: AnalysisPanel.set_analysis(analysis: EngineAnalysis, phase: str) -> None

- [ ] **Step 1: 写快速/加深顺序和过期结果测试**

~~~python
def test_service_emits_quick_then_deep(fake_engine, qtbot) -> None:
    service = fake_engine.analysis_service()
    seen = []
    service.quick_ready.connect(lambda item: seen.append(("quick", item.position_id)))
    service.deep_ready.connect(lambda item: seen.append(("deep", item.position_id)))
    service.submit(fake_engine.start_board)
    qtbot.waitUntil(lambda: len(seen) == 2)
    assert seen == [("quick", fake_engine.start_board.position_id),
                    ("deep", fake_engine.start_board.position_id)]


def test_controller_discards_old_position_result(app_harness) -> None:
    old = app_harness.start_analysis()
    app_harness.sync_next_position()
    app_harness.deliver_analysis(old)
    assert app_harness.window.analysis_panel.position_id != old.position_id
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\engine\test_analysis_service.py tests\ui\test_analysis_panel.py tests\integration\test_stale_analysis.py -q

Expected: FAIL，AnalysisService 尚不存在。

- [ ] **Step 3: 实现串行工作线程和渐进 UI**

AnalysisService 使用单独 QThread。submit 新局面时设置 generation、发送 stop、清空排队任务，然后执行 500 ms MultiPV=3 快速分析和 3000 ms 加深分析；每次发信号前核对 generation。

AnalysisPanel 展示红方视角评估条、当前行棋方、前三候选、每个候选的中文记谱/评分/深度和可展开 PV。mate 值显示“红方 N 步成杀”或“黑方 N 步成杀”，不换算百分比。快速结果标注“快速估计”，加深结果标注实际深度和用时。

- [ ] **Step 4: 运行分析测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\unit\engine tests\ui\test_analysis_panel.py tests\integration\test_stale_analysis.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 快速结果先到、加深结果覆盖、旧 position_id 永不显示。

- [ ] **Step 5: 提交渐进分析**

~~~powershell
git add src\xiangqi_agent\engine\service.py src\xiangqi_agent\ui\analysis_panel.py src\xiangqi_agent\application tests
git commit -m "feat: add progressive cancellable analysis"
~~~

### Task 15: 构建可验证的教练证据包

**Files:**
- Create: src/xiangqi_agent/domain/coach.py
- Create: src/xiangqi_agent/coach/__init__.py
- Create: src/xiangqi_agent/coach/evidence.py
- Create: tests/unit/coach/test_evidence.py

**Interfaces:**
- Consumes: BoardState、最近 MoveEvent、EngineAnalysis、用户执方
- Produces: CoachEvidence
- Produces: build_evidence(board, moves, analysis, user_side) -> CoachEvidence
- Produces: allowed_move_map: dict[str, str]，键为 candidate_1..3，值为中文走法

- [ ] **Step 1: 写证据完整性和隐私测试**

~~~python
from xiangqi_agent.coach.evidence import build_evidence


def test_evidence_contains_only_verified_candidates(engine_fixture) -> None:
    evidence = build_evidence(
        engine_fixture.board,
        engine_fixture.moves,
        engine_fixture.analysis,
        user_side="w",
    )
    assert set(evidence.allowed_move_map) == {"candidate_1", "candidate_2", "candidate_3"}
    assert evidence.fen == engine_fixture.board_fen
    assert "screenshot" not in evidence.model_dump()
    assert all(line.uci in engine_fixture.legal_uci for line in evidence.candidates)
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\coach\test_evidence.py -q

Expected: FAIL，CoachEvidence 尚不存在。

- [ ] **Step 3: 实现事实提取**

CoachEvidence 使用 Pydantic frozen model，字段固定为 position_id、fen、user_side、phase、recent_moves、material_facts、king_safety_facts、immediate_tactics、candidates、allowed_move_map、actual_move_review。phase 按总子力和 ply 确定为 opening、middlegame 或 endgame。

事实提取必须由程序计算：各类子数量、当前是否被将、候选首步是否吃子/将军、PV 中前 6 个半回合、用户实际走法相对最佳的 cp 损失。不得把未验证的自然语言判断写入 evidence。

- [ ] **Step 4: 运行教练证据测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\coach\test_evidence.py -q
.\.venv\Scripts\mypy src
~~~

Expected: 证据 JSON 可序列化，且不包含 NumPy 数组、图片字节、API Key 或未经验证的走法。

- [ ] **Step 5: 提交证据层**

~~~powershell
git add src\xiangqi_agent\domain\coach.py src\xiangqi_agent\coach tests\unit\coach
git commit -m "feat: build grounded coaching evidence"
~~~

### Task 16: DeepSeek V4 客户端、JSON 校验和本地降级

**Files:**
- Create: src/xiangqi_agent/coach/prompts.py
- Create: src/xiangqi_agent/coach/client.py
- Create: src/xiangqi_agent/coach/service.py
- Create: tests/unit/coach/test_validation.py
- Create: tests/integration/coach/test_deepseek_client.py
- Create: tests/fixtures/deepseek/valid.json
- Create: tests/fixtures/deepseek/invalid_candidate.json

**Interfaces:**
- Consumes: CoachEvidence、用户问题、SecretStore
- Produces: CoachExplanation
- Produces: DeepSeekClient.explain(evidence, question, deep=False) -> CoachExplanation
- Produces: CoachService.request(...) Qt signals explanation_ready、failed

- [ ] **Step 1: 写请求格式、越界候选和降级测试**

~~~python
def test_request_uses_current_flash_model(fake_http, evidence) -> None:
    client = fake_http.client(response_fixture="valid.json")
    result = client.explain(evidence, "为什么推荐这一步？", deep=False)
    request = fake_http.last_json
    assert request["model"] == "deepseek-v4-flash"
    assert request["thinking"] == {"type": "disabled"}
    assert request["response_format"] == {"type": "json_object"}
    assert result.candidate_id == "candidate_1"


def test_unlisted_candidate_falls_back_after_one_retry(fake_http, evidence) -> None:
    fake_http.queue("invalid_candidate.json", "invalid_candidate.json")
    result = fake_http.client().explain(evidence, "下一步呢？")
    assert result.source == "local_fallback"
    assert result.candidate_id in evidence.allowed_move_map
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\coach\test_validation.py tests\integration\coach\test_deepseek_client.py -q

Expected: FAIL，DeepSeekClient 尚不存在。

- [ ] **Step 3: 实现严格结构化调用**

系统提示词固定包含以下约束：

~~~text
你是中国象棋学习教练。所有具体走法只能引用 evidence.allowed_move_map 的 candidate_id。
不得发明走法、评分、吃子、将军或变化线。证据不足时明确说明不足。
只输出一个 JSON 对象，字段必须是 position_summary、main_plan、candidate_id、
why、opponent_threat、alternatives、training_question、confidence。
~~~

HTTP 请求只发送 CoachEvidence.model_dump(mode="json") 和当前问题。实时模式使用 deepseek-v4-flash、thinking disabled；深度模式使用 deepseek-v4-pro、thinking enabled、reasoning_effort high。超时 12 秒。

CoachExplanation 校验 candidate_id 和 alternatives 中的全部 ID；再扫描回答中的 UCI 走法和中文记谱，任何不在 allowed_move_map 中的具体走法使校验失败。首次失败追加校验错误重试一次；再次失败返回由 evidence 确定性生成的本地模板解释。

- [ ] **Step 4: 运行客户端和脱敏测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests\unit\coach tests\integration\coach -q
.\.venv\Scripts\python -m pytest tests\unit\diagnostics\test_logging.py -q
~~~

Expected: 有效 JSON 通过，越界走法被拒绝，超时/余额不足/空内容都回退，日志不含测试密钥。

- [ ] **Step 5: 提交 DeepSeek 教练服务**

~~~powershell
git add src\xiangqi_agent\coach tests\unit\coach tests\integration\coach tests\fixtures\deepseek
git commit -m "feat: add grounded DeepSeek coaching"
~~~

### Task 17: 分级提示、追问、走法比较和教练面板

**Files:**
- Modify: src/xiangqi_agent/ui/coach_panel.py
- Modify: src/xiangqi_agent/ui/analysis_panel.py
- Create: src/xiangqi_agent/ui/dialogs/settings_dialog.py
- Modify: src/xiangqi_agent/application/controller.py
- Modify: src/xiangqi_agent/domain/notation.py
- Create: tests/unit/domain/test_move_reference.py
- Create: tests/ui/test_coach_panel.py
- Create: tests/integration/test_coach_workflow.py

**Interfaces:**
- Consumes: CoachExplanation、EngineAnalysis、BoardState
- Produces: resolve_move_reference(board, text: str) -> Move | None
- Produces: CoachPanel.question_submitted(str) Qt signal
- Produces: CoachPanel.hint_level_requested(int) Qt signal，级别 1..4

- [ ] **Step 1: 写分级揭示和中文走法引用测试**

~~~python
def test_resolve_exact_chinese_move_from_legal_moves(start_board) -> None:
    from xiangqi_agent.domain.notation import resolve_move_reference
    move = resolve_move_reference(start_board, "我想走炮二平五，可以吗？")
    assert move is not None and move.uci == "h2e2"


def test_hints_reveal_in_order(qtbot, coach_panel, explanation) -> None:
    coach_panel.set_explanation(explanation)
    assert coach_panel.visible_sections() == ("position_summary",)
    coach_panel.reveal_level(2)
    assert "main_plan" in coach_panel.visible_sections()
    coach_panel.reveal_level(3)
    assert "candidate" in coach_panel.visible_sections()
    coach_panel.reveal_level(4)
    assert "variation" in coach_panel.visible_sections()
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\domain\test_move_reference.py tests\ui\test_coach_panel.py tests\integration\test_coach_workflow.py -q

Expected: FAIL，走法引用和提示级别尚未实现。

- [ ] **Step 3: 实现学习式交互**

resolve_move_reference 枚举当前全部 legal_moves，把 to_chinese 的结果与用户文本做标准化精确包含匹配；恰好匹配一个才返回，零个或多个都返回 None。用户提到可解析走法时，先用 Pikafish 分析该分支并写入 evidence.actual_move_review，再调用 DeepSeek；无法解析时允许回答战略问题，但 UI 明确提示“未识别到具体走法，未进行走法比较”。

CoachPanel 默认只显示一级形势主题；四级分别揭示形势、计划、候选、PV。每次新 position_id 清空旧问题上下文，但保留会话历史供复盘。发送按钮在无 API Key 时仍可用，走本地模板解释。

SettingsDialog 提供“逐层提示/直接显示前三候选”、DeepSeek API Key 保存/清除、默认实时模型只读展示、深度复盘开关、诊断裁图开关和 Pikafish 路径。API Key 输入框永不回显现有密钥，只显示“已配置/未配置”；保存后立即清空输入控件。

- [ ] **Step 4: 运行学习流程测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\unit\domain\test_move_reference.py tests\ui\test_coach_panel.py tests\integration\test_coach_workflow.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 提示按级别揭示；可解析走法先经引擎比较；新局面不显示旧答案。

- [ ] **Step 5: 提交教学交互**

~~~powershell
git add src\xiangqi_agent\ui src\xiangqi_agent\application src\xiangqi_agent\domain\notation.py tests
git commit -m "feat: add progressive coaching interactions"
~~~

### Task 18: SQLite 会话、棋谱、分析和复盘

**Files:**
- Create: src/xiangqi_agent/storage/__init__.py
- Create: src/xiangqi_agent/storage/db.py
- Create: src/xiangqi_agent/storage/repository.py
- Create: src/xiangqi_agent/ui/review_panel.py
- Modify: src/xiangqi_agent/ui/main_window.py
- Modify: src/xiangqi_agent/application/controller.py
- Create: tests/unit/storage/test_migrations.py
- Create: tests/integration/storage/test_repository.py
- Create: tests/ui/test_review_panel.py

**Interfaces:**
- Consumes: BoardState、MoveEvent、EngineAnalysis、CoachExplanation
- Produces: Database.open(path: Path) -> Database
- Produces: SessionRepository.start_session(...) -> int
- Produces: record_position、record_move、record_analysis、record_coach_message
- Produces: load_session(session_id: int) -> SessionReview

- [ ] **Step 1: 写迁移、原子事务和回放测试**

~~~python
def test_repository_round_trips_session(tmp_path, completed_session) -> None:
    repo = completed_session.repository(tmp_path / "agent.db")
    session_id = repo.save(completed_session)
    loaded = repo.load_session(session_id)
    assert [m.uci for m in loaded.moves] == [m.uci for m in completed_session.moves]
    assert loaded.positions[-1].position_id == completed_session.positions[-1].position_id


def test_failed_move_transaction_writes_nothing(repository, invalid_move) -> None:
    before = repository.count_rows("moves")
    with pytest.raises(ValueError):
        repository.record_move(invalid_move)
    assert repository.count_rows("moves") == before
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\storage tests\integration\storage tests\ui\test_review_panel.py -q

Expected: FAIL，storage 模块不存在。

- [ ] **Step 3: 实现版本化 SQLite 和复盘视图**

db.py 使用 PRAGMA foreign_keys=ON、journal_mode=WAL、busy_timeout=3000，并以 PRAGMA user_version 执行版本 1 建表。表字段与设计说明第 13 节一致；分析和证据使用排序键固定的 JSON 文本。

每次已确认同步在单个事务内写 positions 和 moves；分析和教练回答可后写，但必须以 position_id 外键关联。ReviewPanel 提供棋步列表、镜像棋盘、评分曲线、关键失误筛选和该局问答记录，不重新调用云端。

- [ ] **Step 4: 运行存储和 UI 测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\unit\storage tests\integration\storage tests\ui\test_review_panel.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 数据往返一致、失败事务不留半条记录、复盘点击任一步同步棋盘和分析。

- [ ] **Step 5: 提交本地学习记录**

~~~powershell
git add src\xiangqi_agent\storage src\xiangqi_agent\ui\review_panel.py src\xiangqi_agent\ui\main_window.py src\xiangqi_agent\application tests
git commit -m "feat: persist games and local reviews"
~~~

### Task 19: 故障恢复、诊断裁图和隐私保留策略

**Files:**
- Create: src/xiangqi_agent/diagnostics/snapshots.py
- Modify: src/xiangqi_agent/sync/service.py
- Modify: src/xiangqi_agent/application/controller.py
- Modify: src/xiangqi_agent/ui/dialogs/correction_dialog.py
- Create: tests/unit/diagnostics/test_snapshots.py
- Create: tests/integration/test_recovery.py
- Create: tests/ui/test_sync_warning.py

**Interfaces:**
- Consumes: 棋盘裁图、SyncUpdate、AppSettings
- Produces: SnapshotStore.save_incident(crop, incident_id) -> Path | None
- Produces: SnapshotStore.purge_expired(now: datetime) -> int
- Produces: ApplicationController.resume_after_capture_loss() -> None

- [ ] **Step 1: 写默认不保存、只存裁图和七日清理测试**

~~~python
from datetime import UTC, datetime, timedelta


def test_snapshot_store_is_disabled_by_default(snapshot_store, crop) -> None:
    assert snapshot_store.save_incident(crop, "incident-1") is None


def test_purge_removes_only_files_older_than_seven_days(enabled_snapshot_store) -> None:
    old = enabled_snapshot_store.create_fixture(age=timedelta(days=8))
    fresh = enabled_snapshot_store.create_fixture(age=timedelta(days=6))
    assert enabled_snapshot_store.purge_expired(datetime.now(UTC)) == 1
    assert not old.exists() and fresh.exists()
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\unit\diagnostics\test_snapshots.py tests\integration\test_recovery.py tests\ui\test_sync_warning.py -q

Expected: FAIL，SnapshotStore 尚不存在。

- [ ] **Step 3: 实现恢复矩阵**

实现以下确定行为：

- 空帧、窗口关闭或最小化：进入 PAUSED，停止引擎加深任务，保留 confirmed_board。
- 恢复捕获：完整识别；若与 confirmed_board 相同则继续，若恰好一个合法走法则同步，否则要求确认。
- 连续 3 次非法变化：保存一次裁图（仅用户开启诊断时），显示 correction_dialog。
- 引擎第一次崩溃：自动重启并重试当前局面一次；第二次显示禁用状态。
- DeepSeek 失败：不改变同步/引擎状态，只在 coach_panel 显示本地解释来源。
- 数据库失败：切换只读会话，提供导出当前 FEN 和 UCI 棋步列表的按钮。

SnapshotStore 只接受 BoardGeometry 透视后的棋盘图，拒绝宽高比不在 0.75..1.0 的整屏图片；文件名使用 incident UUID，不含窗口标题或用户信息。

- [ ] **Step 4: 运行恢复和隐私测试**

Run:

~~~powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python -m pytest tests\unit\diagnostics tests\integration\test_recovery.py tests\ui\test_sync_warning.py -q
Remove-Item Env:QT_QPA_PLATFORM
~~~

Expected: 所有故障都保持最后确认局面；默认运行不产生 PNG；过期裁图准确清理。

- [ ] **Step 5: 提交恢复机制**

~~~powershell
git add src\xiangqi_agent\diagnostics src\xiangqi_agent\sync src\xiangqi_agent\application src\xiangqi_agent\ui tests
git commit -m "feat: add privacy-safe recovery diagnostics"
~~~

### Task 20: Windows 打包、许可证和安装自检

**Files:**
- Create: packaging/xiangqi_agent.spec
- Create: packaging/requirements-lock.txt
- Create: LICENSE
- Create: README.md
- Modify: THIRD_PARTY_NOTICES.md
- Create: tests/integration/test_asset_integrity.py
- Create: tests/integration/test_packaged_layout.py

**Interfaces:**
- Consumes: pyproject.toml、assets.lock.json、模型、Pikafish 路径
- Produces: dist/天天象棋学习助手/天天象棋学习助手.exe
- Produces: 应用内“环境自检”报告

- [ ] **Step 1: 写资产完整性和发布目录测试**

~~~python
def test_every_locked_asset_has_matching_hash(asset_registry) -> None:
    for asset in asset_registry:
        assert asset.local_path.exists()
        assert asset.sha256 == asset_registry.hash_file(asset.local_path)
        assert asset.license in {"MIT", "Apache-2.0", "GPL-3.0-only"}


def test_packaged_layout_contains_notices(packaged_dir) -> None:
    assert (packaged_dir / "天天象棋学习助手.exe").exists()
    assert (packaged_dir / "THIRD_PARTY_NOTICES.md").exists()
    bundled_engine = packaged_dir / "engines" / "pikafish" / "pikafish.exe"
    if bundled_engine.exists():
        assert (packaged_dir / "licenses" / "Pikafish-GPL-3.0.txt").exists()
    else:
        assert (packaged_dir / "scripts" / "download_pikafish.py").exists()
~~~

- [ ] **Step 2: 确认测试失败**

Run: .\.venv\Scripts\python -m pytest tests\integration\test_asset_integrity.py tests\integration\test_packaged_layout.py -q

Expected: FAIL，发布目录和许可证文件尚未生成。

- [ ] **Step 3: 实现可重复目录版打包**

PyInstaller spec 使用 windowed=True、collect Qt plugins、包含选定 ONNX 模型、默认图标和许可证，不包含用户 settings、数据库、API Key 或诊断图片。Pikafish 若随包发布，放入 engines/pikafish/ 并附 GPL 文本、精确源码 URL 和 assets.lock 中的 SHA-256；若 NNUE 再分发权利未通过审计，则安装向导要求用户从官方 Release 下载，打包测试相应验证 engines 目录为空且下载脚本存在。

README.md 写清安装、学习模式边界、首次标定、API Key 设置、Pikafish 安装、最小化限制、隐私和卸载数据位置。环境自检依次验证 Windows 版本、Python/打包模式、模型哈希、Pikafish uciok、数据库可写、Credential Manager 和 DeepSeek 连通性；DeepSeek 连通性允许跳过。

- [ ] **Step 4: 锁定依赖、构建并运行打包测试**

Run:

~~~powershell
.\.venv\Scripts\python -m pip freeze | Set-Content -Encoding utf8 packaging\requirements-lock.txt
.\.venv\Scripts\pyinstaller --noconfirm packaging\xiangqi_agent.spec
.\.venv\Scripts\python -m pytest tests\integration\test_asset_integrity.py tests\integration\test_packaged_layout.py -q
~~~

Expected: 生成目录版应用；许可证、模型哈希和启动自检全部通过。

- [ ] **Step 5: 提交发布工程**

~~~powershell
git add packaging LICENSE README.md THIRD_PARTY_NOTICES.md tests\integration
git commit -m "build: package auditable Windows application"
~~~

### Task 21: 离线回放、真实天天象棋验收和发布门

**Files:**
- Create: scripts/run_replay.py
- Create: scripts/run_acceptance.py
- Create: tests/replay/test_full_game.py
- Create: tests/acceptance/conftest.py
- Create: tests/acceptance/test_release_gate.py
- Create: docs/acceptance-checklist.md
- Create: docs/recognition-report.md
- Create: docs/release-notes-0.1.0.md

**Interfaces:**
- Consumes: 录制的棋盘裁图序列、人工真值棋谱、完整应用
- Produces: replay-report.json、acceptance-report.json
- Produces: 进程退出码 0 代表全部自动发布门通过

- [ ] **Step 1: 写发布门聚合测试**

~~~python
def test_release_gate(release_metrics) -> None:
    assert release_metrics.complete_games == 10
    assert release_metrics.total_half_moves >= 400
    assert release_metrics.move_sequence_accuracy == 1.0
    assert release_metrics.stable_board_accuracy >= 0.99
    assert release_metrics.sync_latency_p95_seconds <= 1.0
    assert release_metrics.quick_analysis_p95_seconds <= 2.0
    assert release_metrics.crashes == 0
    assert release_metrics.orphan_engine_processes == 0
    assert release_metrics.secret_leaks == 0
~~~

- [ ] **Step 2: 确认空报告不能通过**

Run: .\.venv\Scripts\python -m pytest tests\acceptance\test_release_gate.py -q

Expected: FAIL，缺少 acceptance-report.json 或全部指标为零。

- [ ] **Step 3: 实现离线重放与指标采集**

run_replay.py 按原时间戳或 --speed 选项回放裁图，记录每个稳定观察、确认走法、同步延迟、误拒绝和错误接受；输出的走法序列必须与真值逐半回合比较。run_acceptance.py 聚合 10 盘报告、30 分钟耐久运行、窗口缩放/DPI 场景、引擎残留检查和日志密钥扫描。tests/acceptance/conftest.py 从环境变量 XIANGQI_ACCEPTANCE_REPORT 指向的 JSON 加载 release_metrics；变量缺失或文件不存在时测试明确失败。

docs/acceptance-checklist.md 必须逐项记录：

- 红方在下和黑方在下各至少 3 盘。
- 100%、125%、150% DPI 均验证。
- 窗口移动、缩放、遮挡、最小化与恢复均验证。
- 普通走子、吃子、将军、绝杀、悔棋后重新完整同步均验证。
- DeepSeek 未配置、超时、余额不足时本地分析仍可用。
- 诊断关闭时没有截图，开启时只有棋盘裁图且七日清理有效。

- [ ] **Step 4: 执行完整自动和人工验收**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy src
.\.venv\Scripts\python scripts\run_replay.py --manifest tests\fixtures\frames\manifest.json --output .local\replay-report.json
.\.venv\Scripts\python scripts\run_acceptance.py --replay .local\replay-report.json --output .local\acceptance-report.json
$env:XIANGQI_ACCEPTANCE_REPORT=(Resolve-Path '.local\acceptance-report.json').Path
.\.venv\Scripts\python -m pytest tests\acceptance\test_release_gate.py -q -m acceptance
Remove-Item Env:XIANGQI_ACCEPTANCE_REPORT
~~~

Expected: 所有自动检查通过；人工清单签字完成；若任一发布门失败，修复对应模块并重新执行完整门，不修改门槛。

- [ ] **Step 5: 写识别报告和 0.1.0 发布说明**

docs/recognition-report.md 记录测试机器、微信版本、天天象棋皮肤、模型来源、训练集/验证集数量、混淆矩阵摘要、各 DPI 指标和已知限制。release notes 只声明通过验收的功能，不把未测试主题或分辨率描述为支持。

- [ ] **Step 6: 提交验收证据并打标签**

~~~powershell
git add scripts tests\replay tests\acceptance docs
git commit -m "test: complete Windows learning assistant acceptance"
git tag -a v0.1.0 -m "Xiangqi learning assistant 0.1.0"
~~~

## Milestone Checkpoints

1. **领域内核完成（Task 1-4）：** 无 GUI、微信、模型或引擎也能完成 FEN、规则、记谱和配置测试。
2. **可见桌面原型（Task 5-8）：** 能启动独立窗口、选择微信窗口、捕获画面并完成四角标定。
3. **可靠同步原型（Task 9-12）：** 能从真实天天象棋画面更新镜像棋盘，错误识别不会污染确认局面。
4. **本地分析原型（Task 13-14）：** 不配置 DeepSeek 也能显示两阶段 Pikafish 形势和前三候选。
5. **教学助手原型（Task 15-17）：** 能基于结构化证据分级解释、追问和比较用户想法。
6. **可复盘版本（Task 18-19）：** 棋谱、评分、问答、本地恢复和隐私策略形成闭环。
7. **可交付版本（Task 20-21）：** Windows 目录版、许可证、十盘真机验收和 0.1.0 发布门全部通过。

## Execution Notes

- 当前目录尚未初始化 Git，Task 1 才执行 git init；计划文档本身因此尚无提交记录。
- assets.lock.json 在 Task 9 由实际下载与哈希计算产生，禁止手工填入虚假哈希。
- Task 9 是识别可靠性的硬门；模型不达标时必须走补充采集和训练分支，不能提前进入 Task 10。
- Task 13 的 Pikafish 再分发材料和 Task 20 的打包方式必须一起审计；个人本地开发可先使用 .local 目录中的官方二进制。
- Task 21 的真人窗口验收只在人机、残局或复盘模式执行。
