# M1 Foundation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 建立可测试的 Python 工程、领域模型、规则与记谱、安全配置、PySide6 桌面壳、微信窗口发现、只读捕获和棋盘标定基础。

**Architecture:** 先固化不依赖 Qt、OpenCV、Windows 或网络的领域内核，再以协议隔离平台窗口和帧源，最后把捕获与标定结果通过 Qt 信号接入独立桌面壳。

**Tech Stack:** Python 3.12、PySide6、windows-capture 2.0.1、OpenCV、NumPy、ONNX Runtime、Pikafish 2026-01-02、HTTPX、Pydantic、SQLite、keyring、pytest、pytest-qt、ruff、mypy、PyInstaller。

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

**Roadmap Scope:** Tasks 1-8 from docs/superpowers/plans/2026-08-28-xiangqi-learning-agent.md. The task sections below are copied verbatim from that roadmap.

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
- 每个任务保留完整接口、命令、测试预期和提交消息；不得跳过失败测试或验收门。

---

## Planned File Structure

~~~text
.
├─ pyproject.toml
├─ src/xiangqi_agent/
│  ├─ bootstrap.py
│  ├─ config.py
│  ├─ application/controller.py
│  ├─ domain/{board,fen,rules,notation}.py
│  ├─ diagnostics/logging.py
│  ├─ platform/{shortcut,windows}.py
│  ├─ capture/{protocol,fake,windows_capture_source}.py
│  ├─ vision/{types,geometry}.py
│  └─ ui/{main_window,board_widget}.py
├─ tests/{unit,integration,ui}/
└─ scripts/smoke_capture.py
~~~

文件仍按总路线图的职责边界拆分；本节只列出本里程碑直接创建或修改的主要区域。

---

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
  "PySide6>=6.8,<6.10",
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

---

## Milestone Acceptance Gate

- `pytest -q`、`ruff check .` 和 `mypy src` 全部通过。
- FEN、position_id、规则、中文记谱、配置、脱敏和密钥存储具有自动化测试证据。
- 独立窗口可以选择目标微信窗口、建立只读捕获并完成四角标定；不执行任何点击、注入、内存读取或流量代理。
- 捕获冒烟测试确认窗口关闭或最小化时明确暂停，且日志不包含窗口标题和 API Key。

## Milestone Tag

全部验收门通过并提交验收证据后，创建 annotated tag `v0.1.0-m1`。

## Push Gate

只有本里程碑的全部测试通过、独立评审完成、隐私扫描无发现且工作树干净时，才允许推送分支和 annotated tag。任一条件未满足时不得推送。
