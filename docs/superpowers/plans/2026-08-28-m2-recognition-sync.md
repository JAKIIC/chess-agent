# M2 Recognition and Sync Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 通过可审计的识别资产评估门、ONNX 整盘识别、多帧稳定与合法走法验证，实现不会污染已确认局面的可靠同步原型。

**Architecture:** 视觉层只产出 ObservedPosition，时序层只确认稳定观察，sync 层作为唯一可更新 BoardState 的边界；任何低置信、非法或多步变化都保持最后确认局面。

**Tech Stack:** Python 3.12、PySide6、windows-capture 2.0.1、OpenCV、NumPy、ONNX Runtime、Pikafish 2026-01-02、HTTPX、Pydantic、SQLite、keyring、pytest、pytest-qt、ruff、mypy、PyInstaller。

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

**Roadmap Scope:** Tasks 9-12 from docs/superpowers/plans/2026-08-28-xiangqi-learning-agent.md. The task sections below are copied verbatim from that roadmap.

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
├─ assets.lock.json
├─ THIRD_PARTY_NOTICES.md
├─ src/xiangqi_agent/
│  ├─ vision/{locator,model,recognizer,temporal}.py
│  ├─ sync/{state_machine,service}.py
│  ├─ application/controller.py
│  └─ ui/{board_widget,main_window}.py
├─ src/xiangqi_agent/ui/dialogs/
│  ├─ connect_dialog.py
│  ├─ calibration_dialog.py
│  └─ correction_dialog.py
├─ scripts/{evaluate_models,collect_intersections,train_classifier}.py
└─ tests/{unit,integration,ui,fixtures}/
~~~

文件仍按总路线图的职责边界拆分；本节只列出本里程碑直接创建或修改的主要区域。

---

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

---

## Milestone Acceptance Gate

- 候选识别资产具有固定来源、提交、许可证和 SHA-256；仅达到既定准确率、F1 与 CPU 延迟门槛的模型可进入运行时。
- 回放与合成测试覆盖红黑方向、DPI、选中高亮、动画和吃子变化。
- 只有连续稳定且可由唯一合法走法解释的观察更新 confirmed_board；不确定观察、非法变化和过期结果均不得进入分析层。
- 连接、初始局面与行棋方确认、镜像棋盘高亮和手工纠正流程通过离线及 UI 测试。

## Milestone Tag

全部验收门通过并提交验收证据后，创建 annotated tag `v0.1.0-m2`。

## Push Gate

只有本里程碑的全部测试通过、独立评审完成、隐私扫描无发现且工作树干净时，才允许推送分支和 annotated tag。任一条件未满足时不得推送。
