# Stage C 隔离采集、事后复核与验证晋级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不要求用户预知人机回复、也不保存完整棋盘画面的前提下，把一次真实终局事件安全地隔离保存，由本地用户事后确认或纠正，再经棋规、占位与哈希校验晋级为可冻结、可确定性重放的 Stage C V2 样本。

**Architecture:** `LiveSyncSession` 只产生内存中的终局证据；独立占位观察器为起始与终局提供 90 点空/非空证据；隔离记录器保存无真值事件；复核存储保存不可变人工标签；晋级服务重新应用棋规并验证占位和来源哈希；V2 目录自包含来源清单、复核清单和小裁片。现有 V1 读取与重放继续兼容，但新的事后复核入口只生成 V2。

**Tech Stack:** Python 3.12、NumPy、OpenCV、PySide6、pytest、Ruff、mypy、现有 `BoardState`/`legal_moves()`/`RuleStateCommitter`/`LiveSyncSession`/Stage C 决策门。

**Spec:** `docs/superpowers/specs/2026-09-01-stage-c-quarantine-review-design.md`

## Global Constraints

- 用户负责天天象棋中的全部操作；代码不得点击、走棋、注入、读取微信进程内存、激活、移动、缩放或最小化窗口。
- 只用于人机练习、残局训练和赛后复盘；不得为真人在线对局解锁实时建议。
- DeepSeek、Pikafish 和现有观察器候选都不能生成真值；本切片不得调用引擎或教练。
- 诊断采集默认关闭；隔离与复核必须分别显式启用。
- 不持久化完整窗口、完整棋盘、视频、窗口标题、昵称、头像、账号、DeepSeek 请求或 API Key。
- 只允许保存 1–4 个 48×48 BGRA 交点前后 PNG、90 点数值证据和匿名 ID；所有运行数据位于 Git 忽略的 `.local/`。
- 隔离、复核、已晋级目录分别为 `.local/stage-c-quarantine`、`.local/stage-c-reviews`、`.local/stage-c-reviewed`；三者不得相同或互为父子目录。
- 已晋级 V2 目录必须自包含 `source-event-manifest.json` 和 `review-manifest.json`，不能依赖隔离数据继续存在。
- 保留现有 `HumanAiStageCSampleV1` 的只读加载与确定性重放；新的隔离、复核、晋级入口不得生成 V1。
- 生产代码逐任务执行 RED → GREEN → REFACTOR；先看见目标测试按预期失败，再写最小实现。
- 不修改、不暂存、不提交用户所有的未跟踪文件 `docs/superpowers/plans/2026-08-28-remaining-execution-plan.md`。
- 本切片完成后只提交并推送 `develop`；不创建 milestone tag，不合并 `main`。

## Planned File Structure

```text
src/xiangqi_agent/
├─ vision/
│  └─ occupancy.py
├─ sync/
│  ├─ transition_capture.py
│  ├─ tracker.py
│  └─ live_session.py
├─ diagnostics/
│  ├─ stage_c_quarantine.py
│  ├─ stage_c_review.py
│  ├─ stage_c_reviewed_samples.py
│  ├─ stage_c_promotion.py
│  ├─ stage_c_replay.py
│  └─ stage_c_gate.py
└─ ui/
   ├─ stage_c_review_panel.py
   ├─ capture_panel.py
   └─ main_window.py
scripts/
├─ collect_human_ai_quarantine.py
├─ review_human_ai_stage_c.py
├─ promote_human_ai_stage_c.py
├─ freeze_human_ai_stage_c.py
└─ evaluate_human_ai_stage_c.py
tests/
├─ integration/diagnostics/test_stage_c_review_flow.py
└─ unit/
   ├─ vision/test_occupancy.py
   ├─ sync/test_transition_capture.py
   ├─ sync/test_tracker.py
   ├─ sync/test_live_session.py
   ├─ diagnostics/test_stage_c_quarantine.py
   ├─ diagnostics/test_stage_c_review.py
   ├─ diagnostics/test_stage_c_reviewed_samples.py
   ├─ diagnostics/test_stage_c_promotion.py
   ├─ diagnostics/test_stage_c_replay.py
   ├─ diagnostics/test_stage_c_gate.py
   ├─ scripts/test_collect_human_ai_quarantine.py
   ├─ scripts/test_review_human_ai_stage_c.py
   ├─ scripts/test_promote_human_ai_stage_c.py
   ├─ scripts/test_freeze_human_ai_stage_c.py
   ├─ scripts/test_evaluate_human_ai_stage_c.py
   └─ ui/test_stage_c_review_panel.py
docs/status/stage-c-quarantine-review.md
```

---

### Task 1: Privacy-safe 90-point Occupancy Evidence

**Files:**
- Create: `src/xiangqi_agent/vision/occupancy.py`
- Modify: `src/xiangqi_agent/vision/__init__.py`
- Create: `tests/unit/vision/test_occupancy.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class OccupancyEvidence:
    occupied: tuple[bool, ...]
    confidences: tuple[float, ...]
    algorithm_version: str


@dataclass(frozen=True, slots=True)
class OccupancyComparison:
    mismatched_points: tuple[int, ...]
    low_confidence_points: tuple[int, ...]

    @property
    def accepted(self) -> bool: ...


class OccupancyObserver(Protocol):
    def observe(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
    ) -> OccupancyEvidence: ...


class CircularOccupancyObserver:
    algorithm_version = "circular-occupancy-v1"

    def observe(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
    ) -> OccupancyEvidence: ...


def compare_occupancy(
    evidence: OccupancyEvidence,
    board: BoardState,
    *,
    minimum_confidence: float,
) -> OccupancyComparison: ...
```

- [ ] **Step 1: Write failing contract tests**

Create immutable evidence fixtures and assert exactly 90 booleans, exactly 90 finite confidences in `[0, 1]`, a non-empty algorithm version, defensive tuple ownership and stable point order. Reject booleans masquerading as numeric confidences, NaN, infinity, wrong lengths and non-`BoardState` comparison inputs.

```python
def test_compare_occupancy_reports_mismatch_and_low_confidence_separately() -> None:
    evidence = evidence_for(START_BOARD, confidence=0.95)
    occupied = list(evidence.occupied)
    occupied[0] = not occupied[0]
    confidences = list(evidence.confidences)
    confidences[1] = 0.49

    comparison = compare_occupancy(
        replace(evidence, occupied=tuple(occupied), confidences=tuple(confidences)),
        START_BOARD,
        minimum_confidence=0.50,
    )

    assert comparison.mismatched_points == (0,)
    assert comparison.low_confidence_points == (1,)
    assert not comparison.accepted
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\vision\test_occupancy.py -q
```

Expected: collection fails because `xiangqi_agent.vision.occupancy` does not exist.

- [ ] **Step 3: Implement deterministic circular occupancy scoring**

Use `BoardGeometry.crop_intersections(frame, size=48)` to create only 90 owned local patches. For each patch:

1. Convert BGRA to grayscale and Lab luminance.
2. Measure Canny edge density inside a circular annulus covering radii 14–22 pixels.
3. Measure absolute Lab-luminance contrast between the center disk and four corner background regions.
4. Measure radial-gradient agreement in the annulus.
5. Normalize the three values with frozen constants and compute `score = 0.50 * annulus + 0.30 * contrast + 0.20 * radial`.
6. Mark occupied at `score >= 0.52`; derive confidence from the distance to `0.52`, clipped to `[0, 1]`.

The returned object contains no patch or frame reference. Keep the constants and algorithm version in the module so a later change requires a new version rather than silently changing old evidence.

- [ ] **Step 4: Test geometry, orientation and privacy behavior**

Generate synthetic board-line patches plus circular-piece patches at two scales and both orientations. Assert stable 90-point order, correct empty/occupied classification, deterministic repeated output, no mutation after changing the source frame and no array-valued field on `OccupancyEvidence`. Add white-frame and uniform-overlay cases that produce low confidence or mismatches against the standard board.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\vision\test_occupancy.py tests\unit\vision\test_geometry.py -q
git add src\xiangqi_agent\vision\occupancy.py src\xiangqi_agent\vision\__init__.py tests\unit\vision\test_occupancy.py
git commit -m "feat: add privacy-safe board occupancy evidence"
```

---

### Task 2: Occupancy-gated Baselines and Terminal Evidence

**Files:**
- Modify: `src/xiangqi_agent/sync/transition_capture.py`
- Modify: `src/xiangqi_agent/sync/tracker.py`
- Modify: `src/xiangqi_agent/sync/live_session.py`
- Create: `tests/unit/sync/test_transition_capture.py`
- Modify: `tests/unit/sync/test_tracker.py`
- Modify: `tests/unit/sync/test_live_session.py`

**Interface changes:**

```python
@dataclass(frozen=True, slots=True)
class TransitionCaptureEvidence:
    changed_points: tuple[int, ...]
    local_differences: tuple[float, ...]
    crops: tuple[TransitionPointEvidence, ...]
    decision_latency_ms: float
    before_occupancy: OccupancyEvidence | None = None
    after_occupancy: OccupancyEvidence | None = None


class StableMoveTracker:
    def __init__(
        ...,
        capture_transition_evidence: bool = False,
        occupancy_observer: OccupancyObserver | None = None,
    ) -> None: ...


class LiveSyncSession:
    def __init__(
        ...,
        capture_transition_evidence: bool = False,
        occupancy_observer: OccupancyObserver | None = None,
        require_matching_baseline: bool = False,
        baseline_minimum_confidence: float = 0.65,
    ) -> None: ...
```

- [ ] **Step 1: Write failing transition evidence tests**

Assert `before_occupancy` and `after_occupancy` are both absent or both present. With an observer, `build_transition_capture_evidence()` must call it exactly once for the confirmed frame and once for the settled frame and preserve the current 1–4 crop limit. Without an observer, the current API and serialized evidence remain unchanged.

- [ ] **Step 2: Write failing baseline safety tests**

Use `FakeFrameSource` plus a literal occupancy observer to prove:

- matching high-confidence occupancy emits `BASELINE_READY` once;
- one mismatched point emits `CONTEXT_INVALID` and never initializes the tracker;
- one point below the baseline confidence floor cannot create a baseline;
- a later matching stable frame can establish the baseline without restarting the session;
- `require_matching_baseline=True` without an observer fails at construction;
- defaults preserve all existing `LiveSyncSession` behavior.

```python
def test_mismatched_occupancy_never_becomes_a_white_screen_baseline() -> None:
    source = FakeFrameSource()
    observer = SequenceOccupancyObserver((mismatched(START_BOARD),))
    updates: list[LiveSyncUpdate] = []
    session = LiveSyncSession(
        source,
        START_BOARD,
        QUAD,
        on_update=updates.append,
        occupancy_observer=observer,
        require_matching_baseline=True,
    )

    session.start()
    source.emit_stable(WHITE_FRAME)

    assert LiveSyncStatus.BASELINE_READY not in {item.status for item in updates}
    assert LiveSyncStatus.CONTEXT_INVALID in {item.status for item in updates}
```

- [ ] **Step 3: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync\test_transition_capture.py tests\unit\sync\test_tracker.py tests\unit\sync\test_live_session.py -q
```

Expected: failures show the new occupancy arguments and evidence fields are absent.

- [ ] **Step 4: Implement baseline and transition integration**

Before constructing `StableMoveTracker`, `LiveSyncSession` observes the stable frame and compares it to the confirmed board. A mismatch or low-confidence point emits a coalesced `CONTEXT_INVALID` update with message `target board is not visibly consistent with the confirmed position`, retains no confirmed frame and continues waiting. A matching frame initializes normally.

When terminal capture is enabled, the tracker observes the retained confirmed frame and settled terminal frame while building `TransitionCaptureEvidence`. It does not persist them. Context invalidation, resize failure and `close()` clear all frame-backed state. Optional defaults must keep existing callers source-compatible.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync\test_transition_capture.py tests\unit\sync\test_tracker.py tests\unit\sync\test_live_session.py tests\unit\scripts\test_collect_human_ai_stage_c.py -q
git add src\xiangqi_agent\sync\transition_capture.py src\xiangqi_agent\sync\tracker.py src\xiangqi_agent\sync\live_session.py tests\unit\sync
git commit -m "feat: validate live baselines with occupancy evidence"
```

---

### Task 3: Unlabelled Quarantine Contract, Loader and Recorder

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_quarantine.py`
- Modify: `src/xiangqi_agent/diagnostics/__init__.py`
- Create: `tests/unit/diagnostics/test_stage_c_quarantine.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class QuarantinedStageCEventV1:
    event_id: str
    session_id: str
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
    observed_status: StageCObservedStatus
    observed_moves_uci: tuple[str, ...]
    observed_final_position_id: str | None
    side_to_move: Side
    orientation: Orientation
    changed_points: tuple[int, ...]
    local_differences: tuple[float, ...]
    candidates: tuple[StageCCandidateRecord, ...]
    rejection_reasons: tuple[str, ...]
    capture_context: CaptureContext
    feature_version: str
    threshold_profile_version: str
    decision_latency_ms: float
    before_occupancy: OccupancyEvidence
    after_occupancy: OccupancyEvidence
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class LoadedQuarantinedStageCEvent:
    metadata: QuarantinedStageCEventV1
    crops: tuple[TransitionPointCrops, ...]
    directory: Path
    manifest_bytes: bytes


class QuarantineEventRecorder:
    def record(
        self,
        event: QuarantinedStageCEventV1,
        crops: tuple[TransitionPointCrops, ...],
    ) -> Path: ...


class QuarantineEventLoader:
    def load(self, event_dir: Path) -> LoadedQuarantinedStageCEvent: ...
```

- [ ] **Step 1: Write failing schema tests that forbid truth fields**

Assert valid accepted-observation and rejected-observation shapes while proving that neither the dataclass fields nor JSON manifest contain `expected_outcome`, `ground_truth_moves_uci`, `expected_final_position_id`, `label_kind`, `scenario`, `passed` or `accepted_as_truth`.

```python
def test_quarantine_manifest_contains_observation_but_no_truth(tmp_path: Path) -> None:
    event_dir = enabled_recorder(tmp_path).record(EVENT, CROPS)
    payload = json.loads((event_dir / "manifest.json").read_text("utf-8"))

    assert payload["observed_status"] == "rejected"
    assert {
        "expected_outcome",
        "ground_truth_moves_uci",
        "expected_final_position_id",
        "label_kind",
        "scenario",
    }.isdisjoint(payload)
```

Also test exact 90-point occupancy, matching algorithm versions, 1–4 crop points, sorted candidates, anonymous path-safe IDs, UTC timestamps, finite latency, and exact manifest fields.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_quarantine.py -q
```

Expected: collection fails because the quarantine module does not exist.

- [ ] **Step 3: Implement fail-closed atomic persistence**

Write to `<root>/.<event_id>.<uuid>.tmp`, encode only declared 48×48 crops, hash each PNG, write sorted UTF-8 JSON, reload and verify the temporary directory, then atomically rename to `<root>/<session_id>/<event_id>`. The recorder is disabled unless `enabled=True`, refuses symlinks/path traversal/existing destinations, enforces a 256 MiB root quota and seven-day UTC retention, and never purges data before the incoming event has fully validated.

The loader requires exactly `manifest.json` plus declared crops, validates all hashes and returns the original manifest bytes for later provenance hashing.

- [ ] **Step 4: Add tamper, quota, cleanup and privacy tests**

Cover changed PNG, extra file, malformed UTF-8 JSON, duplicate ID, interrupted temporary directory, exact retention cutoff, byte quota, symlink escape and source-array mutation. Scan every manifest string to ensure it contains no title, nickname, avatar, account, key, request, full-frame path or user name.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_quarantine.py tests\unit\diagnostics\test_stage_c_samples.py -q
git add src\xiangqi_agent\diagnostics\stage_c_quarantine.py src\xiangqi_agent\diagnostics\__init__.py tests\unit\diagnostics\test_stage_c_quarantine.py
git commit -m "feat: record unlabeled Stage C events in quarantine"
```

---

### Task 4: One-event Unlabelled Human-vs-AI Collector

**Files:**
- Create: `scripts/collect_human_ai_quarantine.py`
- Create: `tests/unit/scripts/test_collect_human_ai_quarantine.py`
- Modify: `tests/unit/scripts/test_collect_human_ai_stage_c.py` only if shared test fixtures are extracted

**CLI contract:**

```text
collect_human_ai_quarantine.py
  --fen <confirmed-fen>
  --quad <x1,y1;x2,y2;x3,y3;x4,y4>
  --session-id <anonymous-id>
  --output-root .local/stage-c-quarantine
  [--hwnd <integer>]
  [--timeout-seconds 180]
```

The parser must not expose `--expected-moves`, `--expect-reject`, `--scenario` or any other truth argument.

**Function:**

```python
def collect_human_ai_quarantine_event(
    *,
    source: FrameSource,
    board: BoardState,
    quad: NormalizedQuad,
    session_id: str,
    output_root: Path,
    timeout_seconds: float,
    on_update: Callable[[LiveSyncUpdate], None] | None = None,
) -> Path: ...
```

- [ ] **Step 1: Write failing CLI and fake-frame tests**

Prove the collector:

- starts `HUMAN_VS_AI` at 2 FPS with burst sampling and `capture_transition_evidence=True`;
- uses `CircularOccupancyObserver` and `require_matching_baseline=True`;
- records only after one accepted or safely rejected terminal event;
- converts terminal evidence to `QuarantinedStageCEventV1` without adding truth;
- writes nothing for timeout, no target window, multiple unselected target windows, window close, baseline mismatch, resize or missing occupancy;
- closes the source exactly once even when an error occurs;
- never produces a full-frame-sized image.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\scripts\test_collect_human_ai_quarantine.py -q
```

Expected: collection fails because the new collector script does not exist.

- [ ] **Step 3: Implement the collector without computer control**

Use `WindowsWindowCatalog`, `filter_target_windows()` and optional `--hwnd` for read-only selection, then `VisibleWindowCaptureSource(window, fps=2, burst_fps=20)`. Do not invoke any API that activates, clicks, moves, resizes or minimizes the window. Print only stable JSON lines containing anonymous status, `event_id`, `session_id`, frame size, point count and error code; omit title, FEN and moves from console output.

At the first terminal event, detach the immutable transition evidence, close the session, build the quarantine event and call an explicitly enabled recorder. A terminal update without both occupancy snapshots is an integrity error and writes no event.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\scripts\test_collect_human_ai_quarantine.py tests\unit\scripts\test_collect_human_ai_stage_c.py tests\unit\sync\test_live_session.py -q
git add scripts\collect_human_ai_quarantine.py tests\unit\scripts\test_collect_human_ai_quarantine.py tests\unit\scripts\test_collect_human_ai_stage_c.py
git commit -m "feat: collect one unlabeled human-ai event safely"
```

---

### Task 5: Immutable Local Review Contract and Legal Move Choices

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_review.py`
- Create: `scripts/review_human_ai_stage_c.py`
- Create: `tests/unit/diagnostics/test_stage_c_review.py`
- Create: `tests/unit/scripts/test_review_human_ai_stage_c.py`

**Interfaces:**

```python
class StageCLabelKind(StrEnum):
    VALID_TWO_PLY = "valid_two_ply"
    EXPECTED_REJECTION = "expected_rejection"
    DISCARD = "discard"


class StageCReviewOutcome(StrEnum):
    CANDIDATE_CONFIRMED = "candidate_confirmed"
    LEGAL_MOVE_CORRECTION = "legal_move_correction"
    EXPECTED_REJECTION = "expected_rejection"
    DISCARDED = "discarded"


@dataclass(frozen=True, slots=True, kw_only=True)
class StageCReviewV1:
    review_id: str
    event_id: str
    session_id: str
    created_at_utc: str
    event_manifest_sha256: str
    label_kind: StageCLabelKind
    moves_uci: tuple[str, ...]
    expected_final_position_id: str | None
    scenario: StageCScenario | None
    review_outcome: StageCReviewOutcome
    supersedes_review_id: str | None
    reviewer_kind: str = "local_user"
    ui_version: str = "stage-c-review-v1"
    rules_version: str = "xiangqi-rules-v1"
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class ReviewMoveChoice:
    uci: str
    chinese: str
    resulting_position_id: str


def legal_review_choices(board: BoardState) -> tuple[ReviewMoveChoice, ...]: ...


class StageCReviewStore:
    def submit(self, review: StageCReviewV1) -> Path: ...

    def active_review(self, session_id: str, event_id: str) -> StageCReviewV1 | None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class StageCReviewDraft:
    label_kind: StageCLabelKind
    moves_uci: tuple[str, ...]
    scenario: StageCScenario | None
    review_outcome: StageCReviewOutcome
    supersedes_review_id: str | None = None


class StageCReviewService:
    def submit(self, event_dir: Path, draft: StageCReviewDraft) -> Path: ...
```

- [ ] **Step 1: Write failing schema and legal-choice tests**

At the dataclass boundary, `valid_two_ply` requires exactly two structurally valid UCI moves, a final `position_id`, no rejection scenario and either candidate confirmation or legal correction. `expected_rejection` requires one of the six frozen rejection scenarios, 0–3 UCI moves and no expected final ID. `discard` requires no moves, final ID or scenario. `StageCReviewService` loads the event and then requires every supplied move to form a sequentially legal chain; for valid labels it derives the final `position_id` itself instead of trusting user input.

Assert `legal_review_choices()` is sorted by `(chinese, uci)`, contains only `legal_moves(board)`, and derives every resulting ID by `apply_move()`. After selecting the first move, call it again on the projected board for the legal second-move list.

- [ ] **Step 2: Write failing immutable-store tests**

The store writes exactly one file at `<review-root>/<session_id>/<event_id>/<review_id>.json` with exclusive creation. A correction must set `supersedes_review_id` to the active review; it does not overwrite the old file. Reject cycles, unknown superseded IDs, multiple active branches, path traversal, symlinks and an event hash that is not 64 lowercase hex characters. Enforce an 8 MiB review-root quota; review files are small JSON only and the service refuses a new write rather than deleting an active review.

- [ ] **Step 3: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_review.py tests\unit\scripts\test_review_human_ai_stage_c.py -q
```

Expected: collection fails because review types, store and CLI are absent.

- [ ] **Step 4: Implement review validation and CLI**

The CLI loads one quarantined event by anonymous path and computes the event manifest hash itself. Exact commands are:

```text
--label valid_two_ply --moves h2e2,h7e7 --outcome candidate_confirmed
--label valid_two_ply --moves h2e2,h7e7 --outcome legal_move_correction
--label expected_rejection --scenario occlusion [--moves <0-to-3-comma-separated-uci>]
--label discard
```

`StageCReviewService` validates every move against the existing rules, computes the source-manifest hash and final ID, creates the immutable review and then delegates to the store. The CLI prints legal choices as Chinese notation plus UCI only when `--list-legal` is supplied. It never prints candidate scores and never calls engine or coach code.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_review.py tests\unit\scripts\test_review_human_ai_stage_c.py tests\unit\domain\test_notation.py tests\unit\domain\test_rules.py -q
git add src\xiangqi_agent\diagnostics\stage_c_review.py scripts\review_human_ai_stage_c.py tests\unit\diagnostics\test_stage_c_review.py tests\unit\scripts\test_review_human_ai_stage_c.py
git commit -m "feat: review quarantined Stage C events"
```

---

### Task 6: Deterministic Promotion and Self-contained V2 Samples

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_reviewed_samples.py`
- Create: `src/xiangqi_agent/diagnostics/stage_c_promotion.py`
- Modify: `src/xiangqi_agent/diagnostics/stage_c_samples.py`
- Create: `scripts/promote_human_ai_stage_c.py`
- Create: `tests/unit/diagnostics/test_stage_c_reviewed_samples.py`
- Create: `tests/unit/diagnostics/test_stage_c_promotion.py`
- Create: `tests/unit/scripts/test_promote_human_ai_stage_c.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewedStageCSampleV2:
    sample_id: str
    session_id: str
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
    expected_outcome: StageCExpectedOutcome
    scenario: StageCScenario
    ground_truth_moves_uci: tuple[str, ...]
    expected_final_position_id: str | None
    observed_status: StageCObservedStatus
    observed_moves_uci: tuple[str, ...]
    observed_final_position_id: str | None
    side_to_move: Side
    orientation: Orientation
    changed_points: tuple[int, ...]
    local_differences: tuple[float, ...]
    candidates: tuple[StageCCandidateRecord, ...]
    rejection_reasons: tuple[str, ...]
    capture_context: CaptureContext
    feature_version: str
    threshold_profile_version: str
    decision_latency_ms: float
    source_event_manifest_sha256: str
    review_manifest_sha256: str
    label_source: str = "post_event_local_user_review"
    review_outcome: StageCReviewOutcome
    occupancy_verifier_version: str
    promotion_verifier_version: str
    promoted_at_utc: str
    schema_version: int = 2


class PromotionStatus(StrEnum):
    PROMOTABLE = "promotable"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: PromotionStatus
    reason_codes: tuple[str, ...]
    projected_final_position_id: str | None


class StageCPromotionVerifier:
    verifier_version = "stage-c-promotion-v1"

    def verify(self, event_dir: Path, review_path: Path) -> PromotionDecision: ...


class StageCPromotionService:
    def promote(
        self,
        event_dir: Path,
        review_path: Path,
        reviewed_root: Path,
    ) -> Path: ...


def purge_expired_reviewed_samples(
    reviewed_root: Path,
    *,
    protected_relative_paths: frozenset[str],
    now_utc: datetime,
    retention_days: int = 30,
) -> tuple[Path, ...]: ...
```

- [ ] **Step 1: Extract shared replay-field validation without changing V1**

Write a characterization test that serializing and loading an existing V1 fixture remains byte-for-byte compatible. Extract private validation helpers from `HumanAiStageCSampleV1.__post_init__` only as needed so V2 can reuse identical FEN, candidates, observations, differences, latency and version checks. Do not change V1 field names, manifest shape or schema version.

- [ ] **Step 2: Write failing valid-event promotion tests**

Assert the verifier, in fixed order:

1. loads and hashes the exact event and review bytes;
2. checks IDs, paths, timestamps and the active immutable review;
3. applies exactly two legal moves from the confirmed FEN;
4. matches the projected final `position_id` to the review;
5. requires all baseline points above confidence `0.65` and matching the confirmed board;
6. requires every high-confidence final point to match the projected board;
7. requires every rule-changed point to have positive local difference and final confidence at least `0.65`.

A low-confidence changed point returns `NEEDS_REVIEW`; illegal moves, wrong final ID, occupancy mismatch, modified crop/event/review or inconsistent ID returns `REJECTED`. The observer may have rejected or omitted the true candidate; a valid user label can still be `PROMOTABLE` and will later count as a missed valid event.

- [ ] **Step 3: Write failing rejection-event promotion tests**

Cover all six scenarios. Resize and capture-context invalidation need machine terminal evidence. Three-ply requires exactly three legal moves. Multiple-candidate needs at least two replayable candidates or non-unique occupancy/local evidence. Selection highlight, animation and occlusion require that the observed status did not commit a new position. A user cannot relabel an ordinary accepted event as resize or occlusion without matching evidence.

- [ ] **Step 4: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_reviewed_samples.py tests\unit\diagnostics\test_stage_c_promotion.py tests\unit\scripts\test_promote_human_ai_stage_c.py -q
```

Expected: failures show V2, verifier, service and CLI do not exist.

- [ ] **Step 5: Implement atomic self-contained promotion**

For `PROMOTABLE`, derive all truth fields from the review and rule projection, never from observer scores. Write a temporary sibling directory containing:

```text
manifest.json
source-event-manifest.json
review-manifest.json
point-<index>-before.png
point-<index>-after.png
```

Copy bytes only after their source hashes pass; record SHA-256 for both sidecars and every crop; reload the temporary V2 directory using the real loader; then atomically rename to `<reviewed-root>/<session_id>/<event_id>`. Refuse an existing target, repeated promotion, symlink boundary, quarantine/reviewed root overlap or source mutation between verification and copy.

The CLI exits `0` and prints only the anonymous sample ID on promotion, `1` for `needs_review`, and `2` for integrity/configuration rejection. `discard` never calls promotion. Add deterministic 30-day cleanup tests: cleanup requires the caller to provide the complete frozen relative-path protection set, deletes only expired unprotected V2 directories, and fails closed on a malformed or symlinked directory. It has no default that could silently ignore frozen protection.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_samples.py tests\unit\diagnostics\test_stage_c_reviewed_samples.py tests\unit\diagnostics\test_stage_c_promotion.py tests\unit\scripts\test_promote_human_ai_stage_c.py -q
git add src\xiangqi_agent\diagnostics\stage_c_samples.py src\xiangqi_agent\diagnostics\stage_c_reviewed_samples.py src\xiangqi_agent\diagnostics\stage_c_promotion.py scripts\promote_human_ai_stage_c.py tests\unit\diagnostics tests\unit\scripts\test_promote_human_ai_stage_c.py
git commit -m "feat: promote verified Stage C samples"
```

---

### Task 7: V2 Replay, Reviewed-only Freeze and Audit Metrics

**Files:**
- Modify: `src/xiangqi_agent/diagnostics/stage_c_replay.py`
- Modify: `src/xiangqi_agent/diagnostics/stage_c_gate.py`
- Modify: `scripts/freeze_human_ai_stage_c.py`
- Modify: `scripts/evaluate_human_ai_stage_c.py`
- Modify: `tests/unit/diagnostics/test_stage_c_replay.py`
- Modify: `tests/unit/diagnostics/test_stage_c_gate.py`
- Modify: `tests/unit/scripts/test_freeze_human_ai_stage_c.py`
- Modify: `tests/unit/scripts/test_evaluate_human_ai_stage_c.py`

**Interface changes:**

```python
type StageCSampleMetadata = HumanAiStageCSampleV1 | ReviewedStageCSampleV2


@dataclass(frozen=True, slots=True)
class LoadedHumanAiStageCSample:
    metadata: StageCSampleMetadata
    crops: tuple[TransitionPointCrops, ...]
    directory: Path


def freeze_reviewed_human_ai_stage_c(
    reviewed_root: Path,
    output_name: str,
    *,
    feature_version: str = DEFAULT_STAGE_C_FEATURE_VERSION,
    threshold_profile: SequenceThresholdProfile = DEFAULT_STAGE_C_THRESHOLD_PROFILE,
    created_at_utc: str | None = None,
) -> Path: ...
```

- [ ] **Step 1: Write failing V2 loader and replay tests**

Load both existing V1 fixtures and new V2 fixtures. For V2 require exactly manifest, two provenance sidecars and declared crops; verify both sidecar hashes, crop hashes, `label_source`, review outcome, verifier versions and promotion timestamp. Changing either sidecar by one byte must fail before replay. Replaying the same V2 twice must produce the same decision as its equivalent V1 evidence.

- [ ] **Step 2: Write failing reviewed-only freeze tests**

`freeze_reviewed_human_ai_stage_c()` accepts only a root whose direct sample directories are `<session>/<sample>`, loads only V2 and validates self-contained provenance. It rejects the quarantine root, review root, nested unknown manifest, V1 sample, `needs_review`, discarded event, extra file, mixed feature/profile version and any source-sidecar mutation. Keep the old `freeze_human_ai_stage_c()` callable for read-only legacy tests, but change the production CLI to the reviewed-only function.

- [ ] **Step 3: Write failing audit-metric tests**

Extend replay results and reports with optional review provenance. The report must separately count `candidate_confirmed`, `legal_move_correction` and `expected_rejection` V2 samples while preserving existing 30 valid sessions, 30 rejection samples, six scenarios, zero false accepts, 80% coverage and 500 ms P95 gates. V1 contributes to legacy totals but not to review-outcome counts.

- [ ] **Step 4: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_replay.py tests\unit\diagnostics\test_stage_c_gate.py tests\unit\scripts\test_freeze_human_ai_stage_c.py tests\unit\scripts\test_evaluate_human_ai_stage_c.py -q
```

Expected: failures identify missing V2 parsing, provenance checks and reviewed-only freeze behavior.

- [ ] **Step 5: Implement compatibility and fail-closed discovery**

Branch only on integer `schema_version`: `1` uses the unchanged V1 parser; `2` uses the V2 parser. Unknown versions fail. The reviewed freeze function discovers only `reviewed_root/*/*/manifest.json`, rejects any other manifest placement and verifies each sample before writing the freeze manifest with exclusive creation. Evaluation continues to load only paths named by the frozen manifest, so deleting the original quarantine/review roots after promotion does not affect replay.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_replay.py tests\unit\diagnostics\test_stage_c_gate.py tests\unit\scripts\test_freeze_human_ai_stage_c.py tests\unit\scripts\test_evaluate_human_ai_stage_c.py -q
git add src\xiangqi_agent\diagnostics\stage_c_replay.py src\xiangqi_agent\diagnostics\stage_c_gate.py scripts\freeze_human_ai_stage_c.py scripts\evaluate_human_ai_stage_c.py tests\unit\diagnostics tests\unit\scripts
git commit -m "feat: freeze reviewed Stage C evidence"
```

---

### Task 8: Non-technical Qt Review Card

**Files:**
- Create: `src/xiangqi_agent/ui/stage_c_review_panel.py`
- Modify: `src/xiangqi_agent/ui/capture_panel.py`
- Modify: `src/xiangqi_agent/ui/main_window.py`
- Create: `tests/unit/ui/test_stage_c_review_panel.py`
- Modify: `tests/unit/ui/test_capture_panel.py`
- Modify: `tests/unit/ui/test_main_window.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class ReviewMoveLine:
    uci: str
    chinese: str


@dataclass(frozen=True, slots=True)
class StageCReviewCard:
    event_id: str
    session_id: str
    confirmed_position_id: str
    candidate_lines: tuple[ReviewMoveLine, ...]


class StageCReviewPanel(QWidget):
    review_completed = Signal(str, str)

    def load_event(self, event_dir: Path) -> None: ...
    def invalidate(self, reason: str) -> None: ...
```

- [ ] **Step 1: Write failing UI contract tests**

Using Qt test helpers and temporary stores, assert:

- the panel shows the starting mirror board and at most two Chinese candidate moves;
- no score, threshold, distance, confidence, FEN, raw JSON or window title appears in any visible label;
- buttons are exactly “走法正确”“实际走法不同”“这是干扰画面”“无法确定，丢弃”;
- correction mode lists only legal first moves and, after selection, legal second moves;
- rejection mode lists exactly the six frozen scenarios;
- submitting creates one immutable review and invokes promotion only once;
- `needs_review` displays a stable plain-language reason and leaves the event quarantined;
- capture restart, session change or starting `position_id` change invalidates the old card;
- fake engine and coach services receive zero calls.

- [ ] **Step 2: Verify RED**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\unit\ui\test_stage_c_review_panel.py tests\unit\ui\test_capture_panel.py tests\unit\ui\test_main_window.py -q
```

Expected: collection fails because the review panel does not exist.

- [ ] **Step 3: Implement explicit evidence mode and review card**

Add a capture-panel checkbox labelled “帮助改进识别（本地保存小裁片）”, default off. Only this mode enables occupancy-gated transition capture and the quarantine recorder. A terminal event pauses further Stage C event recording, emits the anonymous event path to `StageCReviewPanel`, and waits for review/discard before allowing another event.

Map candidates to `ReviewMoveLine` before giving them to the widget so scores never enter the UI model. Submit drafts through `StageCReviewService`, which performs the same rule and hash checks as the CLI; call `StageCPromotionService` only after a valid local selection. Keep regular mirror-board synchronization usable when evidence mode is off.

- [ ] **Step 4: Test close and stale-event lifecycle**

Close the main window during pending review, call `close()` twice, switch sync modes and emit a late terminal update. Assert no half-written directory, background thread, late promotion or stale card remains. The application must never perform a chess-window control action.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\unit\ui\test_stage_c_review_panel.py tests\unit\ui\test_capture_panel.py tests\unit\ui\test_main_window.py tests\unit\sync\test_live_session.py -q
git add src\xiangqi_agent\ui\stage_c_review_panel.py src\xiangqi_agent\ui\capture_panel.py src\xiangqi_agent\ui\main_window.py tests\unit\ui
git commit -m "feat: add local Stage C review card"
```

---

### Task 9: End-to-end Verification, Status and Safe Push

**Files:**
- Create: `tests/integration/diagnostics/test_stage_c_review_flow.py`
- Create: `docs/status/stage-c-quarantine-review.md`
- Modify only after a failing regression test: production files from Tasks 1–8

- [ ] **Step 1: Write the end-to-end fake-frame test**

Drive real services through:

```text
FakeFrameSource
  -> LiveSyncSession terminal evidence
  -> QuarantineEventRecorder
  -> StageCReviewStore
  -> StageCPromotionService
  -> HumanAiStageCSampleLoader
  -> HumanAiStageCReplayer
```

Cover two full flows:

1. observer candidate confirmed and promoted;
2. observer safely rejected a real legal two-ply event, user corrected it, V2 promotes and replay reports `missed_valid=True`.

Delete the source quarantine and review roots after promotion, then prove the V2 sample still loads and replays from its self-contained sidecars. Tamper with each sidecar in separate cases and prove failure.

- [ ] **Step 2: Run focused diagnostics and UI suites**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\unit\vision\test_occupancy.py tests\unit\sync tests\unit\diagnostics tests\unit\scripts tests\unit\ui\test_stage_c_review_panel.py tests\integration\diagnostics\test_stage_c_review_flow.py -q
```

- [ ] **Step 3: Run the complete automatic quality gate once**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src
git diff --check
```

Record exact pass/fail counts and elapsed time. Do not describe a partial or skipped command as passed.

- [ ] **Step 4: Run privacy and tracked-asset checks**

```powershell
git ls-files | Select-String -Pattern '(?i)(^|/)(\.env($|\.)|\.local/|.*\.(png|jpg|jpeg|db|sqlite|onnx|exe|dll)$)'
rg -n --hidden -g '!.git/**' -g '!.local/**' 'sk-[A-Za-z0-9_-]{20,}' .
git status --short
```

Expected: the first two commands return no findings. `git status --short` may show only planned tracked work plus the preserved user-owned untracked `2026-08-28-remaining-execution-plan.md`; never add that file.

- [ ] **Step 5: Perform one user-controlled real smoke only after automation passes**

Ask the user to keep one visible, unobstructed human-vs-AI board at the confirmed FEN and to perform all moves themselves. Establish an occupancy-matching baseline, capture at most one terminal event, show the review card, let the user confirm/correct/discard, then attempt promotion. Do not control the PC and do not save a full screenshot. A safe rejection or `needs_review` is recorded truthfully rather than called a pass.

- [ ] **Step 6: Write the status report**

`docs/status/stage-c-quarantine-review.md` must contain:

- commit range and date;
- occupancy algorithm/version and real limitations;
- automatic command outputs;
- real smoke outcome without private visual data;
- quarantine/reviewed sample counts by state;
- proof that V2 survives source cleanup;
- direct-confirmation versus correction counts;
- unresolved risks;
- explicit statement that engine/coach remains locked until the existing 30/30 Stage C gate passes.

- [ ] **Step 7: Commit verified evidence**

```powershell
git add tests\integration\diagnostics\test_stage_c_review_flow.py docs\status\stage-c-quarantine-review.md
git commit -m "test: verify reviewed Stage C evidence flow"
git status --short
```

If implementation changes were needed after a failing test, commit each fix with its test before this documentation commit.

- [ ] **Step 8: Push without force and without a tag**

```powershell
git fetch origin
git merge-base --is-ancestor origin/develop develop
git push origin develop
```

Stop if the ancestry check fails or authentication fails. Preserve local commits; do not force-push and do not create a milestone tag.

---

## Post-plan Continuation

After all nine tasks pass:

1. Collect 30 independent valid human-vs-AI events and 30 rejection events through the reviewed V2 path, with the user controlling every board action.
2. Freeze the reviewed root once and run the unchanged hard gate: zero false accepts, at least 80% valid coverage, all six rejection scenarios and P95 at most 500 ms.
3. If the gate fails, tune only on a separate development set, create a new feature/profile version and collect a new blind set; never tune against the frozen set.
4. Only after the frozen Stage C report passes, create the next plan for final-confirmed-position Pikafish analysis and DeepSeek explanation, retaining the global kill switch and evidence-only prompts.
5. SQLite review history, 30-minute lifecycle, PyInstaller packaging and ten-game product acceptance remain later release work.
