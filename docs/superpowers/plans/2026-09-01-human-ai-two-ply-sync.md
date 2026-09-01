# Human-vs-AI Two-Ply Atomic Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在显式人机练习模式中，安全识别并原子提交“用户一步 + 人机立即应手”的唯一合法双步视觉变化，同时保持默认严格单步行为不变。

**Architecture:** 保留 `LegalMoveDiffObserver` 作为第一道严格单步门；新增独立的 `LegalTwoPlyDiffObserver`，只在显式人机模式且单步变化无法解释时枚举两步合法链并比较提交前帧与最终稳定帧。`StableMoveTracker` 是唯一决定是否原子提交的状态机，`LiveSyncSession`、UI、Pikafish 和 DeepSeek 只接收完整确认后的 `moves` 与最终 `BoardState`。

**Tech Stack:** Python 3.12、NumPy、OpenCV、PySide6、pytest、Ruff、mypy

**Spec:** `docs/superpowers/specs/2026-09-01-human-ai-two-ply-sync-design.md`

## Global Constraints

- 默认模式必须是 `SyncMode.STRICT_SINGLE`，不得从窗口标题或时间间隔自动推断人机模式。
- 只有 `SyncMode.HUMAN_VS_AI` 可以尝试双步链，且最多恰好 2 步。
- 严格单步观察器、阈值和既有阶段 C 指标保持不变。
- 只有唯一高置信候选可以接受；0 个或多个候选均暂停。
- 双步全成或全不成，不得发布第一步中间状态。
- Pikafish 和 DeepSeek 只接收原子提交后的最终 `position_id`。
- 产品不点击、不自动走棋、不注入、不读取微信进程内存。
- 默认不保存完整截图；诊断数据必须显式开启并位于 Git 忽略的 `.local/`。
- 不修改或提交用户未跟踪的 `docs/superpowers/plans/2026-08-28-remaining-execution-plan.md`。

---

## File Structure

- Create `src/xiangqi_agent/sync/mode.py`: 定义显式同步模式，不承载 UI 文案。
- Modify `src/xiangqi_agent/sync/evidence.py`: 定义双步候选、证据和提议的不可变契约。
- Modify `src/xiangqi_agent/sync/committer.py`: 提供同一棋规边界下的纯投影和双步原子提交。
- Create `src/xiangqi_agent/sync/sequence_observer.py`: 枚举、评分并拒绝不唯一的两步合法链。
- Modify `src/xiangqi_agent/sync/tracker.py`: 严格单步优先，受限双步回退，输出规范 `moves`。
- Modify `src/xiangqi_agent/sync/live_session.py`: 传播模式、走法序列和前后位置编号。
- Modify `src/xiangqi_agent/sync/__init__.py`: 导出新公共契约。
- Modify `src/xiangqi_agent/ui/capture_panel.py`: 增加显式模式选择并在连接时锁定。
- Modify `src/xiangqi_agent/ui/main_window.py`: 校验并显示 1 或 2 步，只分析最终确认局面。
- Create `src/xiangqi_agent/diagnostics/transition_samples.py`: 保存隐私受限的双步 V2 诊断样本。
- Create `tests/unit/sync/test_sequence_observer.py`: 双步视觉评分的真实合成帧测试。
- Modify `tests/unit/sync/test_committer.py`: 投影、双步原子提交和输入验证。
- Modify `tests/unit/sync/test_tracker.py`: 模式门、原子性、三步/歧义和恢复路径。
- Modify `tests/unit/sync/test_live_session.py`: 连续两单步与合并双步事件流。
- Modify `tests/unit/ui/test_capture_panel.py`: 模式显式选择与连接期间锁定。
- Modify `tests/unit/ui/test_main_window.py`: 双步记谱、最终局面和分析代次。
- Create `tests/unit/diagnostics/test_transition_samples.py`: V2 清单、裁片、隐私与清理。
- Create `docs/status/human-ai-two-ply-sync.md`: 自动门、真实冒烟和遗留风险证据。

---

### Task 1: Explicit mode and immutable sequence contracts

**Files:**
- Create: `src/xiangqi_agent/sync/mode.py`
- Modify: `src/xiangqi_agent/sync/evidence.py`
- Modify: `src/xiangqi_agent/sync/__init__.py`
- Test: `tests/unit/sync/test_tracker.py`

**Interfaces:**
- Produces: `SyncMode.STRICT_SINGLE`, `SyncMode.HUMAN_VS_AI`
- Produces: `SequenceCandidateEvidence`, `MoveSequenceEvidence`, `MoveSequenceProposal`
- Produces: accepted sequence invariant: exactly two moves; rejected sequence invariant: no moves

- [ ] **Step 1: Write failing contract tests**

Add tests that instantiate the intended public types and assert invalid accepted/rejected shapes raise `ValueError`:

```python
def test_sequence_proposal_requires_exactly_two_moves_when_accepted() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        MoveSequenceProposal(
            status=ObservationStatus.ACCEPTED,
            moves=(),
            evidence_score=1.0,
            evidence=MoveSequenceEvidence((), (), (), "two-ply-v1"),
        )


def test_sequence_proposal_rejects_moves_on_ambiguous_result() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    reply = _move(apply_move(board, move), "h7e7")
    with pytest.raises(ValueError, match="must not expose"):
        MoveSequenceProposal(
            status=ObservationStatus.AMBIGUOUS,
            moves=(move, reply),
            evidence_score=0.0,
            evidence=MoveSequenceEvidence((), (), ("candidate_margin",), "two-ply-v1"),
        )
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/sync/test_tracker.py -k sequence_proposal -q`

Expected: collection fails because the sequence types and mode do not exist.

- [ ] **Step 3: Implement the immutable contracts**

Use these exact public shapes:

```python
class SyncMode(StrEnum):
    STRICT_SINGLE = "strict_single"
    HUMAN_VS_AI = "human_vs_ai"


@dataclass(frozen=True, slots=True)
class SequenceCandidateEvidence:
    moves: tuple[Move, Move]
    changed_points: tuple[int, ...]
    expected_change_floor: float
    unexpected_difference: float
    maximum_template_distance: float
    minimum_template_margin: float
    minimum_template_confidence: float
    score: float
    final_position_id: str


@dataclass(frozen=True, slots=True)
class MoveSequenceEvidence:
    candidates: tuple[SequenceCandidateEvidence, ...]
    local_differences: tuple[float, ...]
    rejection_reasons: tuple[str, ...]
    feature_version: str


@dataclass(frozen=True, slots=True)
class MoveSequenceProposal:
    status: ObservationStatus
    moves: tuple[Move, ...]
    evidence_score: float
    evidence: MoveSequenceEvidence

    def __post_init__(self) -> None:
        if self.status is ObservationStatus.ACCEPTED and len(self.moves) != 2:
            raise ValueError("accepted sequence must contain exactly two moves")
        if self.status is not ObservationStatus.ACCEPTED and self.moves:
            raise ValueError("rejected sequence must not expose moves")
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/unit/sync/test_tracker.py -k sequence_proposal -q`

Expected: both tests pass.

- [ ] **Step 5: Run focused lint and commit**

Run: `ruff check src/xiangqi_agent/sync tests/unit/sync/test_tracker.py`

Commit: `git commit -m "feat: add explicit human-ai sync contracts"`

---

### Task 2: One rule boundary for projection and atomic commit

**Files:**
- Modify: `src/xiangqi_agent/sync/committer.py`
- Modify: `tests/unit/sync/test_committer.py`

**Interfaces:**
- Consumes: `BoardState`, `Move`
- Produces: `StateCommitter.project(board, moves) -> BoardState`
- Produces: `StateCommitter.commit_sequence(board, moves) -> BoardState`

- [ ] **Step 1: Write failing projection and atomicity tests**

```python
def test_rule_committer_projects_two_legal_plies_without_mutating_input() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    after_first = apply_move(board, first)
    second = _move(after_first, "h7e7")

    projected = RuleStateCommitter().project(board, (first, second))

    assert projected == apply_move(after_first, second)
    assert board == parse_fen(START)


def test_rule_committer_rejects_whole_sequence_when_second_move_is_illegal() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    illegal_second = Move("a0a9", 81, 0)

    with pytest.raises(ValueError, match="legal move"):
        RuleStateCommitter().commit_sequence(board, (first, illegal_second))

    assert board == parse_fen(START)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/sync/test_committer.py -q`

Expected: fails because `project` and `commit_sequence` are absent.

- [ ] **Step 3: Add the shared transition boundary**

```python
class StateCommitter(Protocol):
    def commit(self, board: BoardState, move: Move) -> BoardState: ...
    def project(self, board: BoardState, moves: tuple[Move, ...]) -> BoardState: ...
    def commit_sequence(self, board: BoardState, moves: tuple[Move, Move]) -> BoardState: ...


class RuleStateCommitter:
    def project(self, board: BoardState, moves: tuple[Move, ...]) -> BoardState:
        if not moves or len(moves) > 2:
            raise ValueError("projection must contain one or two moves")
        projected = board
        for move in moves:
            projected = apply_move(projected, move)
        return projected

    def commit_sequence(self, board: BoardState, moves: tuple[Move, Move]) -> BoardState:
        if len(moves) != 2:
            raise ValueError("atomic sequence must contain exactly two moves")
        return self.project(board, moves)
```

- [ ] **Step 4: Verify GREEN and regression**

Run: `python -m pytest tests/unit/sync/test_committer.py tests/unit/domain/test_rules.py -q`

Expected: all tests pass and the input board remains immutable.

- [ ] **Step 5: Commit**

Commit: `git commit -m "feat: add atomic two-ply rule projection"`

---

### Task 3: Legal two-ply visual observer

**Files:**
- Create: `src/xiangqi_agent/sync/sequence_observer.py`
- Modify: `src/xiangqi_agent/sync/__init__.py`
- Create: `tests/unit/sync/test_sequence_observer.py`

**Interfaces:**
- Consumes: confirmed `BoardState`, before/after BGRA frames, `BoardGeometry`, `StateCommitter`
- Produces: `MoveSequenceObserver.observe(...) -> MoveSequenceProposal`
- Produces: `LegalTwoPlyDiffObserver`, feature version `two-ply-template-v1`

- [ ] **Step 1: Write a failing unique-chain test using real rendered frames**

Create a literal 90-intersection renderer like existing tracker tests, then:

```python
def test_two_ply_observer_accepts_the_only_legal_chain_matching_final_frame() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board, _render(board), _render(final), _geometry()
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    assert proposal.evidence.candidates[0].final_position_id == final.position_id
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/sync/test_sequence_observer.py::test_two_ply_observer_accepts_the_only_legal_chain_matching_final_frame -q`

Expected: collection fails because `LegalTwoPlyDiffObserver` does not exist.

- [ ] **Step 3: Implement candidate enumeration and final-position scoring**

Implement a deterministic observer that:

1. Crops all 90 intersections once for each frame.
2. Builds `PieceTemplateBank.from_position(board, geometry, before)`.
3. Enumerates `legal_moves(board)`, projects each first move, enumerates the reply, and projects the final board.
4. Derives `changed_points` from literal before/final piece tuples; never assumes four points.
5. Scores only those points against exact final symbols and treats strong differences elsewhere as unexpected.
6. Sorts with `(-score, first.uci, second.uci)` for deterministic replay.
7. Requires absolute score, template distance, template margin, confidence and best-vs-second margin gates.

The observer constructor must validate finite positive thresholds and expose no runtime threshold relaxation.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/unit/sync/test_sequence_observer.py::test_two_ply_observer_accepts_the_only_legal_chain_matching_final_frame -q`

Expected: passes with exactly the expected move order.

- [ ] **Step 5: Add failing safety cases**

Add separate tests for:

- multiple chains with indistinguishable endpoint evidence return `AMBIGUOUS` and no moves;
- a final frame representing three plies is rejected;
- a shared endpoint and a capture produce 2 or 3 changed points without an index error;
- missing required template class returns `AMBIGUOUS` with `template_unavailable`;
- no visual change returns `NO_CHANGE`;
- a strong unrelated change returns `AMBIGUOUS` with `outside_change`.

For the ambiguity test, inject a tiny deterministic `PieceTemplateBank` double only at the template boundary; assert the real observer result, not calls on the double.

- [ ] **Step 6: Implement the minimum rejection logic and verify**

Run: `python -m pytest tests/unit/sync/test_sequence_observer.py -q`

Expected: all sequence observer tests pass.

- [ ] **Step 7: Focused quality gate and commit**

Run: `ruff check src/xiangqi_agent/sync/sequence_observer.py tests/unit/sync/test_sequence_observer.py`

Run: `mypy src/xiangqi_agent/sync`

Commit: `git commit -m "feat: score unique legal two-ply transitions"`

---

### Task 4: Tracker fallback and all-or-nothing state transition

**Files:**
- Modify: `src/xiangqi_agent/sync/tracker.py`
- Modify: `tests/unit/sync/test_tracker.py`

**Interfaces:**
- Consumes: `SyncMode`, strict `MoveObserver`, optional `MoveSequenceObserver`
- Produces: `TrackingUpdate.moves: tuple[Move, ...]`
- Preserves: `TrackingUpdate.move` compatibility property returning a value only for one move

- [ ] **Step 1: Write failing strict-mode and human-mode tests**

```python
def test_strict_tracker_never_invokes_two_ply_observer() -> None:
    sequence = RecordingSequenceObserver(_accepted_sequence(board))
    tracker = _tracker(board, mode=SyncMode.STRICT_SINGLE, sequence_observer=sequence)
    result = _settle(tracker, _render(final))
    assert result.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert sequence.calls == 0
    assert result.board == board


def test_human_ai_tracker_atomically_accepts_unique_two_ply_fallback() -> None:
    tracker = _tracker(board, mode=SyncMode.HUMAN_VS_AI)
    result = _settle(tracker, _render(final))
    assert result.status is TrackingStatus.ACCEPTED
    assert result.moves == (first, second)
    assert result.move is None
    assert result.board == final
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/sync/test_tracker.py -k 'strict_tracker_never or atomically_accepts' -q`

Expected: fails because tracker has no mode, sequence observer or `moves` field.

- [ ] **Step 3: Implement strict-first fallback**

Change `TrackingUpdate` so `moves` is canonical and add:

```python
@property
def move(self) -> Move | None:
    return self.moves[0] if len(self.moves) == 1 else None
```

In `push()`:

- commit an accepted strict proposal as `(move,)` exactly as today;
- only call the sequence observer when mode is `HUMAN_VS_AI` and strict rejection reasons contain an eligible merged-change reason;
- call `commit_sequence()` only after the sequence proposal is accepted with two moves;
- assign `_board` and `_confirmed_frame` only after the whole call succeeds;
- otherwise call `_pause()` and preserve the confirmed state.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/unit/sync/test_tracker.py -k 'strict_tracker_never or atomically_accepts' -q`

Expected: both pass.

- [ ] **Step 5: Add failing atomicity and recovery tests**

Cover these mutations independently:

- `commit_sequence()` raises on the second move: board and confirmed frame remain unchanged;
- accepted proposal contains wrong final chain: committer rejects and tracker pauses;
- ambiguous sequence never publishes either move;
- a normal stable platform still yields two separate one-move updates in human mode;
- resize/context invalidation prevents sequence fallback;
- after manual recovery, stale pre-recovery evidence cannot commit.

- [ ] **Step 6: Implement and run the complete tracker suite**

Run: `python -m pytest tests/unit/sync/test_tracker.py tests/unit/sync/test_committer.py tests/unit/sync/test_sequence_observer.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

Commit: `git commit -m "feat: commit merged human-ai replies atomically"`

---

### Task 5: Live session event migration and stale-result safety

**Files:**
- Modify: `src/xiangqi_agent/sync/live_session.py`
- Modify: `tests/unit/sync/test_live_session.py`

**Interfaces:**
- Consumes: `sync_mode: SyncMode = SyncMode.STRICT_SINGLE`
- Produces: `LiveSyncUpdate.moves`, `before_position_id`, `after_position_id`, `sync_mode`
- Preserves: `LiveSyncUpdate.move` compatibility property for single-move events only

- [ ] **Step 1: Write a failing default-mode compatibility test**

Construct `LiveSyncSession` without a mode, feed a merged two-ply final frame, and assert it pauses with no board change. Then construct it with `SyncMode.HUMAN_VS_AI` and assert one accepted event contains exactly two ordered moves and the final position IDs.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/sync/test_live_session.py -k 'merged_two_ply or defaults_to_strict' -q`

Expected: fails because the session cannot accept a mode or emit `moves`.

- [ ] **Step 3: Propagate the canonical event**

Update `LiveSyncSession` to:

- store `sync_mode` at construction;
- instantiate `LegalTwoPlyDiffObserver` only for human mode;
- pass mode and observer into `StableMoveTracker`;
- emit `before_position_id` from the pre-push board and `after_position_id` from `update.board`;
- use `"unique legal move accepted"` for one move and `"unique legal two-ply sequence accepted atomically"` for two;
- never emit an accepted event if the session generation or recovery state has changed.

- [ ] **Step 4: Verify focused live-session behavior**

Run: `python -m pytest tests/unit/sync/test_live_session.py -q`

Expected: separate single moves, merged double moves, resize, close and recovery tests all pass.

- [ ] **Step 5: Commit**

Commit: `git commit -m "feat: publish confirmed move sequences from live sync"`

---

### Task 6: Explicit UI mode and final-position-only analysis

**Files:**
- Modify: `src/xiangqi_agent/ui/capture_panel.py`
- Modify: `src/xiangqi_agent/ui/main_window.py`
- Modify: `tests/unit/ui/test_capture_panel.py`
- Modify: `tests/unit/ui/test_main_window.py`

**Interfaces:**
- Produces: mode selector labels `严格单步` and `人机练习（可同步连续应手）`
- Consumes: `LiveSyncUpdate.moves`
- Produces: ordered Chinese notation and one analysis submission for final board only

- [ ] **Step 1: Write failing mode-selector tests**

Assert the panel defaults to strict mode, passes the selected enum to the created session, disables the selector while connected, and re-enables it on close. Use a session factory injection rather than patching Qt internals.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/ui/test_capture_panel.py -k sync_mode -q`

Expected: fails because the selector and session factory boundary are absent.

- [ ] **Step 3: Add the explicit selector**

Add `mode_combo` next to the orientation selector. Its stored data must be enum values, and `_toggle_capture()` must construct the session with `SyncMode(str(mode_combo.currentData()))`. Mode changes while connected are impossible because the control is disabled.

- [ ] **Step 4: Write failing double-notation and engine-generation tests**

Feed `MainWindow._on_sync_update()` a two-move event whose before and after IDs are exact. Assert:

- the displayed phase contains both hand-derived Chinese notations in order;
- `BoardWidget` receives only the final board and highlights the second move;
- `AnalysisService.submit()` is called once with the final board when the mode-specific gate is enabled;
- an incorrect `before_position_id`, `after_position_id`, move order or final board is ignored;
- no analysis is submitted for an intermediate board.

- [ ] **Step 5: Verify RED**

Run: `python -m pytest tests/unit/ui/test_main_window.py -k 'two_ply or final_position_only' -q`

Expected: fails because main window only accepts `update.move`.

- [ ] **Step 6: Implement ordered replay validation**

Starting from the current `_board`, use `RuleStateCommitter.project(before, update.moves)` to derive each notation and verify the final `position_id` matches both the event and `update.board`. Adopt only the final board; use `last_move=update.moves[-1]` and submit analysis once.

- [ ] **Step 7: Run UI regression**

Run: `python -m pytest tests/unit/ui/test_capture_panel.py tests/unit/ui/test_main_window.py -q`

Expected: all UI tests pass without opening a real window.

- [ ] **Step 8: Commit**

Commit: `git commit -m "feat: expose safe human-ai sync mode in UI"`

---

### Task 7: Privacy-safe TransitionSampleV2

**Files:**
- Create: `src/xiangqi_agent/diagnostics/transition_samples.py`
- Create: `tests/unit/diagnostics/test_transition_samples.py`
- Modify: `src/xiangqi_agent/diagnostics/__init__.py`

**Interfaces:**
- Produces: `TransitionPointCrops`, `TransitionSampleV2`, `TransitionSampleRecorder`
- Persists: exactly 2–4 changed point pairs, manifest hashes and no full frame
- Reuses: default disabled, byte quota and UTC retention semantics from endpoint diagnostics

- [ ] **Step 1: Write failing schema and disabled-default tests**

Use a temporary directory and assert:

- `moves_uci` must contain exactly two valid UCI strings;
- `changed_points` must contain 2–4 unique indices in stable ascending order;
- each point has exactly one 48×48 BGRA before/after crop;
- recording while disabled raises `DiagnosticsDisabledError` and creates no path.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/diagnostics/test_transition_samples.py -q`

Expected: collection fails because V2 types do not exist.

- [ ] **Step 3: Implement V2 atomic persistence**

Write each sample to a temporary sibling directory, hash all PNG bytes, write UTF-8 JSON, then rename atomically. Filenames must be deterministic: `point-00-before.png`, `point-00-after.png`, and so on in changed-point order. The manifest may contain FEN and anonymous position IDs, but no title, nickname, avatar, account, API key or full-frame path.

- [ ] **Step 4: Add quota, corruption and retention tests**

Assert successful recording purges samples older than the configured UTC interval, a rejected invalid record does not purge existing samples, byte quota is enforced before rename, and duplicate sample IDs do not overwrite data.

- [ ] **Step 5: Verify GREEN and commit**

Run: `python -m pytest tests/unit/diagnostics/test_transition_samples.py tests/unit/diagnostics/test_endpoint_samples.py -q`

Commit: `git commit -m "feat: record privacy-safe two-ply diagnostics"`

---

### Task 8: Full verification, one real smoke, and status evidence

**Files:**
- Create: `docs/status/human-ai-two-ply-sync.md`
- Modify only if evidence requires a fix: files from Tasks 1–7 with a new failing regression test first

**Interfaces:**
- Produces: reproducible automatic gate evidence and one real visible-window result
- Does not produce: a Stage C pass claim before 30 valid and 30 rejected frozen samples exist

- [ ] **Step 1: Run focused and full automatic gates**

Run in this order:

```powershell
python -m pytest tests/unit/sync tests/unit/ui/test_capture_panel.py tests/unit/ui/test_main_window.py tests/unit/diagnostics/test_transition_samples.py -q
python -m pytest -q
ruff check .
mypy src
git diff --check
```

Expected: every command exits 0 with no warnings attributable to the new work.

- [ ] **Step 2: Run privacy and tracked-file scans**

Confirm no tracked file contains an API key pattern, `.env`, database, full screenshot, diagnostic PNG, model binary, engine binary or `.local/` content. Confirm the user-owned untracked 2026-08-28 plan remains untracked and unstaged.

- [ ] **Step 3: Perform one real visible-window smoke**

Preconditions: exactly one visible `天天象棋` top-level window, human-vs-AI practice board, fixed theme, no obstruction. Use the existing manual normalized quad or recalibrate after resize. Start the app in `HUMAN_VS_AI`, make one user move and allow the AI reply. Record only:

- capture backend and frame size;
- whether the event arrived as two singles or one atomic double;
- ordered UCI moves and before/final position IDs;
- decision latency and final mirror-board equality;
- whether any ambiguous event correctly paused.

Do not persist a full screenshot. Computer control, if used under the user's temporary authorization, is limited to the named test move and ends immediately afterward.

- [ ] **Step 4: Write the status report**

Document automatic command outputs, the real smoke result, backend limitations, and the explicit statement: `Human-vs-AI Stage C remains locked until 30 merged valid events and 30 rejection events are frozen with zero false accepts.`

- [ ] **Step 5: Final review and commit**

Run the full automatic gates again after any smoke-derived fix. Review the complete diff against all 12 sections of the spec. Commit only intended files:

`git commit -m "test: validate human-ai two-ply sync vertical"`

- [ ] **Step 6: Push the verified checkpoint**

Push `develop` without force. If the remote advanced unexpectedly, stop and inspect rather than overwriting it.

---

## Post-plan continuation toward the full project

This plan completes the currently blocking real-time synchronization vertical. After it passes, continue the active whole-project goal in this order:

1. Freeze and execute the separate human-vs-AI Stage C dataset gate; keep live engine/coach locked until zero false accepts is proven.
2. Finish replay persistence and user-facing review navigation without weakening privacy defaults.
3. Run capture interruption, 30-minute lifecycle, Pikafish residual-process, DeepSeek fallback and stale-generation acceptance tests.
4. Build and verify the Windows directory package with locked external asset downloads and license records.
5. Complete the approved ten-game human-vs-AI/endgame/review acceptance set, then update release evidence and merge only after every original product requirement is proven.
