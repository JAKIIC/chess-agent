# Human-vs-AI Stage C Frozen Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可审计、可重复、失败关闭的人机双步冻结盲测工具；只有至少 30 个独立真实双步事件和 30 个拒绝事件达到零错误接受、正确最终 FEN、覆盖率和延迟门，实时 Pikafish/DeepSeek 才允许消费该模式的最终确认局面。

**Architecture:** 保留 `TransitionSampleV2` 作为已完成的双步小裁片诊断格式；新增独立 `HumanAiStageCSampleV1`，把人工真值、观察器证据、最多四个交点小裁片和决策延迟记录为版本化事件。`SequenceDecisionGate` 从观察器中提取唯一的门限决策逻辑，现场观察和离线回放共用它；冻结清单固定样本哈希、特征版本和门限配置，聚合器只读取清单列出的盲测样本并以非零退出码阻止不合格报告。

**Tech Stack:** Python 3.12、NumPy、OpenCV、Pydantic-free dataclasses、pytest、现有 `LiveSyncSession`/`RuleStateCommitter`/`TransitionPointCrops`。

**Spec:** `docs/superpowers/specs/2026-09-01-human-ai-two-ply-sync-design.md`

## Global Constraints

- 仅在用户显式选择 `SyncMode.HUMAN_VS_AI` 时评估双步链；严格单步模式保持不变。
- 每个候选恰好包含两步，顺序为用户一步后人机一步；三步或更多必须拒绝。
- 现场稳定采样为 2 FPS，变化期间最高 20 FPS；画面稳定后才决策。
- 诊断默认关闭；开启后只保存 1–4 个 48×48 BGRA 交点前后小裁片和非隐私数值证据。
- 不保存完整窗口、完整棋盘图、窗口标题、头像、昵称、API Key 或 DeepSeek 请求。
- 样本只进入 Git 忽略的 `.local/`，默认 7 天 UTC 保留并执行字节配额。
- 盲测前冻结特征版本、门限配置、样本相对路径和每份样本清单 SHA-256；冻结后不得原地改写。
- 有效集至少 30 个真实双步事件，且来自至少 30 个不同会话；拒绝集至少 30 个事件。
- 拒绝集必须覆盖多候选、点选/悬停高亮、连续动画、遮挡、resize 和三步合并六类场景。
- 自动接受的双步链、顺序和最终 FEN 必须 100% 正确；错误接受数必须为 0。
- 有效事件自动接受覆盖率必须不低于 80%；最终稳定至决策 P95 必须不超过 500 ms。
- 任何样本损坏、版本混用、标签矛盾、非法 UCI、哈希不符或样本不足都必须让门失败。
- 所有生产改动严格执行 RED → GREEN → REFACTOR；不得根据盲测结果调门限。

---

## Planned File Structure

```text
src/xiangqi_agent/
├─ sync/
│  ├─ sequence_gate.py
│  ├─ sequence_observer.py
│  ├─ tracker.py
│  └─ live_session.py
└─ diagnostics/
   ├─ stage_c_samples.py
   ├─ stage_c_replay.py
   └─ stage_c_gate.py
scripts/
├─ collect_human_ai_stage_c.py
├─ freeze_human_ai_stage_c.py
└─ evaluate_human_ai_stage_c.py
tests/unit/
├─ sync/test_sequence_gate.py
├─ diagnostics/test_stage_c_samples.py
├─ diagnostics/test_stage_c_replay.py
├─ diagnostics/test_stage_c_gate.py
└─ scripts/
   ├─ test_collect_human_ai_stage_c.py
   ├─ test_freeze_human_ai_stage_c.py
   └─ test_evaluate_human_ai_stage_c.py
docs/status/human-ai-stage-c.md
```

---

### Task 1: Shared, Versioned Two-ply Decision Gate

**Files:**
- Create: `src/xiangqi_agent/sync/sequence_gate.py`
- Modify: `src/xiangqi_agent/sync/sequence_observer.py`
- Create: `tests/unit/sync/test_sequence_gate.py`
- Modify: `tests/unit/sync/test_sequence_observer.py`

**Interfaces:**
- Produces: `SequenceThresholdProfile`
- Produces: `SequenceDecision`
- Produces: `SequenceDecisionGate.evaluate(candidates, *, template_unavailable=False) -> SequenceDecision`
- Consumes: ranked `tuple[SequenceCandidateEvidence, ...]`

- [ ] **Step 1: Write failing gate tests**

Create literal candidates and assert the real gate:

```python
def test_gate_accepts_only_the_unique_candidate_passing_every_hard_threshold() -> None:
    decision = SequenceDecisionGate(PROFILE).evaluate((GOOD, DISTANT_RUNNER_UP))
    assert decision.accepted
    assert decision.candidate == GOOD
    assert decision.rejection_reasons == ()


def test_gate_rejects_equal_scoring_candidates_without_exposing_moves() -> None:
    decision = SequenceDecisionGate(PROFILE).evaluate((GOOD, replace(GOOD, moves=OTHER)))
    assert not decision.accepted
    assert decision.candidate is None
    assert decision.rejection_reasons == ("candidate_margin",)
```

Add one literal test per existing hard condition: expected change, outside change, score, margin, template distance, template margin and template evidence score. Add zero-candidate cases for `no_legal_candidates` and `template_unavailable`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync\test_sequence_gate.py -q
```

Expected: collection fails because `xiangqi_agent.sync.sequence_gate` does not exist.

- [ ] **Step 3: Implement the minimal shared gate**

`SequenceThresholdProfile` contains the seven existing thresholds plus `profile_version`. `SequenceDecisionGate` assumes candidates are already deterministically ranked and returns either one candidate or no candidate; rejected decisions never expose moves.

`LegalTwoPlyDiffObserver` continues to compute local differences, templates and candidates, but delegates all threshold decisions to this gate. Its public constructor remains source-compatible and builds profile `human-ai-two-ply-v1` from the supplied arguments.

- [ ] **Step 4: Verify behavior preservation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync\test_sequence_gate.py tests\unit\sync\test_sequence_observer.py tests\unit\sync\test_tracker.py -q
```

Expected: all tests pass; existing candidate order, accepted moves, rejection reasons and feature version remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src\xiangqi_agent\sync\sequence_gate.py src\xiangqi_agent\sync\sequence_observer.py tests\unit\sync
git commit -m "refactor: share frozen two-ply decision gate"
```

---

### Task 2: Privacy-safe Stage C Event Contract and Recorder

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_samples.py`
- Modify: `src/xiangqi_agent/diagnostics/__init__.py`
- Create: `tests/unit/diagnostics/test_stage_c_samples.py`

**Interfaces:**
- Produces: `StageCExpectedOutcome`, `StageCScenario`, `StageCCandidateRecord`
- Produces: `HumanAiStageCSampleV1`
- Produces: `HumanAiStageCSampleRecorder.record(sample, crops) -> Path`
- Reuses: `TransitionPointCrops`, `DiagnosticsDisabledError`, `SampleQuotaExceededError`

- [ ] **Step 1: Write failing schema tests**

Assert these literal contracts:

```python
def test_valid_event_requires_exactly_two_ground_truth_moves() -> None:
    with pytest.raises(ValueError, match="two ground-truth moves"):
        sample(expected_outcome=StageCExpectedOutcome.ACCEPT, ground_truth_moves_uci=())


def test_rejection_event_never_claims_an_expected_final_position() -> None:
    with pytest.raises(ValueError, match="rejection event"):
        sample(
            expected_outcome=StageCExpectedOutcome.REJECT,
            expected_final_position_id=FINAL_ID,
        )


def test_recorder_is_disabled_without_explicit_opt_in(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsDisabledError):
        HumanAiStageCSampleRecorder(tmp_path).record(VALID_SAMPLE, CROPS)
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())
```

Also test:

- accepted observations expose exactly two observed moves and an observed final ID;
- rejected observations expose no observed moves and keep `confirmed_position_id`;
- `local_differences` contains exactly 90 finite non-negative values;
- candidate records contain at most the top two ranked legal chains;
- changed points are 1–4 unique indices in ascending order and match crops;
- `decision_latency_ms` is finite and non-negative;
- scenarios are exactly `valid_two_ply`, `multiple_candidates`, `selection_highlight`, `continuous_animation`, `occlusion`, `resize`, `three_ply`;
- sample/session identifiers and timestamps are path-safe and UTC.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_samples.py -q
```

Expected: collection fails because the Stage C sample module does not exist.

- [ ] **Step 3: Implement immutable schema and atomic persistence**

The manifest stores only JSON-safe evidence, anonymous IDs and relative crop names. The recorder encodes one before/after PNG pair per selected point, hashes every PNG, writes a sorted UTF-8 manifest into a temporary sibling directory, then atomically renames it. It rejects unexpected crop count, duplicate IDs and existing destinations before mutating retention state.

- [ ] **Step 4: Add integrity, quota and retention tests**

Test corrupt PNG input, a quota one byte below the required size, duplicate sample ID, invalid record not purging old samples, exact seven-day UTC cutoff and deterministic filenames. Scan manifest keys and string values to prove no title, nickname, avatar, account, key, request or full-frame path is persisted.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_samples.py tests\unit\diagnostics\test_transition_samples.py tests\unit\diagnostics\test_endpoint_samples.py -q
```

Commit:

```powershell
git add src\xiangqi_agent\diagnostics tests\unit\diagnostics
git commit -m "feat: record frozen human-ai gate events"
```

---

### Task 3: Deterministic Stage C Loader and Replayer

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_replay.py`
- Create: `tests/unit/diagnostics/test_stage_c_replay.py`

**Interfaces:**
- Produces: `StageCSampleIntegrityError`
- Produces: `LoadedHumanAiStageCSample`
- Produces: `HumanAiStageCSampleLoader.load(path) -> LoadedHumanAiStageCSample`
- Produces: `HumanAiStageCReplayer.replay(path) -> HumanAiStageCReplayResult`
- Consumes: `SequenceDecisionGate`, `RuleStateCommitter`

- [ ] **Step 1: Write failing integrity and determinism tests**

Persist controlled valid and rejection fixtures, then assert:

```python
def test_same_frozen_event_replays_to_the_same_decision(tmp_path: Path) -> None:
    first = REPLAYER.replay(record_valid_event(tmp_path))
    second = REPLAYER.replay(record_valid_event(tmp_path, sample_id="second"))
    assert first.without_runtime_and_identity() == second.without_runtime_and_identity()


def test_wrong_observed_final_id_is_an_integrity_error(tmp_path: Path) -> None:
    path = record_event_with_tampered_final_id(tmp_path)
    with pytest.raises(StageCSampleIntegrityError, match="final position"):
        REPLAYER.replay(path)
```

Add failures for a changed PNG hash, extra file, malformed UTF-8 JSON, FEN/position mismatch, illegal candidate chain, wrong candidate final ID, profile mismatch and feature-version mismatch.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_replay.py -q
```

Expected: collection fails because the loader/replayer does not exist.

- [ ] **Step 3: Implement loader and rule-grounded replay**

The loader requires exactly the manifest plus declared crop files and validates every hash. It reconstructs candidate `Move` objects only by matching UCI pairs against `RuleStateCommitter.project_two_ply(confirmed_board)`; serialized source/target indices are never trusted. The replayer invokes the shared gate, applies an accepted candidate atomically, and returns the recomputed moves/final ID, ground-truth comparison, rejection reasons and runtime.

- [ ] **Step 4: Add mutation coverage**

Verify that changing any threshold field, candidate score, runner-up score, candidate order, expected outcome, expected move order or final position ID changes or invalidates the replay result. A rejected replay must never return a new FEN or after-position ID.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_replay.py tests\unit\sync\test_sequence_gate.py -q
git add src\xiangqi_agent\diagnostics\stage_c_replay.py tests\unit\diagnostics\test_stage_c_replay.py
git commit -m "feat: replay human-ai gate evidence deterministically"
```

---

### Task 4: Immutable Blind-set Manifest and Release Gate

**Files:**
- Create: `src/xiangqi_agent/diagnostics/stage_c_gate.py`
- Create: `scripts/freeze_human_ai_stage_c.py`
- Create: `scripts/evaluate_human_ai_stage_c.py`
- Create: `tests/unit/diagnostics/test_stage_c_gate.py`
- Create: `tests/unit/scripts/test_freeze_human_ai_stage_c.py`
- Create: `tests/unit/scripts/test_evaluate_human_ai_stage_c.py`

**Interfaces:**
- Produces: `FrozenStageCManifestV1`
- Produces: `HumanAiStageCMetrics`
- Produces: `HumanAiStageCGate.evaluate(manifest_path) -> HumanAiStageCReport`
- CLI contract: exit `0` only for a complete pass, `1` for metric failure, `2` for integrity/configuration failure

- [ ] **Step 1: Write failing aggregation tests**

Use literal replay results to prove:

- 29 valid events fail the count gate;
- 30 valid events sharing one session fail the 30-session gate;
- 29 rejection events fail;
- any missing rejection scenario fails;
- one wrong-sequence accept or one accepted rejection makes `false_accepts == 1` and fails safety;
- 24/30 correct accepts passes the 80% boundary; 23/30 fails usability;
- a 500.0 ms P95 passes and a value greater than 500.0 fails;
- all hard conditions together are required for `release_pass`.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_gate.py tests\unit\scripts\test_freeze_human_ai_stage_c.py tests\unit\scripts\test_evaluate_human_ai_stage_c.py -q
```

Expected: collection fails because manifest, gate and scripts do not exist.

- [ ] **Step 3: Implement freeze-once manifest**

`freeze_human_ai_stage_c.py` discovers only Stage C manifests beneath an explicit root, sorts portable relative paths, records each manifest SHA-256, locks one feature version and the complete seven-field `SequenceThresholdProfile` plus `profile_version`, and writes a new file using exclusive creation. It refuses an existing output, absolute path, parent traversal, duplicate sample ID, duplicate sample path or mixed version. Evaluation reconstructs `SequenceDecisionGate` only from these frozen values; command-line threshold overrides are not accepted.

- [ ] **Step 4: Implement report and fail-closed exit codes**

`evaluate_human_ai_stage_c.py` loads only paths named in the frozen manifest, checks their current manifest hashes, replays each sample and writes sorted JSON atomically. The report contains counts, distinct sessions, per-scenario counts, correct accepts, false accepts, coverage, accepted precision, final-position accuracy, P95 decision latency, P95 replay runtime, feature/profile versions and a reason list. It contains no FEN, UCI sequence, path outside the blind-set root or crop data.

- [ ] **Step 5: Verify scripts through real files**

Run tests with temporary sample trees and invoke each script's `main(argv)` against real files. Do not assert source strings or mocks. Confirm a failing gate still writes an auditable report and returns `1`; corruption returns `2` without claiming metric failure.

- [ ] **Step 6: Commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\diagnostics\test_stage_c_gate.py tests\unit\scripts\test_freeze_human_ai_stage_c.py tests\unit\scripts\test_evaluate_human_ai_stage_c.py -q
git add src\xiangqi_agent\diagnostics\stage_c_gate.py scripts\freeze_human_ai_stage_c.py scripts\evaluate_human_ai_stage_c.py tests\unit
git commit -m "feat: enforce frozen human-ai Stage C gate"
```

---

### Task 5: Explicit Live Collection Hook and One-event Collector

**Files:**
- Create: `src/xiangqi_agent/sync/transition_capture.py`
- Modify: `src/xiangqi_agent/sync/tracker.py`
- Modify: `src/xiangqi_agent/sync/live_session.py`
- Create: `scripts/collect_human_ai_stage_c.py`
- Modify: `tests/unit/sync/test_tracker.py`
- Modify: `tests/unit/sync/test_live_session.py`
- Create: `tests/unit/scripts/test_collect_human_ai_stage_c.py`

**Interfaces:**
- Produces: `TransitionCaptureEvidence`
- Adds optional `capture_transition_evidence: bool = False` to `StableMoveTracker` and `LiveSyncSession`
- Adds optional `transition_evidence` to `TrackingUpdate` and `LiveSyncUpdate`
- CLI consumes: `--fen`, `--quad`, exactly one of `--expected-moves` or `--expect-reject`, `--scenario`, `--session-id`, `--output-root`

- [ ] **Step 1: Write failing disabled-default tests**

Assert normal sessions expose `transition_evidence is None` and allocate no crop arrays. With explicit capture enabled, one terminal decision contains 1–4 highest-difference points, stable ascending indices, 48×48 owned before/after BGRA crops, all 90 local differences and `decision_latency_ms`; mutating the source frames afterward must not change evidence.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync\test_tracker.py tests\unit\sync\test_live_session.py tests\unit\scripts\test_collect_human_ai_stage_c.py -q
```

Expected: failures because transition evidence and collector arguments do not exist.

- [ ] **Step 3: Implement bounded in-memory evidence**

The tracker selects the four greatest positive local differences from the already-computed terminal observation, crops only those points from the confirmed and settled frames, and copies the arrays. It does not persist anything. `LiveSyncSession` forwards evidence only when explicitly enabled and otherwise preserves the current memory and API behavior.

- [ ] **Step 4: Implement manual-ground-truth collector**

The collector requires exactly one visible `天天象棋` candidate, `HUMAN_VS_AI`, explicit FEN/quad and an explicit human ground-truth label supplied before the event. It starts the session, prints `BASELINE_READY`, waits for one accepted or paused terminal event, converts evidence to `HumanAiStageCSampleV1`, then records via the explicitly enabled recorder. It never clicks, moves, resizes, activates or minimizes a window. Timeout or capture loss writes no sample.

- [ ] **Step 5: Test with fake frames and commit**

Use `FakeFrameSource` and synthetic boards to prove accepted, rejected, timeout, resize, window-close and duplicate-ID behavior. The CLI test must verify no sample appears before a terminal event and no full-frame-sized PNG exists afterward.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync tests\unit\scripts\test_collect_human_ai_stage_c.py -q
git add src\xiangqi_agent\sync scripts\collect_human_ai_stage_c.py tests\unit
git commit -m "feat: collect labeled human-ai gate events safely"
```

---

### Task 6: Automatic Gates, Real Smoke, and Stage C Status

**Files:**
- Create: `docs/status/human-ai-stage-c.md`
- Modify only after a failing regression test: files from Tasks 1–5

- [ ] **Step 1: Run focused and full automatic gates**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\sync tests\unit\diagnostics tests\unit\scripts -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src
git diff --check
```

- [ ] **Step 2: Run refined privacy scans**

Verify no tracked `.env`, `.db`, `.sqlite`, `.png`, `.onnx`, engine binary, model binary, `.local` content or long API-key-like literal exists. Preserve the user-owned untracked `docs/superpowers/plans/2026-08-28-remaining-execution-plan.md` unchanged and unstaged.

- [ ] **Step 3: Complete one read-only real smoke**

With the user controlling all UI, start from the confirmed FEN, establish a quiet baseline, instruct one legal user move, allow the human-vs-AI reply and record whether the observer emits two singles, one atomic double or a safe rejection. Do not describe a rejection as a pass; do not save a full screenshot.

- [ ] **Step 4: Write truthful status evidence**

Document the automatic outputs, real smoke result, capture backend/size, ambient hover/highlight limitation and sample counts. Until the frozen report has 30 valid independent sessions, 30 rejection events, zero false accepts, at least 80% coverage and P95 at most 500 ms, state explicitly that realtime engine/coach remains locked.

- [ ] **Step 5: Commit and push only the verified checkpoint**

```powershell
git add docs\status\human-ai-stage-c.md
git commit -m "test: document human-ai Stage C evidence"
git status --short
git push origin develop
```

Do not create a milestone tag. If remote history advanced, stop and inspect instead of forcing.

---

## Post-plan Continuation

After this plan's infrastructure is complete:

1. Collect the frozen 30 valid/30 rejection set only through user-controlled human-vs-AI practice sessions.
2. If Stage C fails, use a separate development set to improve features; never tune against the frozen set.
3. When Stage C passes, enable final-confirmed-position Pikafish/DeepSeek only for `HUMAN_VS_AI` and retain the global kill switch.
4. Continue SQLite review persistence, 30-minute lifecycle, package/license audit and ten-game release acceptance.
