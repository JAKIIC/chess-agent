# Visual Evidence Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a privacy-preserving, deterministic endpoint-sample and replay loop, then replace uncalibrated whole-patch RGB semantics with explicit evidence and robust source-to-target instance matching.

**Architecture:** `MoveObserver` produces immutable evidence and a proposal without creating a new board. A rule-backed committer is the only component that advances `BoardState`. Opt-in diagnostics persist only four endpoint crops plus sanitized metadata under `.local/`; a deterministic replayer compares versioned feature extractors before any real-time threshold is frozen.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pytest, Ruff, mypy, existing Win32/WGC capture and Xiangqi domain modules.

**Spec:** `docs/superpowers/specs/2026-08-30-visual-evidence-loop-design.md`

## Global Constraints

- Do not click, inject, automate moves, inspect WeChat memory, or support live engine advice in human online games.
- Use only a visible, unobstructed target window and the existing manual four-corner geometry.
- Diagnostic persistence is disabled by default and may write only endpoint crops under `.local/endpoint-samples/`.
- Never persist a full capture frame, window title, nickname, avatar, API key, or account field.
- A proposal that fails any hard gate must not change `BoardState` or `position_id`.
- Use `evidence_score`, not probability-like `confidence`, until a held-out calibration exists.
- Keep the user's untracked `docs/superpowers/plans/2026-08-28-remaining-execution-plan.md` unchanged.
- All production behavior changes follow red-green-refactor with focused tests first.

---

### Task 1: Separate visual evidence from board-state commit

**Files:**
- Create: `src/xiangqi_agent/sync/evidence.py`
- Create: `src/xiangqi_agent/sync/committer.py`
- Modify: `src/xiangqi_agent/sync/move_observer.py`
- Modify: `src/xiangqi_agent/sync/tracker.py`
- Modify: `src/xiangqi_agent/sync/__init__.py`
- Modify: `tests/unit/sync/test_move_observer.py`
- Modify: `tests/unit/sync/test_tracker.py`
- Create: `tests/unit/sync/test_committer.py`

**Interfaces:**
- Produces: `CandidateEvidence`, `MoveEvidence`, `MoveProposal`, `StateCommitter`, `RuleStateCommitter.commit(board, move) -> BoardState`.
- Changes: `MoveObserver.observe(...) -> MoveProposal`; proposals never carry a post-move `BoardState`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_observer_proposes_move_without_creating_board_state() -> None:
    proposal = observer.observe(board, before, after, geometry)
    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.move == move
    assert not hasattr(proposal, "after")

def test_rule_committer_is_the_only_component_that_advances_board() -> None:
    after = RuleStateCommitter().commit(board, move)
    assert after.position_id != board.position_id
    assert after.side_to_move != board.side_to_move
```

- [ ] **Step 2: Run the focused tests and verify they fail because the new contracts do not exist**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/sync/test_committer.py tests/unit/sync/test_move_observer.py tests/unit/sync/test_tracker.py -q`

- [ ] **Step 3: Implement immutable evidence/proposal contracts and rule committer**

```python
@dataclass(frozen=True, slots=True)
class MoveProposal:
    status: ObservationStatus
    move: Move | None
    evidence_score: float
    evidence: MoveEvidence

class RuleStateCommitter:
    def commit(self, board: BoardState, move: Move) -> BoardState:
        return apply_move(board, move)
```

- [ ] **Step 4: Update tracker to commit only an accepted legal proposal and verify focused tests pass**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/sync -q`

- [ ] **Step 5: Commit the isolated change**

```powershell
git add src/xiangqi_agent/sync tests/unit/sync
git commit -m "refactor: separate move evidence from state commit"
```

### Task 2: Add layered context validity and manual recovery

**Files:**
- Create: `src/xiangqi_agent/capture/context.py`
- Modify: `src/xiangqi_agent/sync/tracker.py`
- Create: `tests/unit/capture/test_context.py`
- Modify: `tests/unit/sync/test_tracker.py`

**Interfaces:**
- Produces: `CaptureContext`, `ContextStatus`, `TrackingStatus.DESYNCHRONIZED`, `TrackingStatus.MANUAL_RECOVERY_REQUIRED`.
- Produces: `StableMoveTracker.invalidate_context()`, `mark_desynchronized()`, and `recover(board, frame)`.

- [ ] **Step 1: Write failing state and generation tests**

```python
def test_context_change_invalidates_old_capture_generation() -> None:
    first = CaptureContext((100, 100), (100, 100), 1.0, "g1", "t1", 1)
    second = CaptureContext((101, 100), (100, 100), 1.0, "g1", "t1", 2)
    assert not first.compatible_with(second)

def test_manual_recovery_replaces_board_and_confirmed_frame() -> None:
    tracker.mark_desynchronized()
    update = tracker.recover(recovered_board, recovered_frame)
    assert update.status is TrackingStatus.WATCHING
    assert tracker.board == recovered_board
```

- [ ] **Step 2: Run focused tests and verify the missing status/API failures**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/capture/test_context.py tests/unit/sync/test_tracker.py -q`

- [ ] **Step 3: Implement separate capture-context and sync-state contracts**

```python
@dataclass(frozen=True, slots=True)
class CaptureContext:
    wgc_size: tuple[int, int]
    client_size: tuple[int, int]
    dpi_scale: float
    geometry_revision: str
    theme_fingerprint: str
    generation_id: int
```

- [ ] **Step 4: Implement explicit recovery and verify it resets all motion/pause counters**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/capture/test_context.py tests/unit/sync/test_tracker.py -q`

- [ ] **Step 5: Commit the state changes**

```powershell
git add src/xiangqi_agent/capture/context.py src/xiangqi_agent/sync/tracker.py tests/unit/capture/test_context.py tests/unit/sync/test_tracker.py
git commit -m "feat: add explicit sync recovery states"
```

### Task 3: Define and persist privacy-safe endpoint samples

**Files:**
- Create: `src/xiangqi_agent/diagnostics/endpoint_samples.py`
- Modify: `src/xiangqi_agent/diagnostics/__init__.py`
- Create: `tests/unit/diagnostics/test_endpoint_samples.py`

**Interfaces:**
- Produces: `EndpointSampleV1`, `EndpointCrops`, `EndpointSampleRecorder.record(...) -> Path`, `delete_session(session_id)`, `delete_all()`.
- Consumes: confirmed FEN/position ID, actual/probe UCI, sanitized `CaptureContext`, four BGRA endpoint crops and serialized top-k evidence.

- [ ] **Step 1: Write failing round-trip, opt-in, size-limit, quota and privacy tests**

```python
def test_recorder_writes_only_four_small_crops_and_sanitized_manifest(tmp_path: Path) -> None:
    sample_dir = recorder(tmp_path, enabled=True).record(sample, crops)
    assert sorted(path.name for path in sample_dir.iterdir()) == [
        "manifest.json", "source_after.png", "source_before.png",
        "target_after.png", "target_before.png",
    ]
    manifest = json.loads((sample_dir / "manifest.json").read_text("utf-8"))
    assert "window_title" not in json.dumps(manifest)
    assert all(cv2.imread(str(path)).shape[:2] == (48, 48) for path in sample_dir.glob("*.png"))

def test_recorder_disabled_by_default_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsDisabledError):
        EndpointSampleRecorder(tmp_path).record(sample, crops)
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the new test file and verify it fails because the module is absent**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/diagnostics/test_endpoint_samples.py -q`

- [ ] **Step 3: Implement validated immutable schema and atomic recorder**

```python
@dataclass(frozen=True, slots=True)
class EndpointCrops:
    source_before: NDArray[np.uint8]
    source_after: NDArray[np.uint8]
    target_before: NDArray[np.uint8]
    target_after: NDArray[np.uint8]
```

- [ ] **Step 4: Implement quota rejection and application-level deletion; verify no full-frame dimensions can be written**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/diagnostics/test_endpoint_samples.py -q`

- [ ] **Step 5: Commit the diagnostics store**

```powershell
git add src/xiangqi_agent/diagnostics tests/unit/diagnostics/test_endpoint_samples.py
git commit -m "feat: record privacy-safe endpoint samples"
```

### Task 4: Build deterministic sample loading and replay

**Files:**
- Create: `src/xiangqi_agent/diagnostics/endpoint_replay.py`
- Create: `scripts/replay_endpoint_samples.py`
- Create: `tests/unit/diagnostics/test_endpoint_replay.py`
- Create: `tests/unit/scripts/test_replay_endpoint_samples.py`

**Interfaces:**
- Produces: `EndpointSampleLoader.load(path)`, `EndpointReplayResult`, `EndpointReplayer.replay(sample_dir, extractor, gate)`.
- CLI emits deterministic JSON excluding runtime duration from equality checks.

- [ ] **Step 1: Write failing corruption and deterministic replay tests**

```python
def test_same_sample_feature_and_threshold_versions_replay_identically(sample_dir: Path) -> None:
    first = replayer.replay(sample_dir)
    second = replayer.replay(sample_dir)
    assert first.without_runtime() == second.without_runtime()

def test_loader_rejects_changed_crop_hash(sample_dir: Path) -> None:
    (sample_dir / "source_before.png").write_bytes(b"changed")
    with pytest.raises(SampleIntegrityError):
        loader.load(sample_dir)
```

- [ ] **Step 2: Verify tests fail for absent loader/replayer**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/diagnostics/test_endpoint_replay.py tests/unit/scripts/test_replay_endpoint_samples.py -q`

- [ ] **Step 3: Implement manifest/crop hash validation and deterministic result objects**

- [ ] **Step 4: Implement CLI JSON output and verify focused tests pass**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/diagnostics/test_endpoint_replay.py tests/unit/scripts/test_replay_endpoint_samples.py -q`

- [ ] **Step 5: Commit loader and replay CLI**

```powershell
git add src/xiangqi_agent/diagnostics/endpoint_replay.py scripts/replay_endpoint_samples.py tests/unit/diagnostics/test_endpoint_replay.py tests/unit/scripts/test_replay_endpoint_samples.py
git commit -m "feat: replay endpoint evidence deterministically"
```

### Task 5: Version the RGB baseline and robust endpoint feature contract

**Files:**
- Create: `src/xiangqi_agent/vision/endpoint_features.py`
- Create: `tests/unit/vision/test_endpoint_features.py`

**Interfaces:**
- Produces: `EndpointFeatureExtractor`, `EndpointFeatures`, `RgbBaselineExtractor`, `MaskedLabExtractor`, `AlignedGradientExtractor`, `InstanceTransferExtractor`.

- [ ] **Step 1: Write failing invariant tests using hand-built synthetic pieces and backgrounds**

```python
def test_instance_transfer_is_stable_across_board_background_and_small_translation() -> None:
    features = InstanceTransferExtractor(max_shift=3).extract(crops)
    assert features.instance_distance < 0.12
    assert features.best_shift == (2, -1)

def test_mask_rejects_background_only_similarity() -> None:
    features = MaskedLabExtractor().extract(background_only_crops)
    assert features.instance_evidence_score < 0.5
```

- [ ] **Step 2: Run tests and verify missing extractor failures**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/vision/test_endpoint_features.py -q`

- [ ] **Step 3: Implement versioned RGB and circular-mask Lab extractors**

```python
class EndpointFeatureExtractor(Protocol):
    version: str
    def extract(self, crops: EndpointCrops) -> EndpointFeatures: ...
```

- [ ] **Step 4: Implement Sobel gradient plus bounded integer alignment and source-before to target-after instance evidence**

- [ ] **Step 5: Run focused tests and commit feature implementations**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/vision/test_endpoint_features.py -q`

```powershell
git add src/xiangqi_agent/vision/endpoint_features.py tests/unit/vision/test_endpoint_features.py
git commit -m "feat: compare robust endpoint visual features"
```

### Task 6: Integrate independent semantic hard gates

**Files:**
- Create: `src/xiangqi_agent/sync/semantic_gate.py`
- Modify: `src/xiangqi_agent/sync/move_observer.py`
- Create: `tests/unit/sync/test_semantic_gate.py`
- Modify: `tests/unit/sync/test_move_observer.py`

**Interfaces:**
- Produces: `SemanticThresholds`, `SemanticGateResult`, `MoveSemanticGate.evaluate(...)`.
- Requires separate passes for source empty, instance transfer, side consistency and candidate margin; no weighted total may override a failed hard gate.

- [ ] **Step 1: Write one failing test for each independent hard gate**

```python
@pytest.mark.parametrize("failed_gate", ["source_empty", "instance", "side", "margin"])
def test_any_failed_semantic_gate_rejects_the_move(failed_gate: str) -> None:
    evidence = evidence_with_only_gate_failed(failed_gate)
    result = gate.evaluate(evidence)
    assert not result.accepted
    assert result.rejection_reasons == (failed_gate,)
```

- [ ] **Step 2: Verify tests fail because the semantic gate is absent**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/sync/test_semantic_gate.py -q`

- [ ] **Step 3: Implement named hard gates and update observer to emit rejection reasons**

- [ ] **Step 4: Verify synthetic accepted moves remain accepted and every ambiguous path preserves the old board**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/sync/test_semantic_gate.py tests/unit/sync/test_move_observer.py tests/unit/sync/test_tracker.py -q`

- [ ] **Step 5: Commit semantic integration**

```powershell
git add src/xiangqi_agent/sync/semantic_gate.py src/xiangqi_agent/sync/move_observer.py tests/unit/sync
git commit -m "feat: gate moves with independent endpoint evidence"
```

### Task 7: Add opt-in real capture sampling and dataset evaluation

**Files:**
- Modify: `scripts/probe_move_observer.py`
- Create: `scripts/evaluate_endpoint_samples.py`
- Modify: `tests/unit/scripts/test_probe_move_observer.py`
- Create: `tests/unit/scripts/test_evaluate_endpoint_samples.py`

**Interfaces:**
- Adds explicit `--record-endpoints`, `--session-id`, `--actual-uci` and `--sample-root` options.
- Produces a JSON evaluation summary with top-1 correctness, accepted precision, coverage, rejection reasons and P95 feature duration.

- [ ] **Step 1: Write failing CLI privacy and metric tests**

```python
def test_probe_requires_explicit_opt_in_before_writing_samples(tmp_path: Path) -> None:
    assert run_probe_without_record_flag(tmp_path) == 0
    assert list(tmp_path.rglob("*.png")) == []

def test_evaluator_reports_zero_false_accepts_and_coverage() -> None:
    report = evaluate(labeled_results)
    assert report.false_accepts == 0
    assert report.accepted_precision == 1.0
    assert report.coverage == 0.85
```

- [ ] **Step 2: Verify focused tests fail for missing CLI options/evaluator**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/scripts/test_probe_move_observer.py tests/unit/scripts/test_evaluate_endpoint_samples.py -q`

- [ ] **Step 3: Implement opt-in recorder wiring without changing default probe behavior**

- [ ] **Step 4: Implement deterministic dataset metrics and verify focused tests pass**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/unit/scripts/test_probe_move_observer.py tests/unit/scripts/test_evaluate_endpoint_samples.py -q`

- [ ] **Step 5: Commit capture/evaluation tooling**

```powershell
git add scripts/probe_move_observer.py scripts/evaluate_endpoint_samples.py tests/unit/scripts
git commit -m "feat: collect and evaluate endpoint evidence"
```

### Task 8: Run quality gates and perform the real-data checkpoints

**Files:**
- Create: `docs/status/accelerated-batch-1-4.md`
- Create after feature experiments: `docs/status/accelerated-batch-1-5.md`
- Modify: `README.md`

**Interfaces:**
- Produces an auditable report for smoke, development-set and frozen blind-test gates.

- [ ] **Step 1: Run automated quality gates before real capture**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
git diff --check
```

- [ ] **Step 2: Capture five labeled real single moves with endpoint recording explicitly enabled and inspect every four-crop tuple**

- [ ] **Step 3: Verify the sample root contains no full-frame image, title/account field, API key pattern or tracked file**

```powershell
git check-ignore .local/endpoint-samples
git status --short --untracked-files=all
```

- [ ] **Step 4: Build the 40-positive/40-reject development set, compare feature versions, select thresholds and freeze the profile**

- [ ] **Step 5: Run the 30-positive/30-reject new-session blind test and enforce zero false accepts before real-time engine integration**

- [ ] **Step 6: Record exact evidence and limitations in Batch 1.4/1.5 status reports, run all gates again, then commit**

```powershell
git add README.md docs/status/accelerated-batch-1-4.md docs/status/accelerated-batch-1-5.md
git commit -m "docs: record endpoint evidence validation"
```

