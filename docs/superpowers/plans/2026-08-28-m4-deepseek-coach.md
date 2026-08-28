# M4 DeepSeek Coach Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 把本地规则和 Pikafish 结果封装成可验证证据，通过严格 JSON 契约生成可分级、可追问、可比较的中文教学解释。

**Architecture:** coach 层只读取已确认局面和引擎证据，以稳定 candidate_id 限制模型引用；Pydantic 校验、允许走法扫描和确定性本地模板共同阻止未经验证的结论进入 UI。

**Tech Stack:** Python 3.12、PySide6、windows-capture 2.0.1、OpenCV、NumPy、ONNX Runtime、Pikafish 2026-01-02、HTTPX、Pydantic、SQLite、keyring、pytest、pytest-qt、ruff、mypy、PyInstaller。

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

**Roadmap Scope:** Tasks 15-17 from docs/superpowers/plans/2026-08-28-xiangqi-learning-agent.md. The task sections below are copied verbatim from that roadmap.

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
├─ src/xiangqi_agent/
│  ├─ domain/{coach,notation}.py
│  ├─ coach/{evidence,prompts,client,service}.py
│  ├─ application/controller.py
│  └─ ui/{coach_panel,analysis_panel}.py
├─ src/xiangqi_agent/ui/dialogs/settings_dialog.py
└─ tests/{unit,integration,ui,fixtures/deepseek}/
~~~

文件仍按总路线图的职责边界拆分；本节只列出本里程碑直接创建或修改的主要区域。

---

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

---

## Milestone Acceptance Gate

- CoachEvidence 只包含经过规则层和 Pikafish 验证的结构化文本，不包含截图、API Key 或完整隐私数据。
- DeepSeek 实时与深度模式、12 秒超时、严格 JSON 校验、candidate_id 白名单、具体走法扫描、单次重试和本地降级测试全部通过。
- 分级提示按形势、计划、候选、PV 顺序揭示；用户具体走法只有解析唯一且先经 Pikafish 比较后才进入证据。
- 新 position_id 清除旧问题上下文，无 API Key、超时、余额不足或无效响应时同步和本地引擎分析保持可用。

## Milestone Tag

全部验收门通过并提交验收证据后，创建 annotated tag `v0.1.0-m4`。

## Push Gate

只有本里程碑的全部测试通过、独立评审完成、隐私扫描无发现且工作树干净时，才允许推送分支和 annotated tag。任一条件未满足时不得推送。
