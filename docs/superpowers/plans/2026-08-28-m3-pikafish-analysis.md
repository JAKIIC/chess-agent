# M3 Pikafish Analysis Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 接入固定版本、可校验、受管的 Pikafish 独立进程，并提供可取消、不会显示过期结果的两阶段本地分析。

**Architecture:** engine 层通过 UCI 协议管理独立子进程，只消费已确认 BoardState；所有分析携带 position_id，服务层取消旧搜索并由 UI 丢弃不匹配结果。

**Tech Stack:** Python 3.12、PySide6、windows-capture 2.0.1、OpenCV、NumPy、ONNX Runtime、Pikafish 2026-01-02、HTTPX、Pydantic、SQLite、keyring、pytest、pytest-qt、ruff、mypy、PyInstaller。

**Spec:** docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md

**Roadmap Scope:** Tasks 13-14 from docs/superpowers/plans/2026-08-28-xiangqi-learning-agent.md. The task sections below are copied verbatim from that roadmap.

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
├─ scripts/download_pikafish.py
├─ src/xiangqi_agent/
│  ├─ domain/analysis.py
│  ├─ engine/{protocol,process,service}.py
│  ├─ application/controller.py
│  └─ ui/{analysis_panel,main_window}.py
└─ tests/{unit,integration,ui,fixtures/engine}/
~~~

文件仍按总路线图的职责边界拆分；本节只列出本里程碑直接创建或修改的主要区域。

---

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

---

## Milestone Acceptance Gate

- Pikafish 2026-01-02 的二进制、NNUE、源码链接、许可证和 SHA-256 均可审计，下载校验失败时不启动。
- UCI 握手、配置、info/bestmove 解析、评分归一化、超时、崩溃与 stop/quit 生命周期测试全部通过。
- 每个新局面先产生 500 ms 快速结果，再产生 3000 ms 加深结果；新 position_id 到达后旧结果不得更新 UI。
- 未配置 DeepSeek 时，分析面板仍可显示评分、前三候选和主要变化。

## Milestone Tag

全部验收门通过并提交验收证据后，创建 annotated tag `v0.1.0-m3`。

## Push Gate

只有本里程碑的全部测试通过、独立评审完成、隐私扫描无发现且工作树干净时，才允许推送分支和 annotated tag。任一条件未满足时不得推送。
