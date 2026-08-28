# Accelerated Batch 0：捕获与几何可行性探针

**日期：** 2026-08-28
**分支：** `develop`
**起始检查点：** `aed484538c69d96e3ffdb4b87770e549410cba6c`
**结论：** 可继续采用“已知局面 + 人工四角 + 合法走法视觉差分”的垂直闭环路线，但 Batch 1 前仍需在真实对局棋盘画面上补一次四角现场确认。

## 1. 本批范围

本批只验证以下最短风险链路：

```text
人工选择顶级窗口
  → 可替换 FrameSource
  → 按 HWND 只读捕获 BGRA 帧
  → 人工归一化四角
  → 固定顺序 90 点和局部裁片
  → 全局帧差、局部帧差和稳定判定
```

本批没有实现 MainWindow、自动棋盘定位、ONNX、DeepSeek、SQLite、PyInstaller、里程碑标签或自动走棋。

## 2. 实现内容

### 捕获契约和窗口选择

- `CaptureFrame`：拥有自身内存的只读 BGRA `uint8` 帧，包含单调时间戳和 HWND。
- `FrameSource`：可替换的 `start(on_frame, on_closed)` / `close()` 协议。
- `FakeFrameSource`：支持可控帧、目标关闭和幂等 `close()`。
- `WindowsWindowCatalog`：使用 Win32 `EnumWindows` 枚举可见、非零客户区的微信/天天象棋候选顶级窗口。
- `select_window`：只接受人工给出的候选 HWND；没有候选或 HWND 不匹配时明确失败。
- `WindowsCaptureSource`：通过 `windows-capture==2.0.1` 的 Windows Graphics Capture 后端按 HWND 捕获；默认请求 2 FPS，关闭鼠标指针和黄色边框。

### 人工几何和变化检测

- `NormalizedQuad`：固定采用左上、右上、右下、左下顺序，校验范围、凸性和最小面积。
- `BoardGeometry`：支持红方在下和黑方在下，稳定生成与内部 `BoardState` 顺序一致的 90 点。
- `crop_intersections`：透视展开后输出 90 个拥有自身内存的 48×48 BGRA 小块；窗口尺寸变化后拒绝继续使用旧标定。
- `analyze_frame_change`：输出全局平均绝对帧差、90 个局部帧差、阈值以上变化点和变化最大的若干交点。
- `FrameStabilityDetector`：只有连续指定数量的低差异帧对才报告稳定，任何明显变化都会重置。
- `scripts/probe_capture.py`：支持列出候选、人工 HWND、约 3 秒捕获、人工四角文本输入和 JSON 指标输出；默认不保存图片。

### 配置调整

`AppSettings.capture_fps` 的安全默认值由 5 改为 2。`animation_fps` 暂时保留 10；本批没有实现动态升频控制器。

## 3. 唯一一次真实冒烟

快捷方式成功启动独立 `WeChatAppEx.exe` 小程序窗口。探针按人工选定 HWND 运行一次 3 秒捕获，得到：

| 指标 | 结果 |
|---|---:|
| 捕获后端 | Windows Graphics Capture (`windows-capture==2.0.1`) |
| 帧数 | 4 |
| 首/末帧尺寸 | 2535×1511 / 2535×1511 |
| 相邻尺寸变化 | 0 |
| 首末帧时间跨度 | 2.424 秒 |
| 有效帧率 | 1.237 FPS |
| 时间戳严格单调 | 是 |
| 捕获期间目标关闭 | 否 |

2 FPS 是 WGC 的最小更新间隔请求，不保证静态画面持续以精确 2 FPS 回调。本次静态画面的实际回调率为 1.237 FPS；如果 Batch 1 需要严格的采样节奏，应在 WGC 回调之外增加“保留最新帧 + 2 Hz 采样器”，而不是假设后端恒定推帧。

### 现场几何结果

启动后的画面停留在“玩法大厅”而非对局棋盘。根据“不点击目标程序”的约束，本次没有进入棋局；人工四角选择了大厅中的矩形“象棋”卡片，只验证坐标链路，不把它视为真实棋盘验收。

| 指标 | 结果 |
|---|---:|
| 四角输入方式 | 归一化 TL、TR、BR、BL 文本配置 |
| 交点数 | 90 |
| 第一/最后交点 | (760.20, 247.64) / (1778.87, 552.66) |
| 裁片数 | 90 |
| 单裁片形状 | 48×48×4 |
| 连续帧比较数 | 3 |
| 判定稳定的比较数 | 3 |
| 末尾连续稳定数 | 3 |
| 峰值全局帧差 | 0.0 |

静态实拍证明：帧尺寸、透视映射、90 点顺序、局部裁切和稳定画面判定能在真实 WGC 帧上工作。因为现场没有发生走子，本次没有得到真实“起点/终点局部变化”数据；聚焦测试使用合成变化验证了指定交点能排到第一名，且变化会重置稳定状态。

## 4. 调试图和隐私

默认探针不写图片。本次为人工读取四角显式启用了调试目录：

- `.local/accelerated-batch-0/first-frame.png`
- `.local/accelerated-batch-0/calibrated-grid.png`

`.local/` 已由 `.gitignore` 排除，两张图未进入 Git。报告不记录完整窗口标题，也不包含 API Key、数据库、日志或用户对话。

## 5. 错误路径覆盖

自动测试覆盖：

- 没有候选窗口。
- 人工选择的 HWND 不在候选中。
- 目标窗口在启动捕获前已经关闭。
- 捕获后端报告目标关闭，关闭回调只触发一次。
- 重复调用 `close()` 不重复停止或等待后端。
- 相邻帧尺寸变化能进入探针指标。
- 标定后帧尺寸变化会拒绝旧几何。
- 非 BGRA、非 `uint8` 帧被拒绝。
- 非法、越界、自交或过小四边形被拒绝。

真实窗口关闭和真实窗口尺寸变化没有在唯一一次冒烟中主动制造；本批只使用可控后端覆盖这些分支。

## 6. TDD 和质量门

RED 证据：首次聚焦运行因 `xiangqi_agent.capture`、`xiangqi_agent.platform` 和 `xiangqi_agent.vision` 尚不存在而出现 6 个预期收集错误；序列分析测试随后因 `analyze_change_sequence` 尚不存在而出现预期导入错误。

最终结果：

| 检查 | 结果 |
|---|---|
| Batch 0 聚焦测试 | 30 passed |
| 全量 pytest | 106 passed |
| Ruff | All checks passed |
| mypy | 21 source files, no issues |
| `git diff --check` | 通过，无空白错误；Git 仅提示现有 LF/CRLF 转换策略 |

## 7. 对原计划的修订建议

不删除原 spec、M1–M5 计划或 `2026-08-28-remaining-execution-plan.md`，但从 Batch 1 起不再把它们的横向顺序作为执行顺序。建议新建一份 accelerated vertical-slice spec 和对应短计划，明确以下覆盖关系：

1. 初始同步改为“标准初始局面或用户 FEN”，暂不做任意截图的全盘识别。
2. 稳态采样由 5 FPS 改为 2 FPS；检测到变化后才短暂升频，并等待连续稳定帧。
3. 自动棋盘定位延后；第一版只读取人工归一化四角，并在尺寸变化后要求重新标定。
4. 通用 ONNX 识别延后；第一版用固定主题、标准初始棋盘自动提取红黑 14 类棋子和空位模板。
5. 下一步识别的核心接口改为可替换 `MoveObserver`：枚举当前 `BoardState` 的全部合法走法，只对每个候选的起点/终点局部视觉变化评分。
6. 只有唯一候选同时超过绝对阈值和第二名差距阈值才产生走法；否则进入暂停状态，不更新 `BoardState`。
7. 先完成“捕获 → 稳定 → 唯一合法走法 → BoardState 更新 → 结果输出”的垂直闭环，再接 Pikafish 和最小教学界面。
8. 数据集、ONNX 训练、自动定位、SQLite、PyInstaller、十盘验收和通用主题支持继续保留为后续增强，不进入最近批次。
9. M1–M5 标签暂不创建；先为 accelerated 路线重新定义可验收检查点，再决定标签映射。

## 8. 下一批建议

建议 Accelerated Batch 1 只实现离线/假帧可验证的 `MoveObserver` 垂直内核：

```text
标准初始局面或 FEN
  → 固定主题模板引导
  → 画面变化与稳定门
  → legal_moves(BoardState)
  → 候选起点/终点差分评分
  → 唯一高置信 Move 或明确暂停
  → apply_move 后的新 BoardState
```

Batch 1 不应同时接 Pikafish、DeepSeek 或完整 MainWindow。开始 Batch 1 前，应由用户手工进入一次真实棋局并保持窗口可见，以补齐本批未完成的真实棋盘四角确认和一组真实落子差分样本。
