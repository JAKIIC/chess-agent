# 真实棋局视觉证据闭环设计

**日期：** 2026-08-30  
**状态：** 已按用户批准的加速路线定稿  
**实施范围：** Accelerated Batch 1.4–1.5  
**上位约束：** `docs/superpowers/specs/2026-08-28-xiangqi-learning-agent-design.md`

## 1. 目标

在不扩展到通用 ONNX 全盘识别的前提下，建立一条可回放、可比较、可冻结门限的真实视觉证据闭环。当画面中出现一步合法走棋时，系统必须先产生不改变棋局的证据和候选，只有所有硬门通过后才能由唯一提交层更新 `BoardState`。

本设计直接解决 Batch 1.3 的真实阻塞：突发采样已经抓到用户单步，正确走法 `i0h0`也已排名第一，但整块 RGB 距离受棋盘纹理、高亮、阴影和亚像素偏移干扰，严格语义门拒绝接受。

## 2. 保留的约束

- 只读可见窗口，不点击、不注入、不读取微信进程内存、不自动走棋。
- 仅用于人机练习、残局训练和赛后复盘；不在真人在线对局中提供实时引擎建议。
- 第一版只支持标准初始局面或用户手工 FEN。
- 窗口必须可见、无遮挡；不承诺最小化或遮挡捕获。
- 只使用手工四角标定，不实现自动棋盘定位。
- 默认 2 FPS，变化后短暂突发采样；只在动画结束且画面稳定后推断。
- 只枚举当前 `BoardState` 的合法下一步；只有唯一高证据候选才可接受。
- 默认不保存完整截图，不保存窗口标题、昵称、头像或账号信息。
- 诊断样本功能必须显式开启，默认关闭，仅写入 Git 忽略的 `.local/`。

## 3. 架构边界

```text
CaptureFrame + BoardGeometry + Confirmed BoardState
                    ↓
              MoveObserver
                    ↓
       MoveEvidence + MoveProposal
       （不包含新 BoardState）
                    ↓
             StableMoveTracker
                    ↓
              StateCommitter
                    ↓
            ConfirmedPosition
```

### 3.1 证据与提交

- `MoveEvidence` 是不可变事实对象，记录局部帧差、候选外变化、语义距离、实例迁移证据和拒绝原因。
- `MoveProposal` 只表示 `ACCEPTED / NO_CHANGE / AMBIGUOUS` 和候选走法，不调用 `apply_move()`。
- `StableMoveTracker` 完成稳定帧门、验证提案中走法仍然合法，并把它交给提交层。
- `StateCommitter` 是唯一允许调用 `apply_move()` 并生成新 `BoardState` 的组件。
- 以后的 Pikafish 和 DeepSeek 只订阅 `ConfirmedPosition`，绝不订阅 `MoveProposal`。

### 3.2 分层状态

不用一个枚举混合所有状态：

- 捕获生命周期：`OPEN / CLOSED`，由 `FrameSource` 拥有。
- 捕获上下文：`VALID / CONTEXT_INVALID`，由窗口尺寸、DPI、几何版本和主题指纹决定。
- 棋局同步：`WATCHING / WAITING_FOR_STABLE / ACCEPTED / PAUSED_AMBIGUOUS / DESYNCHRONIZED / MANUAL_RECOVERY_REQUIRED`。

暂停后不得自动猜测。用户可以确认实际走法、重新输入 FEN 或重新标定，然后从新的确认帧重启跟踪。

## 4. 隐私化样本

### 4.1 `EndpointSampleV1`

每个可候选事件保存四个 48×48 BGRA 裁片：

```text
source_before
source_after
target_before
target_after
```

元数据包含：

- `schema_version`、`sample_id`、`session_id`、`sample_kind`、`created_at_utc`。
- `confirmed_fen`、`confirmed_position_id`、`actual_uci`、`probe_uci`、`side_to_move`、`orientation`。
- `source_index`、`target_index`、`top_k_candidates`、`rejection_reason`。
- `wgc_size`、`client_size`、`dpi_scale`、`geometry_revision`、`theme_fingerprint`。
- `feature_version`、`threshold_profile_version`、各项变化和语义分数。

`actual_uci` 可为 `null`，用于高亮、点选未走、动画、两步合并等应拒绝事件。`probe_uci` 是当时得分最高且被保存端点的候选。

### 4.2 磁盘安全

- 样本根目录必须位于 `.local/endpoint-samples/`。
- 每个样本先写临时目录，全部 PNG 和 `manifest.json` 成功后再原子改名。
- 记录器拒绝尺寸超过限制的图像，防止误传完整帧。
- 默认容量上限 256 MiB；超限时拒绝新样本，不静默删除。
- 支持删除单个会话或全部样本，仅承诺应用层删除，不宣称 SSD 安全擦除。
- 任何日志和元数据不得包含窗口标题、完整窗口图、API Key 或账号信息。

## 5. 确定性回放

`EndpointSampleLoader` 对 schema、哈希、图像尺寸和数据类型进行校验。`EndpointReplayer` 在相同样本、特征版本和门限配置下必须返回完全一致的结果：

- 实际走法和探测候选。
- 各项特征分数、接受/拒绝结果和拒绝原因。
- 接受时生成的新 FEN；拒绝时不得生成新 FEN。
- 特征计算耗时，但耗时不参与结果等值判定。

## 6. 特征实验

所有实验通过统一 `EndpointFeatureExtractor` 契约返回非概率性的 `EndpointFeatures`。在没有真实数据校准前，代码和 UI 使用 `evidence_score`，不使用“百分比置信度”。

实验顺序固定为：

1. `rgb-v1`：当前整块 RGB 平均绝对距离基线。
2. `masked-lab-v1`：圆形软掩膜、Lab 色度和局部亮度归一化。
3. `aligned-gradient-v1`：增加 Sobel 梯度并做 ±3 像素平移搜索。
4. `instance-transfer-v1`：把 `source_before` 与 `target_after` 作为同一枚棋子的迁移证据，与空位证据、阵营证据和候选唯一性分别建立硬门。
5. 仅当上述手工特征在冻结盲测上无法达标时，才立项 Siamese ONNX；本设计不实现模型训练。

`MoveSemanticGate` 必须使用多个硬门，不允许一个加权总分掩盖某项证据失败：

```text
legal move
AND stable frames
AND unchanged capture context
AND source and target changed
AND outside change below limit
AND source-after empty evidence passes
AND source-before to target-after instance evidence passes
AND destination side evidence passes
AND top-1 margin passes
```

## 7. 数据门和验收

### 阶段 A：采集链路冒烟

- 5 个真实单步，四元组裁片和标签全部正确。
- 同一样本重复回放结果完全一致。
- 0 张完整窗口截图，0 个窗口标题或账号字段。

### 阶段 B：开发集

- 至少 40 个真实单步，来自不少于 4 个会话，红黑双方均有样本，覆盖至少 5 类棋子和 10 个吃子走法。
- 至少 40 个应拒绝事件，覆盖无变化、点选高亮、动画、两步合并、resize 和遮挡。
- 特征参数和门限只允许在开发集上确定。

### 阶段 C：冻结盲测

- 冻结特征版本和门限后，使用新会话的 30 个真实单步和 30 个拒绝事件。
- 合法单步候选 top-1 准确率不低于 95%。
- 被自动接受的走法和新 FEN 必须 100% 正确；盲测中 0 个错误接受。
- 有效单步自动接受覆盖率目标不低于 85%，未达标只影响可用性，不得放宽错误接受硬门。
- 高亮、动画、两步、resize 等事件更新 `BoardState` 的次数为 0。
- 特征计算 P95 不超过 25 ms；最后一次画面变化到决策 P95 不超过 500 ms。

这些数量是 v0.1 原型工程门，不代表理论错误率为零，也不支持超出已锁定使用场景的可靠性声明。

## 8. 与后续模块的门

- Pikafish UCI 适配器可以用手工 FEN 独立开发和测试。
- 在阶段 C 通过之前，实时视觉链路不得自动触发 Pikafish。
- DeepSeek 暂不接真实 API；只在引擎证据契约完成后用 mock 进行校验器测试。
- 最小诊断窗口可在采集链路通过后实现，完整产品 UI、SQLite、PyInstaller 和十盘验收不属于本设计。

