# 证据约束教练状态

**日期：** 2026-08-31

**分支：** `develop`

**结论：** 本地解释闭环已通过真实 Pikafish 窗口冒烟；DeepSeek 客户端已按当前官方
Chat Completions 契约实现，既通过模拟 HTTP 集成测试，也已使用保存在 Windows
Credential Manager 的用户 Key 完成一次真实受约束调用。

## 1. 教练证据边界

`CoachEvidence` 只包含：

- 当前 `position_id`、FEN、用户执方和程序判定的阶段；
- 各类棋子数量、当前一方是否被将；
- 最多三条 Pikafish 候选的 UCI、中文记谱、红方视角评分、深度和变化线；
- 由规则层复演确认的立即吃子/将军事实；
- `candidate_1..3 → 中文走法` 白名单。

每条 PV 都从当前 `BoardState` 逐步调用合法走法和 `apply_move` 复演；任一步非法就拒绝
整个证据包。证据模型没有截图、图像字节或 API Key 字段。

## 2. DeepSeek 契约与防越界

2026-08-31 核对官方文档后采用：

- `POST https://api.deepseek.com/chat/completions`
- 实时模型 `deepseek-v4-flash`，`thinking.type=disabled`
- 深度模型 `deepseek-v4-pro`，thinking enabled、reasoning effort high
- `response_format={"type":"json_object"}`，系统提示显式要求 JSON
- 12 秒默认超时、非流式响应

官方核对来源：
[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、
[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)、
[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)。

返回对象必须满足固定 `CoachExplanation` 字段，并再次经过：

1. `position_id` 一致性；
2. 主候选和 alternatives 的 candidate 白名单；
3. 回答文字中的 candidate ID 扫描；
4. UCI 走法白名单扫描；
5. 中文具体走法白名单扫描。

推荐着仍只能从 candidate 白名单中选择；解释后续变化时可以引用已经由规则层逐步复演
的 PV 着法。真实冒烟发现并修复了“已验证 PV 被误当成越界走法”的过度拒绝问题，任意
不在候选或 PV 中的走法仍会被拒绝。

首次结构或白名单失败只重试一次；再次失败、空内容、超时、HTTP 错误或余额错误均回退
到确定性的本地证据模板。错误响应和 API Key 不写日志。

## 3. 凭据与用户界面

- API Key 只通过现有 `SecretStore` 进入 Windows Credential Manager。
- 设置窗口只显示“已配置/未配置”，不读取到输入框，也不回显现有 Key。
- 保存后密码输入框立即清空；支持删除系统中保存的 Key。
- 教练默认逐层显示：形势、计划、候选理由、威胁/替代/训练问题。
- 可选择“我是红方/我是黑方”，并可追问“为什么推荐这一步”。
- 中文走法引用通过枚举当前合法走法精确解析；零个或多个匹配都不猜。
- 新 FEN 会清除旧证据和旧回答；异步服务不显示过期 `position_id` 的结果。

## 4. 真实闭环冒烟

使用已安装的真实 Pikafish 对当前已知局面完成加深分析，再在无 API Key 模式提问。
教练页成功显示四级提示、当前首选中文记谱、引擎深度/评分、变化线下一手和训练问题；
窗口关闭后无引擎残留。调试截图只位于 Git 忽略的 `.local/`。

随后把用户提供的 Key 通过隐藏输入写入 Windows Credential Manager，并只以布尔值
确认配置状态。真实 DeepSeek 请求经历“Pikafish 三候选 → CoachEvidence → 云端 JSON →
本地白名单扫描”，最终结果满足：

- `source=deepseek`
- 返回 `position_id` 与当前局面一致
- 主候选在 `allowed_move_map` 中
- alternatives 全部属于白名单
- 置信度为 0.7
- 请求结束后 Pikafish 正常退出

完整 Key、Authorization 头和完整请求均未输出或写入项目文件。

## 5. 自动验证

| 检查 | 结果 |
|---|---|
| Coach 单元/HTTP 集成测试 | 16 passed |
| UI、设置、走法引用新增测试 | 6 passed |
| 全量 pytest | 269 passed |
| Ruff | All checks passed |
| mypy | 54 source files, no issues |
| `git diff --check` | 通过 |

## 6. 尚未验收

- 仅完成一次真实 API 功能冒烟，尚未进行持续负载、限流或长时间服务质量测试。
- 用户提到非前三候选走法时能够精确识别，但尚未启动新的 Pikafish 分支分析来量化损失。
- 视觉严格门仍缺真实端点样本；教练只服务于手工确认或规则层确认的局面。
- 捕获状态、手工四角标定和同步警告尚未并入主窗口。
