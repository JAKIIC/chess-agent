# 证据约束教练状态

**日期：** 2026-08-31

**分支：** `develop`

**结论：** 本地解释闭环已通过真实 Pikafish 窗口冒烟；DeepSeek 客户端已按当前官方
Chat Completions 契约实现并通过模拟 HTTP 集成测试。由于尚未配置真实 API Key，真实
DeepSeek 网络调用明确保持未验收。

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

## 4. 真实本地闭环冒烟

使用已安装的真实 Pikafish 对当前已知局面完成加深分析，再在无 API Key 模式提问。
教练页成功显示四级提示、当前首选中文记谱、引擎深度/评分、变化线下一手和训练问题；
窗口关闭后无引擎残留。调试截图只位于 Git 忽略的 `.local/`。

## 5. 自动验证

| 检查 | 结果 |
|---|---|
| Coach 单元/HTTP 集成测试 | 15 passed |
| UI、设置、走法引用新增测试 | 6 passed |
| 全量 pytest | 268 passed |
| Ruff | All checks passed |
| mypy | 54 source files, no issues |
| `git diff --check` | 通过 |

## 6. 尚未验收

- 尚未使用用户真实 API Key 调用 DeepSeek，因此不宣称真实网络、账户余额或服务质量通过。
- 用户提到非前三候选走法时能够精确识别，但尚未启动新的 Pikafish 分支分析来量化损失。
- 视觉严格门仍缺真实端点样本；教练只服务于手工确认或规则层确认的局面。
- 捕获状态、手工四角标定和同步警告尚未并入主窗口。
