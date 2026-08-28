# 天天象棋学习助手实施路线图

本目录以[完整 21 项路线图](2026-08-28-xiangqi-learning-agent.md)为追踪基线，以[已批准设计说明](../specs/2026-08-28-xiangqi-learning-agent-design.md)为约束权威。五份里程碑计划各自包含完整计划头、全局约束、相关文件结构、原始任务正文、验收门、annotated tag 和推送门，可以分别执行和审查。

| 里程碑 | 范围 | 交付目标 | 检查点 tag |
| --- | --- | --- | --- |
| [M1 Foundation](2026-08-28-m1-foundation.md) | Tasks 1-8 | 工程、领域内核、安全配置、桌面壳、窗口捕获与标定 | `v0.1.0-m1` |
| [M2 Recognition and Sync](2026-08-28-m2-recognition-sync.md) | Tasks 9-12 | 识别资产门、ONNX 识别、多帧稳定与可靠同步 | `v0.1.0-m2` |
| [M3 Pikafish Analysis](2026-08-28-m3-pikafish-analysis.md) | Tasks 13-14 | 可审计受管引擎与两阶段本地分析 | `v0.1.0-m3` |
| [M4 DeepSeek Coach](2026-08-28-m4-deepseek-coach.md) | Tasks 15-17 | 证据约束的 DeepSeek 教练、分级提示与走法比较 | `v0.1.0-m4` |
| [M5 Persistence and Release](2026-08-28-m5-release.md) | Tasks 18-21 | 本地复盘、恢复与隐私、Windows 打包和发布验收 | `v0.1.0-m5` |

## Execution Rules

- 严格按 M1 → M2 → M3 → M4 → M5 顺序执行；识别可靠性门未通过时不得进入依赖其结果的里程碑。
- 每份计划中的任务正文来自完整路线图，接口、命令、测试预期和提交消息保持不变。
- 每个任务先运行失败测试，再实现最小功能并运行通过测试；里程碑结束时执行该计划的完整验收门。
- 只有测试通过、独立评审完成、隐私扫描无发现且工作树干净时，才允许推送分支和对应 annotated tag。
- 最终 `v0.1.0` tag 只在 M5 的全部自动与人工发布门通过后创建。
