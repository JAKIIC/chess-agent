# M0 仓库基线状态

日期：2026-08-28

## 已完成

- 建立公开仓库的安全基线和 MIT 许可证。
- 保留完整产品设计和 21 项详细实施计划。
- 将完整路线图拆分为五份自包含执行计划，并定义里程碑索引、验收门、未来使用的 `v0.1.0-m1` 至 `v0.1.0-m5` annotated tag 名称和统一推送门。
- 将设计说明和完整路线图的实施边界统一为 M1 至 M5 五个里程碑组。
- 确定 `main` 稳定分支、`develop` 开发分支及 `.worktrees/v0.1` 隔离工作区策略。
- 确定每个里程碑只有通过测试、独立评审和隐私扫描并保持干净工作树后，才允许推送和创建 annotated tag。

## 结构验证

2026-08-28 对五份里程碑计划执行结构验证，结果如下：

~~~text
STRUCTURAL VALIDATION: PASS
- milestone files: 5/5
- required sections: 5/5 plans
- global Task headers: 21/21 exactly once
- verbatim task bodies: 21/21 exact matches
- code fences: balanced in 5/5 plans
- milestone tags: 5/5 exact
- push gates: 5/5 complete
- banned markers: 0
~~~

## 当前能力

尚无可运行应用。本里程碑只交付可审计的设计、计划、仓库规则和开发流程。

## 已知限制

- 尚未创建 Python 工程或安装项目依赖。
- 尚未接入微信窗口、Pikafish 或 DeepSeek。
- GitHub 首次推送需要在浏览器完成一次账户授权。

## 隐私与资产

`.gitignore` 排除 API Key、环境文件、完整截图、用户数据库、运行日志、下载的引擎、模型二进制、构建产物和 Superpowers 临时账本。
