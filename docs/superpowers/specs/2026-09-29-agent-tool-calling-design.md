# Agent 工具调用设计

## 1. 目标

在现有 PPMI 预测、医学文献 RAG 和 DeepSeek 解读能力之上，增加一个轻量 Agent 调度层。用户提交一份患者数据和一个研究问题后，Agent 可以按任务需要调用本地预测工具与本地文献检索工具，最后返回带工具轨迹和文献依据的结构化研究结果。

本阶段只完成 Agent 工具调用和结果展示；正式的可下载最终报告放到下一阶段完成。

## 2. 设计原则

- 不引入 LangChain，直接使用 Python、现有 DeepSeek 请求方式和 Pydantic。
- 预测概率始终由本地模型计算，DeepSeek 不参与数值预测。
- 患者 ID 和 12 项原始特征不发送给 DeepSeek。
- Agent 只能调用代码中预先登记的工具，不能执行任意函数或命令。
- 每次运行设置最大调用轮数，防止无限循环和重复消耗 API 额度。
- 所有工具参数、DeepSeek 工具选择和最终输出都经过 Pydantic 校验。
- 功能仅用于科研演示，不输出诊断或治疗建议。

## 3. 方案选择

### 采用方案：轻量原生工具循环

新增一个独立 Agent 模块维护工具白名单和有限状态循环。DeepSeek根据当前任务返回结构化工具调用，本地代码校验后执行工具，再把脱敏结果交还给 DeepSeek，直到生成最终结构化回答或达到调用上限。

选择它的原因：依赖少、执行过程清楚、容易测试，也便于面试时解释 Agent 的“思考—调用工具—观察结果—继续处理”过程。

### 未采用方案

- 固定流水线：实现简单，但每次都按固定顺序执行，不能体现 Agent 根据任务选择工具。
- LangChain Agent：现成功能规模较小，引入框架会增加依赖和学习成本，并掩盖工具循环的核心原理。

## 4. 工具定义

### `predict_risk`

- 输入：无公开参数；患者原始数据由后端上下文注入。
- 执行：调用 `predict_from_patient()`。
- 返回给 Agent：概率、阈值、标签、模型版本以及正负 Top 贡献特征。
- 隐私：不返回患者 ID、12 项原始值或全部贡献明细。

### `search_literature`

- 输入：`question` 和 `top_k`。
- 执行：调用 `LiteratureIndex.search()`。
- 返回给 Agent：稳定的 `source_id`、标题、文件名、页码、相关片段和相似度。
- 约束：问题不能为空，`top_k` 限制在 1 到 5。

## 5. Agent 执行流程

1. `/agent/analyze` 接收患者数据、研究问题和最大文献条数。
2. 后端建立一次只在当前请求内存在的 Agent 上下文。
3. DeepSeek根据任务选择 `predict_risk` 或 `search_literature`。
4. 后端用 Pydantic 校验工具名称和参数，并从白名单执行工具。
5. 工具输出经过允许列表脱敏后加入下一轮上下文。
6. DeepSeek可继续调用另一个工具，或返回最终 JSON。
7. 达到最大轮数仍未结束时，返回可识别的 Agent 上限错误。
8. 后端再次校验最终 JSON，并把结果、引用和工具轨迹返回网页。

单次请求最多执行 4 次工具调用，同一个工具和完全相同参数不允许重复执行。

## 6. 最终输出结构

最终结果包含：

- `answer_summary`：对本次科研问题的简要回答。
- `prediction_summary`：如调用预测工具，说明概率、阈值和主要贡献因素。
- `evidence_summary`：说明检索证据如何支持或限制回答。
- `citations`：只允许引用本次检索实际返回的 `source_id`。
- `tool_trace`：按顺序列出调用的工具名称和成功/失败状态，不记录原始患者数据。
- `research_disclaimer`：固定科研用途免责声明。
- `provider`、`model`：记录外部模型来源和模型名。

如果 Agent 没有调用某项工具，对应摘要明确写为“本次任务未调用该工具”，不得伪造预测或文献。

## 7. 文件变化

- 新增 `app/agent.py`：工具注册、工具执行、Agent 循环、结构化输出与错误类型。
- 修改 `app/deepseek.py`：增加工具调用请求和最终 Agent JSON 校验。
- 修改 `app/api.py`：增加请求模型、依赖注入和 `POST /agent/analyze`。
- 修改 `ui/dashboard.py`：增加“Agent 智能分析”区域与工具轨迹展示。
- 新增 `tests/test_agent.py`：覆盖工具白名单、隐私边界、循环上限、引用校验和成功流程。
- 修改 `tests/test_api.py`、`tests/test_dashboard.py`、`tests/test_deepseek.py`：覆盖新接口、网页请求与 DeepSeek 结构解析。
- 更新 `README.md` 和 `docs/demo-runbook.md`：补充运行和演示方法。

## 8. 错误处理

- 输入缺失或字段非法：HTTP 422，`invalid_agent_request`。
- 模型文件不存在：HTTP 503，`model_artifact_unavailable`。
- 文献索引不存在：HTTP 503，`literature_index_unavailable`。
- DeepSeek 密钥未配置：HTTP 503，`deepseek_not_configured`。
- DeepSeek 请求失败或结构不合法：HTTP 502，`agent_upstream_unavailable`。
- 未知工具、非法参数、重复调用或超过轮数：HTTP 502，`agent_execution_failed`。

所有错误均包含请求编号；服务端日志不记录 API Key、患者原始数据、完整提示词或完整文献正文。

## 9. 测试与完成标准

- 测试先行，每项行为先看到预期失败，再编写最小实现。
- 使用假 DeepSeek 决策器和临时模型/索引完成测试，不消耗真实 API 额度。
- 验证 DeepSeek 请求中不存在 `patient_id` 和患者原始特征值。
- 验证未知工具和非法参数不会被执行。
- 验证引用只能来自本次检索结果。
- 验证最大轮数和重复调用保护生效。
- 验证网页能展示总结、引用、工具轨迹和请求编号。
- 全量测试通过后，手动运行一次真实本地演示。

