# 带文献依据的最终报告设计

## 1. 目标

在现有本地预测、医学文献 RAG 和受限制 Agent 之上，增加一份可下载的 Markdown 科研报告。用户提交患者数据、研究问题和文献条数后，后端运行现有 Agent，再用本地确定性代码把已校验结果整理成固定报告；DeepSeek 不直接控制 Markdown 结构。

## 2. 核心原则

- 报告仅用于科研演示，不用于诊断、治疗或用药决策。
- 患者 12 项原始特征不写入报告，也不发送给 DeepSeek。
- 报告只能引用本次 Agent 实际检索并校验过的 `citations`。
- Markdown 由本地 Python 模板生成，避免大模型输出格式漂移。
- 不新增 PDF、数据库、用户账户或报告历史，保持本阶段范围可控。
- 不把 API Key、完整提示词或异常堆栈放入报告。

## 3. 数据流

1. Streamlit 将当前患者输入、研究问题和 `top_k` 发送到 `POST /report/generate`。
2. FastAPI 复用 `ResearchAgent.run(patient, question, top_k)`。
3. Agent 在既有安全边界内选择本地预测与文献检索工具，并返回经过 Pydantic 和引用白名单校验的结构化结果。
4. `app/report.py` 验证必要字段，并将结构化结果渲染成固定 Markdown。
5. API 返回 `filename`、`media_type`、`markdown`、`citation_count` 和安全工具轨迹。
6. Streamlit 预览报告，并通过下载按钮把同一份 Markdown 保存到本地。

## 4. 报告结构

固定章节按以下顺序生成：

1. 标题与生成时间
2. 研究问题
3. 综合结论
4. 模型预测摘要
5. 文献证据摘要
6. 主要参考证据
7. 工具执行记录
8. 科研用途声明

参考证据包含标题、页码、来源文件、`source_id`、相关性分数和经过规范化的证据片段。Markdown 控制字符和换行需要安全规范化，防止内容破坏报告结构。

## 5. 代码边界

- 新增 `app/report.py`：严格报告输入模型、Markdown 转义/规范化、文件名生成和报告渲染。
- 修改 `app/api.py`：新增 `ReportGenerateRequest`、报告服务注入和 `POST /report/generate`。
- 修改 `ui/dashboard.py`：新增报告请求、预览和 `.md` 下载按钮。
- 新增 `tests/test_report.py`，并扩展 API、Dashboard 测试。
- 更新 `README.md`、`docs/demo-runbook.md` 和 `理解项目.md`。

## 6. API 契约

请求字段与 `/agent/analyze` 相同：12 项患者特征、可选 `patient_id`、`question` 和 1–5 的 `top_k`。

成功响应：

- `filename`：只含安全 ASCII 字符的 `.md` 文件名。
- `media_type`：固定为 `text/markdown; charset=utf-8`。
- `markdown`：完整报告文本。
- `citation_count`：实际写入报告的引用数量。
- `tool_trace`：只包含工具名和状态。
- `provider`、`model`：记录生成分析所用模型。

错误沿用 Agent 资源和上游错误，并增加：

- HTTP 422 `invalid_report_request`：输入字段、问题或 `top_k` 非法。
- HTTP 502 `report_generation_failed`：Agent 结果结构无法形成安全报告。

所有错误继续返回请求编号。

## 7. 测试与完成标准

- 报告章节顺序固定，中文内容和 Unicode 可下载。
- 空摘要、未知字段结构或非法引用结构会被拒绝。
- 引用数量与 Agent 已校验的 `citations` 完全一致，不从模型自由文本提取引用。
- 报告不包含患者 12 项原始值、API Key 或未检索来源。
- API 正确映射输入、资源、DeepSeek、Agent 和报告渲染错误。
- 网页请求 `/report/generate`，显示预览并提供 Markdown 下载。
- 全量自动测试、语法编译、密钥扫描和本地接口烟雾测试通过。
