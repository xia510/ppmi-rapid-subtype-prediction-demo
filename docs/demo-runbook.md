# 本地演示手册

这份手册用于面试演示或项目验收。完整演示约 8～12 分钟，需要两个 PowerShell 终端。

## 0. 演示前检查

确认当前目录与分支：

```powershell
cd "C:\path\to\ppmi-rapid-subtype-prediction-demo"
git status
```

确认 Python 环境：

```powershell
python --version
```

预期为 Python 3.9.x。不要在演示画面或聊天中展示真实 DeepSeek Key。

## 1. 导出模型

```powershell
python scripts\export_model.py `
  --source-root "C:\path\to\authorized\research-data" `
  --artifact "artifacts\model_bundle.joblib"
```

预期输出：

```text
Model bundle saved to: artifacts\model_bundle.joblib
```

模型包包含 12 项特征顺序、训练集标准化器、基础模型、校准模型、研究阈值和版本元数据。

## 2. 构建本地医学文献索引

先把有权使用的开放获取 PDF 放入 `knowledge_base\source_documents\`，不要放入患者病历或含个人身份信息的材料。

```powershell
python scripts\build_literature_index.py
```

预期输出包括文档数和切块数。首次运行会下载向量模型；后续替换文献时重新运行即可。

## 3. 启动 FastAPI（终端 1）

需要展示 DeepSeek 时，在这个终端设置密钥；只演示本地预测时可以跳过第一行。

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
python -m uvicorn app.api:app --reload
```

检查：

1. 打开 `http://127.0.0.1:8000/health`；
2. 确认 `status` 为 `ok`；
3. 确认 `model_artifact_available` 为 `true`；
4. 确认 `literature_index_available` 为 `true`；
5. 打开 `http://127.0.0.1:8000/docs`，展示包括 `/agent/analyze` 在内的五个接口。

## 4. 启动 Streamlit（终端 2）

```powershell
cd "C:\path\to\ppmi-rapid-subtype-prediction-demo"
python -m streamlit run ui\dashboard.py
```

浏览器打开 `http://127.0.0.1:8501`。页面顶部应显示“后端已连接，模型文件可用”。

## 5. 演示本地预测

1. 点击“加载示例数据”；
2. 点击“开始预测”；
3. 展示 Rapid 概率、研究阈值和研究标签；
4. 展示 Top 3 正向/负向贡献；
5. 展开全部 12 项特征贡献度；
6. 强调贡献度描述线性得分，不代表因果关系。

讲解重点：概率来自校准模型；贡献度来自基础模型的 `标准化值 × 系数`，两个模型职责不同但使用相同训练特征和标准化器。

## 6. 演示 DeepSeek 辅助解读

1. 阅读页面的数据发送说明；
2. 勾选外部 API 调用确认；
3. 点击“生成 DeepSeek 辅助解读”；
4. 展示“概率与阈值、特征贡献说明、科研使用说明”三个固定区域。

说明：后端会本地重新预测，只把去标识化摘要发给 DeepSeek。DeepSeek 返回 JSON 后还要经过 Pydantic 校验；它不参与概率计算。

## 7. 演示医学文献 RAG

1. 在“帕金森病医学文献助手”输入不含个人信息的一般性科研问题；
2. 选择检索片段数并勾选外部调用确认；
3. 点击“检索文献并生成回答”；
4. 展示回答、证据局限，以及每条引用的文献名、PDF 文件、页码、相似度和原文片段；
5. 强调流程是“本地检索证据在前，DeepSeek 归纳在后”，且后端拒绝虚构来源编号。

## 8. 演示 Agent 工具调用

1. 完成一次本地预测；
2. 在“Agent 智能分析”输入“结合模型贡献因素和本地文献，说明结果的科研依据与局限”；
3. 将证据数量设为 3，勾选外部调用确认；
4. 点击“运行 Agent 智能分析”；
5. 展示回答、预测摘要、证据摘要、文献引用和工具轨迹；
6. 强调 DeepSeek只选择工具，本地代码才真正执行预测与检索；调用次数最多 4 次，原始患者值不会发送给 DeepSeek。

## 9. 演示请求编号与排错

可在未配置密钥的测试终端触发一次 `/explain`，网页会显示安全中文提示和 12 位请求编号。FastAPI 终端会出现类似：

```text
api_error request_id=a1b2c3d4e5f6 code=deepseek_not_configured
```

请求编号用于把网页错误与后端日志对应起来。日志不记录 API Key、Patient ID 或 12 项原始特征。

## 10. 结束演示

分别在两个终端按 `Ctrl + C` 停止 Streamlit 和 FastAPI。不要关闭一个终端后在同一窗口同时启动两个长期运行服务。

## 常见故障顺序

1. 访问 `/health`：判断 FastAPI 是否启动；
2. 查看 `model_artifact_available`：判断模型包是否存在；
3. 检查网页顶部状态：判断 Streamlit 能否连接 API；
4. 查看 HTTP 状态码和错误代码：区分 422、502、503；
5. 使用 `request_id` 对照终端日志；
6. 最后才检查网络和 DeepSeek 余额。
