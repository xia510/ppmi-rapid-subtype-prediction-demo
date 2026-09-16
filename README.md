# PPMI Rapid Subtype Prediction Demo

一个将 PPMI 队列研究中的 Logistic Regression 模型整理为可复用预测工具的演示项目。

## 项目功能

- 从 JSON 读取一位受试者的 12 项原始特征；
- 使用训练集拟合的 `StandardScaler` 完成特征标准化；
- 使用经过 sigmoid 概率校准的 Logistic Regression 输出 Rapid 亚型概率；
- 按研究阈值输出演示性分类结果；
- 展示全部 12 项特征对该样本线性预测得分的贡献，并列出正向、负向贡献最大的特征。

## 目录说明

```text
app/                 预测与输入校验代码
artifacts/           本地导出的模型文件（不提交到 Git）
examples/            示例输入 JSON
outputs/             本地预测输出（不提交到 Git）
scripts/             从研究训练集导出模型的脚本
tests/               自动化测试
```

## 环境

- Python 3.9+
- pandas 2.3.3
- scikit-learn 1.5.0
- joblib 1.5.1

安装依赖：

```powershell
pip install -r requirements.txt
```

## 使用方式

先在本地从研究训练集导出模型：

```powershell
python scripts/export_model.py
```

再使用示例受试者数据完成预测：

```powershell
python app/predictor.py --input examples/sample_patient.json --artifact artifacts/model_bundle.joblib --output outputs/prediction.json
```

## FastAPI 服务

先在本地导出模型文件（该二进制文件不会提交到 Git）：

```powershell
python scripts/export_model.py
```

启动 API 服务：

```powershell
python -m uvicorn app.api:app --reload
```

然后在浏览器打开 `http://127.0.0.1:8000/docs`。页面会自动展示：

- `GET /health`：服务状态、模型文件是否可用；
- `POST /predict`：输入 12 项原始特征 JSON，返回校准概率、研究阈值与全部特征贡献度。

## Streamlit 预测网页

网页通过 HTTP 调用上面的 FastAPI 服务，不会重复训练或加载模型。请打开两个终端：

第一个终端启动后端：

```powershell
python scripts/export_model.py
python -m uvicorn app.api:app --reload
```

第二个终端启动网页：

```powershell
python -m streamlit run ui/dashboard.py
```

浏览器打开 `http://127.0.0.1:8501`。网页可加载示例数据、提交 12 项特征、展示后端状态、校准概率、研究阈值和贡献度。关闭后端时，网页会提示连接失败。

## DeepSeek 科研辅助解读

网页在完成本地预测后，可额外调用 DeepSeek 生成中文科研辅助解读。预测概率、阈值和贡献度仍完全由本地 Logistic Regression 模型计算；DeepSeek 不参与预测或分类。

调用前，网页会要求确认外部 API 调用。后端只向 DeepSeek 发送去标识化的模型输出摘要：概率、研究阈值、研究标签及 Top 正负贡献特征；不会发送 `patient_id` 或 12 项原始特征。

在**启动 FastAPI 的终端**中设置密钥。不要将密钥发到聊天、写进代码或提交到 Git：

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
$env:DEEPSEEK_MODEL = "deepseek-flash"
python -m uvicorn app.api:app --reload
```

随后在网页完成一次预测，勾选外部调用确认，点击“生成 DeepSeek 辅助解读”。接口为 `POST /explain`；它会先在本地预测，再调用 DeepSeek。当前默认模型为 `deepseek-flash`，接口地址为 `https://api.deepseek.com/chat/completions`。

DeepSeek 会被要求返回 JSON；本地 API 会校验“概率与阈值、特征贡献、科研使用说明”这三项非空文本后，才把结果返回给网页。

### 排错与请求编号

FastAPI 会在启动它的终端输出每次请求的安全日志：请求编号、路径、状态码和耗时；不会记录 API Key、Patient ID 或 12 项原始特征。网页出现错误时会显示“请求编号”。排查问题时只需提供该编号，不要发送密钥或原始受试者数据。

## 结果解释

`rapid_probability` 是校准后的 Rapid 亚型概率；`research_threshold` 是原研究流程中通过交叉验证得到的研究阈值。`contribution` 表示特征对逻辑回归线性得分的贡献，不能被解释为因果效应或临床建议。

## 免责声明

本项目仅用于科研复现与工程学习演示，不可用于临床诊断或医疗决策。

## Learning log

- 2026-09-14: Initialized the Git repository and pushed the project to GitHub.
- 2026-09-14: Practiced the Git command-line workflow.
- 2026-09-15: Added an optional DeepSeek research-explanation API flow.

