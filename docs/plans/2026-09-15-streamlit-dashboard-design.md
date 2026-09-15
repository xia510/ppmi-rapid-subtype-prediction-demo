# Streamlit Dashboard Design

## Purpose

为 PPMI Rapid 亚型研究模型提供一个面向演示者的本地网页。网页不训练模型，也不直接读取 `model_bundle.joblib`；它只收集 12 项原始特征并调用现有 FastAPI `POST /predict`。

## User flow

1. 页面启动时请求 `GET /health`，显示后端和模型 artifact 是否可用。
2. 用户可点击“加载示例数据”，或手动填写受试者编号与 12 项原始特征。
3. 用户提交表单后，Streamlit 将 JSON 发送到 `http://127.0.0.1:8000/predict`。
4. 页面展示校准概率、研究阈值、相对阈值的标签、正负贡献度 Top 3 和全部 12 项贡献度。
5. 后端未启动、模型文件缺失或输入无效时，页面显示可操作的错误信息。

## Architecture

```text
Streamlit dashboard (ui/dashboard.py)
       |  HTTP JSON
       v
FastAPI (/health, /predict)
       v
predict_from_patient + model_bundle.joblib
```

## Visual direction

采用“临床研究报告页”风格：深海军蓝标题区、暖白内容区、克制的青绿色状态强调。输入区与结果区清晰分离，避免将研究输出包装成诊断结论。页面始终显示科研演示免责声明。
