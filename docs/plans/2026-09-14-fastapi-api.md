# FastAPI Prediction API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有命令行预测工具包装为可通过 HTTP 调用的 FastAPI 服务。

**Architecture:** 新建 `app/api.py`，由 FastAPI 接收 JSON 请求并调用 `app.predictor` 中可复用的预测函数。模型仍由训练集标准化器、校准 Logistic Regression 和原有 12 项特征顺序驱动；API 不训练模型，也不改变研究阈值。

**Tech Stack:** Python 3.9、FastAPI、Uvicorn、Pydantic、现有 pandas/scikit-learn/joblib。

---

### Task 1: 安装 API 依赖并实现健康检查

**Files:**
- Modify: `requirements.txt`
- Create: `app/api.py`
- Create: `tests/test_api.py`

**Step 1: 添加运行依赖**

在 `requirements.txt` 新增：

```text
fastapi
uvicorn[standard]
```

**Step 2: 写会失败的健康检查测试**

```python
from fastapi.testclient import TestClient
from app.api import app


def test_health_returns_service_metadata():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

**Step 3: 运行测试，确认因 `app.api` 不存在而失败**

```powershell
& "C:\Users\20284\anaconda3\envs\pd-mci\python.exe" -m unittest tests/test_api.py -v
```

**Step 4: 最小实现**

在 `app/api.py` 创建 FastAPI 实例和 `GET /health`，返回服务状态、模型版本与固定特征数量。

**Step 5: 重新运行测试，确认通过**

### Task 2: 从文件预测重构为内存预测

**Files:**
- Modify: `app/predictor.py`
- Modify: `tests/test_predictor.py`

**Step 1: 写会失败的测试**

测试一个 Python `dict` 可直接得到与 `predict_from_json` 相同结构的预测结果，不写入 JSON 文件。

**Step 2: 运行测试，确认函数尚不存在而失败。**

**Step 3: 最小实现**

提取 `predict_from_patient(patient, artifact_path)`：加载 bundle、校验 12 项原始特征、标准化、用校准模型算概率、用 `base_model` 生成全部贡献度。原 `predict_from_json` 改为读取文件后调用它，再仅负责写输出文件。

**Step 4: 运行全部既有预测测试与新增测试。**

### Task 3: 实现 `POST /predict`

**Files:**
- Modify: `app/api.py`
- Modify: `tests/test_api.py`

**Step 1: 写会失败的 API 测试**

使用真实临时导出 artifact 与 `TestClient`：

```python
response = client.post("/predict", json=valid_patient)
assert response.status_code == 200
assert 0.0 <= response.json()["rapid_probability"] <= 1.0
assert len(response.json()["all_feature_contributions"]) == 12
```

再测试缺少 `LEDD` 时返回 HTTP 422 或带有明确错误信息的 HTTP 400。

**Step 2: 运行测试，确认 `/predict` 尚不存在而失败。**

**Step 3: 最小实现**

定义清晰的 Pydantic 请求模型，包含 `patient_id` 与 12 项数值特征。端点调用 `predict_from_patient`，将 `InputValidationError` 转成客户端可理解的 422 响应。不得把 `joblib` artifact 或训练数据放进 Git。

**Step 4: 运行 API 测试与所有测试。**

### Task 4: 文档、手动演示与提交

**Files:**
- Modify: `README.md`

**Step 1: 补充启动命令**

```powershell
uvicorn app.api:app --reload
```

说明浏览器访问 `http://127.0.0.1:8000/docs`，以及启动前需在本地导出 `artifacts/model_bundle.joblib`。

**Step 2: 运行完整自动化测试**

```powershell
& "C:\Users\20284\anaconda3\envs\pd-mci\python.exe" -m unittest discover -s tests -v
```

**Step 3: 手动验证**

启动 Uvicorn，访问 `/health` 和 `/docs`，用示例 JSON 调用 `/predict`。

**Step 4: 提交功能分支**

```powershell
git add requirements.txt app/api.py app/predictor.py tests/test_api.py tests/test_predictor.py README.md
git commit -m "feat: add FastAPI prediction service"
```
