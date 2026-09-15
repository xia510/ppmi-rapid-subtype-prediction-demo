# Streamlit Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建调用现有 FastAPI 预测接口的 Streamlit 本地网页。

**Architecture:** `ui/dashboard.py` 提供输入表单和结果展示；它使用 HTTP 请求调用 `GET /health` 与 `POST /predict`，不复制模型计算。纯数据组装函数保持独立，供自动化测试验证。

**Tech Stack:** Python 3.9、Streamlit、requests、现有 FastAPI 服务。

---

### Task 1: 添加依赖并测试请求载荷组装

**Files:**
- Modify: `requirements.txt`
- Create: `ui/__init__.py`
- Create: `ui/dashboard.py`
- Create: `tests/test_dashboard.py`

**Step 1:** 在 `requirements.txt` 添加 `streamlit` 与 `requests`，安装后确认可导入。

**Step 2:** 先写失败测试，要求 `build_patient_payload` 保留 `patient_id`，把 12 项输入按固定顺序转为浮点数。

**Step 3:** 运行测试，预期因 `ui.dashboard` 不存在而失败。

**Step 4:** 实现常量 `FEATURE_NAMES` 和最小 `build_patient_payload`。

**Step 5:** 重跑测试，确认通过。

### Task 2: 实现 Streamlit 表单与 FastAPI 调用

**Files:**
- Modify: `ui/dashboard.py`
- Modify: `tests/test_dashboard.py`

**Step 1:** 写失败测试，要求 `format_api_error` 对 422 响应给出“检查必填特征”的可读提示。

**Step 2:** 实现 `health_status`、`request_prediction`、`format_api_error`；请求使用 10 秒超时。

**Step 3:** 实现页面：加载示例数据按钮、12 项数值输入、提交按钮、后端状态、概率与贡献度展示、免责声明。

**Step 4:** 运行网页辅助函数测试和全量测试。

### Task 3: 文档与手动验证

**Files:**
- Modify: `README.md`

**Step 1:** 记录双终端启动方式：先导出 artifact、启动 Uvicorn，再启动 Streamlit。

**Step 2:** 使用示例数据手动验证健康状态、预测结果、后端未启动时的错误提示。

**Step 3:** 运行完整测试，检查 Git 忽略规则，提交网页功能。
