# Configuration Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model-artifact and API-base-URL configuration environment-driven and document safe secret handling.

**Architecture:** A small `app.settings` module reads optional environment variables and supplies defaults. FastAPI consumes the model path setting, while Streamlit consumes the API base URL setting. `.env.example` documents names but is never loaded automatically.

**Tech Stack:** Python 3.9, os.environ, pathlib, FastAPI, Streamlit, unittest.

**Spec:** `docs/plans/2026-09-16-config-management-design.md`

## Global Constraints

- Do not read, create, print, or commit a real API key or `.env` file.
- Keep `.env` ignored and add no dependency.
- Preserve all existing defaults and successful API response shapes.

---

### Task 1: Add and test environment-backed settings

**Files:**
- Create: `app/settings.py`
- Modify: `app/api.py`
- Modify: `ui/dashboard.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests**

```python
def test_model_artifact_path_uses_project_default_when_unset(self):
    self.assertEqual(model_artifact_path(), PROJECT_DIR / "artifacts" / "model_bundle.joblib")

def test_api_base_url_uses_environment_override(self):
    with patch.dict(os.environ, {"PPMI_API_BASE_URL": "http://api.example:9000"}):
        self.assertEqual(api_base_url(), "http://api.example:9000")
```

- [ ] **Step 2: Verify red**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_settings -v`

Expected: FAIL because `app.settings` does not exist.

- [ ] **Step 3: Implement minimum behavior**

Create `model_artifact_path()` and `api_base_url()` using `os.getenv`. Replace the FastAPI default artifact constant and Streamlit hard-coded URL with these functions.

- [ ] **Step 4: Verify green**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_settings tests.test_api tests.test_dashboard -v`

Expected: PASS.

### Task 2: Document the safe configuration workflow

**Files:**
- Create: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_settings.py`

- [ ] **Step 1: Add tests**

```python
def test_env_example_contains_no_secret_value(self):
    text = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
    self.assertIn("DEEPSEEK_API_KEY=", text)
    self.assertNotIn("sk-", text)
```

- [ ] **Step 2: Verify red, then add `.env.example`**

Use only documented names and safe defaults. Explain in README that PowerShell environment variables are set per terminal and `.env.example` is a template, not an active configuration file.

- [ ] **Step 3: Run full verification**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest discover -s tests -v`

Expected: all tests PASS.
