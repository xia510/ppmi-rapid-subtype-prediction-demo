# API Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-safe terminal request logging and stable categorized API error responses.

**Architecture:** FastAPI middleware assigns and returns a request ID for every response. API helpers return a common error JSON shape for expected failures, while the dashboard formats that shape into a concise Chinese message and request ID.

**Tech Stack:** Python 3.9, FastAPI, Starlette middleware, Streamlit, unittest.

**Spec:** `docs/plans/2026-09-16-api-observability-design.md`

## Global Constraints

- Log only request ID, HTTP method, route path, status code, duration, and safe error category.
- Never log API keys, Patient ID, raw feature values, or full prediction output.
- Do not add a dependency, write log files, or call an external logging service.
- Preserve the existing successful `/predict` and `/explain` response shapes.

---

### Task 1: Add request IDs and stable API errors

**Files:**
- Modify: `tests/test_api.py`
- Modify: `app/api.py`

**Interfaces:**
- Produces: response header `X-Request-ID` for every API response.
- Produces: error JSON with `error.code`, `error.message`, and `request_id`.

- [ ] **Step 1: Write failing tests**

```python
def test_health_includes_a_request_id_header(self):
    response = TestClient(app).get("/health")
    self.assertRegex(response.headers["X-Request-ID"], r"^[0-9a-f]{12}$")

def test_explain_without_a_key_returns_categorized_error(self):
    response = client.post("/explain", json=valid_patient)
    self.assertEqual(response.status_code, 503)
    self.assertEqual(response.json()["error"]["code"], "deepseek_not_configured")
    self.assertIn("request_id", response.json())
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_api -v`

Expected: FAIL because request IDs and the structured error object do not exist.

- [ ] **Step 3: Implement the minimum behavior**

Add an HTTP middleware using `uuid.uuid4().hex[:12]` and `time.perf_counter()`. Add `X-Request-ID` on the response and log a single safe completion line. Add an error-response helper returning `JSONResponse` with the documented contract. Use codes `model_artifact_unavailable`, `invalid_input`, `deepseek_not_configured`, and `deepseek_unavailable` in the existing endpoint failure branches.

- [ ] **Step 4: Run tests and verify they pass**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_api -v`

Expected: PASS with no model input or secret in assertions.

### Task 2: Render categorized errors in the dashboard

**Files:**
- Modify: `tests/test_dashboard.py`
- Modify: `ui/dashboard.py`

**Interfaces:**
- Consumes: `{ "error": { "code": str, "message": str }, "request_id": str }`.
- Produces: concise Chinese error text ending in `请求编号：<id>`.

- [ ] **Step 1: Write a failing test**

```python
def test_format_api_error_includes_safe_request_id(self):
    message = format_api_error(
        503,
        {"error": {"code": "deepseek_not_configured", "message": "ignored"}, "request_id": "abc123def456"},
    )
    self.assertIn("DEEPSEEK_API_KEY", message)
    self.assertIn("请求编号：abc123def456", message)
```

- [ ] **Step 2: Run the dashboard tests and verify they fail**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_dashboard -v`

Expected: FAIL because current formatting accepts only FastAPI's old `detail` field.

- [ ] **Step 3: Implement the minimum behavior**

Pass the complete JSON error payload to `format_api_error`. Map the four safe error codes to existing user-facing Chinese messages and append the request ID when present. Do not render raw server messages.

- [ ] **Step 4: Run dashboard tests and verify they pass**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_dashboard -v`

Expected: PASS without an HTTP server.

### Task 3: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-16-api-observability-design.md`
- Modify: `docs/superpowers/plans/2026-09-16-api-observability.md`

- [ ] **Step 1: Update README**

Add a troubleshooting note explaining that the terminal uses request IDs for diagnosis and that users should provide only the request ID, never API keys or raw patient data.

- [ ] **Step 2: Run full verification**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest discover -s tests -v`

Expected: all tests PASS.
