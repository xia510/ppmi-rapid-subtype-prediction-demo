# Evidence-backed Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and download a deterministic Markdown research report from the existing privacy-bounded Agent result and its verified literature citations.

**Architecture:** `app/report.py` validates an Agent result and renders a fixed Markdown document locally. FastAPI runs the existing `ResearchAgent` through `POST /report/generate`, then returns the Markdown and metadata. Streamlit calls this endpoint, previews the result, and exposes a Markdown download button.

**Tech Stack:** Python 3.9, Pydantic, FastAPI, Streamlit, unittest, existing ResearchAgent and requests client.

**Spec:** `docs/superpowers/specs/2026-09-29-evidence-report-design.md`

## Global Constraints

- The report is research-only and must not provide diagnosis, treatment, or medication advice.
- Never include the 12 raw patient feature values, API keys, complete prompts, or stack traces in the report.
- Render citations only from the Agent result's already validated `citations` list.
- Generate Markdown with local deterministic Python code; do not ask DeepSeek to format the complete report.
- Support only Markdown download in this phase; do not add PDF generation or report persistence.
- Keep `top_k` between 1 and 5 and preserve existing Agent call limits.

## Review Focus

- Citation text containing Markdown headings or newlines must not break fixed report sections; covered in Task 1.
- A malformed or empty Agent summary must fail before a report is returned; covered in Task 1.
- A report request with invalid question or `top_k` must return `invalid_report_request`; covered in Task 2.
- Provider, model, tool trace, and citation count must come from validated Agent output, not client-supplied report metadata; covered in Task 2.
- The UI must download the exact Markdown returned by the API without rebuilding it in the browser; covered in Task 3.

---

### Task 1: Deterministic Markdown Report Builder

**Files:**
- Create: `app/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: the dictionary returned by `ResearchAgent.run()` plus the original research question and optional patient ID.
- Produces: `ReportGenerationError`, `render_research_report(agent_result, question, patient_id=None, generated_at=None) -> dict[str, Any]`.

- [ ] **Step 1: Write failing report behavior tests**

Add tests for fixed section order, citation rendering/count, safe filename, Unicode output, missing/empty required summaries, and normalization of heading/newline characters inside citation fields. Assert the report contains only citation objects supplied in `agent_result` and does not include raw patient fields.

- [ ] **Step 2: Run report tests and verify RED**

Run: `python -m unittest tests.test_report -v`

Expected: ERROR because `app.report` does not exist.

- [ ] **Step 3: Implement the strict report model and renderer**

Use Pydantic models with forbidden extra fields for citations and tool trace. Accept the existing Agent result fields, normalize untrusted inline Markdown to single-line text, format scores consistently, derive a UTC timestamp when one is not injected, and return `filename`, `media_type`, `markdown`, `citation_count`, `tool_trace`, `provider`, and `model`.

- [ ] **Step 4: Run report tests and verify GREEN**

Run: `python -m unittest tests.test_report -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/report.py tests/test_report.py
git commit -m "feat: add evidence report renderer"
```

### Task 2: FastAPI Report Endpoint

**Files:**
- Modify: `app/api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `ResearchAgent.run(patient, question, top_k)` and `render_research_report(...)`.
- Produces: `ReportGenerateRequest`, injectable `report_renderer`, and `POST /report/generate`.

- [ ] **Step 1: Write failing endpoint tests**

Add tests for a successful injected Agent/result renderer flow, stripping `question` and `top_k` from the patient dictionary, HTTP 422 `invalid_report_request`, reuse of Agent resource/upstream error codes, and HTTP 502 `report_generation_failed`.

- [ ] **Step 2: Run focused API tests and verify RED**

Run: `python -m unittest tests.test_api.ReportApiTests -v`

Expected: FAIL because `/report/generate` does not exist.

- [ ] **Step 3: Implement request validation, endpoint, and error mapping**

Reuse the lazy Agent resolver. Call the injected/default renderer only after the Agent succeeds. Pass optional patient ID as report metadata, never pass raw feature values to the renderer, and preserve the existing safe request-ID response format.

- [ ] **Step 4: Run API tests and verify GREEN**

Run: `python -m unittest tests.test_api.ReportApiTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/api.py tests/test_api.py
git commit -m "feat: expose evidence report API"
```

### Task 3: Streamlit Report Preview and Download

**Files:**
- Modify: `ui/dashboard.py`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `POST /report/generate` response from Task 2.
- Produces: `request_research_report(patient, question, top_k) -> tuple[Optional[dict], Optional[str]]` and a report preview/download section.

- [ ] **Step 1: Write failing dashboard helper tests**

Test endpoint URL, merged JSON, 120-second timeout, exact returned Markdown preservation, and user-facing mappings for `invalid_report_request` and `report_generation_failed`.

- [ ] **Step 2: Run dashboard tests and verify RED**

Run: `python -m unittest tests.test_dashboard -v`

Expected: FAIL because report request handling does not exist.

- [ ] **Step 3: Implement report UI**

Below the Agent section, add report question and evidence-limit inputs, explicit external-AI consent, generation button, Markdown preview, citation count/provider/model metadata, and `st.download_button` using the exact API response bytes and filename.

- [ ] **Step 4: Run dashboard tests and verify GREEN**

Run: `python -m unittest tests.test_dashboard -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add ui/dashboard.py tests/test_dashboard.py
git commit -m "feat: add report download experience"
```

### Task 4: Documentation and Whole-project Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/demo-runbook.md`
- Modify: `理解项目.md`

**Interfaces:**
- Consumes: completed report endpoint and UI flow.
- Produces: reproducible report demo steps and learner-oriented explanation.

- [ ] **Step 1: Update documentation**

Document the report data flow, fixed sections, `/report/generate` contract, Markdown download steps, citation boundary, privacy boundary, and why deterministic local formatting is safer than free-form LLM report generation.

- [ ] **Step 2: Run safety and syntax checks**

Run: `git diff --check`, `python -m compileall -q app ui tests`, and a committed-files secret scan that excludes intentional `sk-test-` fixtures.

Expected: no whitespace, syntax, or secret findings.

- [ ] **Step 3: Run full automated suite**

Run: `python -m unittest discover -s tests -v`

Expected: all runnable tests PASS; only authorized training-data tests may be skipped when `PPMI_TEST_SOURCE_ROOT` is unset.

- [ ] **Step 4: Run local API smoke test**

Use injected/local services without consuming a real API key to call `/report/generate`; confirm HTTP 200, fixed sections, safe filename, verified citation count, and absence of raw patient values.

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/demo-runbook.md '理解项目.md'
git commit -m "docs: explain evidence report workflow"
```
