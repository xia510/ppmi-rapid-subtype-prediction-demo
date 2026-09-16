# Structured LLM Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return DeepSeek research explanations as a validated three-field JSON object and render the fields separately in Streamlit.

**Architecture:** The existing Chat Completions request asks DeepSeek JSON mode for a fixed response shape. `app.deepseek` parses its JSON string and validates it locally with Pydantic before the existing `/explain` endpoint returns it. Streamlit uses the returned named fields rather than a free-form paragraph.

**Tech Stack:** Python 3.9, requests, FastAPI, Pydantic, Streamlit, unittest.

**Spec:** `docs/plans/2026-09-16-structured-llm-output-design.md`

## Global Constraints

- Keep the current Chat Completions endpoint, `deepseek-flash` default, non-thinking mode, and no new dependency.
- DeepSeek receives only the existing de-identified derived model-output summary.
- Keep the user consent checkbox and the local `/explain` boundary.
- Never call a real external DeepSeek API in automated tests.

---

### Task 1: Validate the structured provider response

**Files:**
- Modify: `tests/test_deepseek.py`
- Modify: `app/deepseek.py`

**Interfaces:**
- Produces: `request_interpretation(prediction, api_key, http_post) -> dict[str, str]` with `probability_summary`, `contribution_summary`, and `research_disclaimer`.
- Produces: `DeepSeekRequestError` for malformed, missing, or blank structured output.

- [ ] **Step 1: Write the failing test**

```python
def test_request_interpretation_parses_the_required_json_fields(self):
    result = request_interpretation(PREDICTION, api_key="test-key", http_post=fake_post)
    self.assertEqual(result["probability_summary"], "概率低于研究阈值。")
    self.assertEqual(result["contribution_summary"], "quip 提高线性得分。")
    self.assertEqual(result["research_disclaimer"], "仅供科研演示，不构成临床建议。")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_deepseek -v`

Expected: FAIL because the current fake response is plain text and the returned object has only `text`.

- [ ] **Step 3: Write minimal implementation**

Add `response_format = {"type": "json_object"}`. Update the prompt with the word `JSON` and an exact three-key example. Parse `message.content` with `json.loads`, validate the three nonblank string fields with Pydantic, and wrap JSON/Pydantic errors as `DeepSeekRequestError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_deepseek -v`

Expected: PASS with no network call.

### Task 2: Carry the named fields through API and dashboard

**Files:**
- Modify: `tests/test_api.py`
- Modify: `tests/test_dashboard.py`
- Modify: `ui/dashboard.py`

**Interfaces:**
- Consumes: `/explain` response `interpretation` containing the three required summaries.
- Produces: labelled Streamlit sections for probability, contribution, and research disclaimer.

- [ ] **Step 1: Write the failing tests**

```python
def test_explain_returns_structured_interpretation(self):
    self.assertEqual(response.json()["interpretation"]["probability_summary"], "概率说明。")

def test_request_explanation_returns_structured_interpretation(self):
    self.assertEqual(result["interpretation"]["contribution_summary"], "贡献说明。")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_api tests.test_dashboard -v`

Expected: FAIL because the fakes and UI still use `text`.

- [ ] **Step 3: Write minimal implementation**

Render `probability_summary`, `contribution_summary`, and `research_disclaimer` under three Chinese headings. Update injected API and dashboard fake data from `text` to the new fields. Retain provider/model metadata and the application-level disclaimer.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest tests.test_api tests.test_dashboard -v`

Expected: PASS with no network call.

### Task 3: Document and verify the finished feature

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-16-structured-llm-output-design.md`
- Modify: `docs/superpowers/plans/2026-09-16-structured-llm-output.md`

**Interfaces:**
- Consumes: completed structured response contract.
- Produces: a README explanation that DeepSeek output is JSON-mode requested and locally validated.

- [ ] **Step 1: Update README**

Add one concise sentence to the DeepSeek section: “DeepSeek is asked for JSON output, and the local API validates the required three research-only text fields before returning them to the page.”

- [ ] **Step 2: Run full verification**

Run: `& 'C:\Users\20284\anaconda3\envs\pd-mci\python.exe' -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 3: Perform manual local verification**

Start FastAPI and Streamlit in separate terminals, submit the demonstration input, check the consent box, and click the explanation button. Confirm all three Chinese sections appear. Do not put an API key in source code or the repository.
