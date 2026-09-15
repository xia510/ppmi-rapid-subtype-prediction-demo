# DeepSeek Explanations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a consent-gated DeepSeek interpretation path without moving local prediction responsibility to the LLM.

**Architecture:** FastAPI predicts locally and sends a de-identified output subset to DeepSeek. Streamlit calls the new endpoint only after user confirmation and presents the returned text alongside the deterministic model result.

**Tech Stack:** Python 3.9, requests, FastAPI, Streamlit, DeepSeek Chat Completions API.

**Spec:** `docs/plans/2026-09-15-deepseek-explanations-design.md`

## Global Constraints

- Read `DEEPSEEK_API_KEY` only from the API process environment.
- Do not send `patient_id` or raw 12-feature values to DeepSeek.
- Keep the original `/predict` endpoint and model outputs unchanged.
- Limit the external response to 400 tokens.

---

### Task 1: Implement a bounded DeepSeek client

**Files:**
- Create: `app/deepseek.py`
- Test: `tests/test_deepseek.py`

- [x] Build a de-identified prompt from only probability, threshold, labels, and contribution summaries.
- [x] Use DeepSeek Chat Completions with `deepseek-flash`, a 30-second timeout, and a 400-token output cap.
- [x] Test request content and response parsing through a fake HTTP client.

### Task 2: Add the FastAPI explanation boundary

**Files:**
- Modify: `app/api.py`
- Test: `tests/test_api.py`

- [x] Add `POST /explain` that runs local prediction before requesting an explanation.
- [x] Remove identifiers and raw values before the external boundary.
- [x] Map missing-key and provider failures to clear HTTP errors.

### Task 3: Expose opt-in UI and documentation

**Files:**
- Modify: `ui/dashboard.py`, `README.md`
- Create: `docs/plans/2026-09-15-deepseek-explanations-design.md`
- Test: `tests/test_dashboard.py`

- [x] Add a consent gate and an explanation button after normal prediction results.
- [x] Document environment-variable configuration and data-sharing boundaries.
- [x] Verify all project tests without a real API key or billable request.
