# Project Understanding Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a detailed Chinese guide that teaches the current project in its real execution order.

**Architecture:** Add one root-level learning document and leave application behavior unchanged. Derive every explanation from the current source and tests, then validate coverage with repository searches and the existing test suite.

**Tech Stack:** Markdown, Python 3.9, scikit-learn, FastAPI, Streamlit, Pydantic, requests, unittest

**Spec:** `docs/plans/2026-09-21-project-understanding-design.md`

## Global Constraints

- Create `理解项目.md` at the repository root.
- Explain the current code in execution order and do not change runtime code.
- Do not include API keys, sensitive data, or machine-specific personal paths.
- Preserve the research-only and non-clinical boundary.

## Review Focus

- A beginner must be able to distinguish model export from runtime prediction.
- The guide must accurately separate calibrated probability, base-model contribution, and DeepSeek wording.
- File relationships and HTTP data flow must match current function calls.
- Configuration and two-terminal startup behavior must not imply `.env` auto-loading.
- Tests must be explained as evidence, including the authorized-data skip mechanism.

---

### Task 1: Write and verify the project understanding guide

**Files:**
- Create: `理解项目.md`
- Test: existing `tests/` suite and Markdown coverage searches

**Interfaces:**
- Consumes: current files under `scripts/`, `app/`, `ui/`, `examples/`, `tests/`, plus root configuration files.
- Produces: a standalone learning path with source-file and function explanations.

- [ ] **Step 1: Confirm the guide is absent**

Run: `Test-Path -LiteralPath '理解项目.md'`

Expected: `False`.

- [ ] **Step 2: Create the complete guide**

Write the sections defined in the design, using selective code excerpts, object-shape examples, call-chain diagrams, misconceptions, debugging guidance and exercises.

- [ ] **Step 3: Check required coverage**

Run searches for every production module, every test module, `model_bundle.joblib`, the standardization/contribution formulas, `/predict`, `/explain`, request IDs and privacy boundaries.

Expected: every required topic appears in `理解项目.md`.

- [ ] **Step 4: Check Markdown and secrets**

Run: `git diff --check` and scan changed files for `sk-`.

Expected: no whitespace errors and no secret-like text.

- [ ] **Step 5: Run the full regression suite**

Run with `PPMI_TEST_SOURCE_ROOT` set to the authorized source directory: `python -m unittest discover -s tests -v`.

Expected: 26 tests pass.

- [ ] **Step 6: Commit**

Commit message: `docs: add detailed project understanding guide`.
