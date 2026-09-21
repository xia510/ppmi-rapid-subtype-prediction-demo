# Portfolio Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a portfolio-grade README, reproducible demo runbook, interview script, and portable model-export command.

**Architecture:** Keep the GitHub landing page concise and route detailed operational and presentation material to focused documents. Make model export portable through explicit CLI arguments while preserving the existing Python function used by tests.

**Tech Stack:** Markdown, Mermaid, Python 3.9, argparse, unittest, FastAPI, Streamlit.

**Spec:** `docs/plans/2026-09-21-portfolio-final-design.md`

## Global Constraints

- Do not commit PPMI-derived CSV files, the model artifact, patient data, or an API key.
- State clearly that the project is a research/engineering demonstration, not a clinical device.
- Every documented command must match a verified command or an explicitly labelled optional step.

## Review Focus

- Source directory missing: export command exits nonzero and names the missing CSV.
- Artifact parent missing: export command creates it.
- DeepSeek key absent: local prediction remains usable; explanation is optional.
- Backend absent: Streamlit troubleshooting names the required service.
- Unauthorized visitor lacks training CSVs: README states the reproducibility limit before quick-start export instructions.

---

### Task 1: Portable model export

**Files:**
- Modify: `scripts/export_model.py`
- Modify: `tests/test_export_model.py`

- [ ] Write a failing CLI test that invokes `--source-root` and `--artifact`.
- [ ] Run `python -m unittest tests.test_export_model -v` and confirm the parser rejects the new arguments.
- [ ] Add `argparse` options and call the existing `export_model_bundle` function.
- [ ] Run the test again and confirm it passes.

### Task 2: Portfolio documentation

**Files:**
- Rewrite: `README.md`
- Create: `docs/demo-runbook.md`
- Create: `docs/project-presentation.md`

- [ ] Add project value, highlights, Mermaid architecture, directory tree, prerequisites, quick start, API table, model explanation, privacy boundary, tests, limitations, and FAQ to README.
- [ ] Add a two-terminal local demo sequence, expected checkpoints, request-ID troubleshooting, and shutdown steps to the runbook.
- [ ] Add one-minute and three-minute scripts plus likely interview follow-ups to the presentation document.

### Task 3: Verification

**Files:**
- Verify: all modified files

- [ ] Run the full 25-test suite.
- [ ] Execute export and CLI prediction commands into a fresh temporary directory.
- [ ] Start FastAPI without reload, verify `/health`, `/docs`, and `/predict`, then stop it.
- [ ] Scan tracked content for secret-like values and broken local-only documentation assumptions.
- [ ] Run `git diff --check` and review the final diff.
