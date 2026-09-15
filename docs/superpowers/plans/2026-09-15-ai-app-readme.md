# AI Application README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the repository README into a concise, reproducible AI application portfolio page for internship reviewers.

**Architecture:** Keep the application implementation unchanged. Document the existing flow from raw patient input through the exported scikit-learn artifact, FastAPI service, and Streamlit dashboard; add clear startup, verification, and interview-facing sections.

**Tech Stack:** Python 3.9+, scikit-learn, FastAPI, Uvicorn, Streamlit, unittest, GitHub.

**Spec:** `README.md`

## Global Constraints

- Describe only functionality present in this repository.
- Preserve the research-only, non-clinical disclaimer.
- Use the configured `pd-mci` Python interpreter in Windows examples.
- Do not commit generated `artifacts/*.joblib` or `outputs/` files.

---

### Task 1: Restructure the GitHub landing page

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: current repository paths, the API endpoints in `app/api.py`, and dashboard entry point `ui/dashboard.py`.
- Produces: runnable Windows instructions and an accurate AI-application portfolio overview.

- [x] **Step 1: Record the current public interface**

Read `app/api.py`, `ui/dashboard.py`, `requirements.txt`, and `tests/` to confirm the README names the exact endpoints, commands, and tested behaviours.

- [x] **Step 2: Rewrite README sections**

Add these sections in order: architecture flow, AI-application highlights, project layout, environment and startup, API contract, test command, result interpretation, interview description, and disclaimer.

- [x] **Step 3: Verify documentation references**

Run:

```powershell
rg "app/api.py|ui/dashboard.py|scripts/export_model.py|tests" README.md
```

Expected: every documented internal path exists in the repository.

- [x] **Step 4: Run the full regression suite**

Run:

```powershell
& "C:\Users\20284\anaconda3\envs\pd-mci\python.exe" -m unittest discover -s tests -v
```

Expected: all tests pass; README work must not change runtime behaviour.

- [x] **Step 5: Commit**

```powershell
git add README.md docs/superpowers/plans/2026-09-15-ai-app-readme.md
git commit -m "docs: polish README for AI application portfolio"
```
