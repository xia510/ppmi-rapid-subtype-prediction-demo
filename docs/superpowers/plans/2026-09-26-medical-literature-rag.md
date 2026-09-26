# Medical Literature RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local PDF literature RAG workflow with grounded DeepSeek answers and traceable citations to the existing PPMI demo.

**Architecture:** Build a local index from page-aware PDF chunks and normalized embeddings, retrieve evidence with cosine similarity, and send only bounded evidence to a structured DeepSeek answerer. FastAPI exposes the workflow and Streamlit renders answers and citations.

**Tech Stack:** Python 3.9, PyMuPDF, sentence-transformers, NumPy, Pydantic, FastAPI, Streamlit, unittest

**Spec:** `docs/plans/2026-09-26-medical-literature-rag-design.md`

## Global Constraints

- Keep PDF documents, generated vectors and metadata out of Git.
- Do not send patient identifiers, raw model features or training data to DeepSeek.
- Tests must inject deterministic embeddings and fake HTTP responses.
- First release uses local retrieval only; no PubMed automation or Agent.
- Answers are research demonstrations, not clinical advice.

## Review Focus

- Empty or image-only PDFs must not create a misleading usable index.
- Chunk overlap must never cause an infinite loop.
- Query embeddings with a different dimension must fail clearly.
- DeepSeek citations outside retrieved source IDs must be rejected.
- Missing local index must produce a stable API error instead of a traceback.

---

### Task 1: Document ingestion and vector index

**Files:**
- Create: `app/rag.py`
- Create: `scripts/build_literature_index.py`
- Create: `tests/test_rag.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `build_literature_index(source_dir, index_dir, embedder, chunk_size, overlap)` and `LiteratureIndex.load(index_dir, embedder).search(question, top_k)`.

- [ ] Write tests for page-aware chunks, persisted metadata, ranked retrieval, empty PDFs and dimension mismatch.
- [ ] Run `python -m unittest tests.test_rag -v` and verify the missing module fails.
- [ ] Implement the minimum ingestion and retrieval code.
- [ ] Run the RAG tests and the complete suite.

### Task 2: Grounded structured DeepSeek answer

**Files:**
- Modify: `app/deepseek.py`
- Modify: `app/rag.py`
- Modify: `tests/test_deepseek.py`
- Modify: `tests/test_rag.py`

**Interfaces:**
- Produces: `request_grounded_answer(question, evidence, ...)` and `LiteratureRAG.ask(question, top_k)`.

- [ ] Write failing tests for evidence-only prompts, valid citations and unknown citation rejection.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement structured answer validation and RAG orchestration.
- [ ] Run focused tests and the complete suite.

### Task 3: FastAPI and Streamlit integration

**Files:**
- Modify: `app/api.py`
- Modify: `app/settings.py`
- Modify: `ui/dashboard.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Produces: `POST /literature/ask`, RAG health metadata and a Streamlit literature question panel.

- [ ] Write failing API and dashboard helper tests.
- [ ] Verify the tests fail for missing behavior.
- [ ] Implement dependency-injected API handling and frontend rendering.
- [ ] Run focused tests and the complete suite.

### Task 4: Documentation and reproducible verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Create: `knowledge_base/README.md`

**Interfaces:**
- Documents: dependency installation, index building, startup, privacy boundary and troubleshooting.

- [ ] Add setup and usage documentation without committing source PDFs or generated indexes.
- [ ] Install pinned compatible dependencies in `pd-mci` and build a temporary test index from a generated PDF.
- [ ] Run the full test suite and a local API smoke test.
- [ ] Review `git diff`, commit the feature branch and report the exact verification evidence.
