# Agent Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a privacy-bounded DeepSeek agent that can choose and call the existing local prediction and literature-search capabilities, then return a validated evidence-backed result and visible tool trace.

**Architecture:** `app/deepseek.py` handles one provider turn and validates either tool calls or a final JSON answer. `app/agent.py` owns the allowlisted finite tool loop and injects patient data only into the local prediction tool. FastAPI exposes the runner through `/agent/analyze`, while Streamlit submits the current patient payload and renders summaries, citations, and the safe trace.

**Tech Stack:** Python 3.9, Pydantic, requests, FastAPI, Streamlit, unittest, existing scikit-learn model bundle and NumPy literature index.

**Spec:** `docs/superpowers/specs/2026-09-29-agent-tool-calling-design.md`

## Global Constraints

- Do not add LangChain or another agent framework.
- Never send patient ID or any of the 12 raw patient feature values to DeepSeek.
- Only `predict_risk` and `search_literature` may be executed.
- Allow at most 4 tool calls and reject an identical repeated call.
- Validate tool arguments, provider responses, final output, and cited source IDs with local code.
- Do not log API keys, patient values, complete prompts, or complete evidence text.
- Keep all output research-only and reject diagnosis, treatment, or medication guidance in prompts.

## Review Focus

- A malformed JSON argument string in a provider tool call must fail safely and execute nothing; covered in Task 1.
- A provider-requested unknown tool must be rejected before dispatch; covered in Task 2.
- Repeated identical calls or a fifth call must end with a bounded execution error; covered in Task 2.
- A final answer citing a source not returned in this run must be rejected; covered in Task 2.
- A request containing valid numeric values that resemble identifiers must still keep every raw patient field out of provider messages; covered in Task 2.

---

### Task 1: DeepSeek Agent Turn Protocol

**Files:**
- Modify: `app/deepseek.py`
- Modify: `tests/test_deepseek.py`

**Interfaces:**
- Consumes: existing `DEEPSEEK_CHAT_URL`, API-key resolution, `requests.post`, and `DeepSeekConfigurationError`.
- Produces: `AgentFinalAnswer`, `build_agent_turn_request(messages, tools) -> dict[str, Any]`, and `request_agent_turn(messages, tools, api_key=None, http_post=requests.post) -> dict[str, Any]`.

- [ ] **Step 1: Write failing provider-protocol tests**

Add tests named `test_agent_request_contains_only_supplied_messages_and_allowlisted_tools`, `test_request_agent_turn_parses_tool_calls`, `test_request_agent_turn_parses_valid_final_json`, and `test_request_agent_turn_rejects_malformed_tool_arguments`. Assert that parsed turns use exactly one of `kind="tool_calls"` or `kind="final"`, preserve provider call IDs, return decoded argument dictionaries, and never accept malformed arguments.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_deepseek.DeepSeekExplanationTests.test_agent_request_contains_only_supplied_messages_and_allowlisted_tools tests.test_deepseek.DeepSeekExplanationTests.test_request_agent_turn_parses_tool_calls tests.test_deepseek.DeepSeekExplanationTests.test_request_agent_turn_parses_valid_final_json tests.test_deepseek.DeepSeekExplanationTests.test_request_agent_turn_rejects_malformed_tool_arguments -v`

Expected: FAIL because the Agent protocol types and functions do not exist.

- [ ] **Step 3: Implement the minimal DeepSeek turn adapter**

Add strict Pydantic models for final fields `answer_summary`, `prediction_summary`, `evidence_summary`, `cited_source_ids`, and `research_disclaimer`. Build an OpenAI-compatible request with the supplied messages, tool schemas, `tool_choice="auto"`, disabled thinking, and no raw patient context. Parse either `message.tool_calls` or JSON `message.content`; convert envelope, HTTP, JSON, and validation failures to `DeepSeekRequestError`.

- [ ] **Step 4: Run focused tests and the existing DeepSeek suite**

Run: `python -m unittest tests.test_deepseek -v`

Expected: PASS with no real network call.

- [ ] **Step 5: Commit**

```powershell
git add app/deepseek.py tests/test_deepseek.py
git commit -m "feat: add DeepSeek agent turn protocol"
```

### Task 2: Allowlisted Agent Tool Loop

**Files:**
- Create: `app/agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: `request_agent_turn(messages, tools)`, `predict_from_patient(patient, artifact_path)`, and `LiteratureIndex.search(question, top_k)`.
- Produces: `AgentExecutionError`, `AgentLimitError`, `ResearchAgent.__init__(predictor, literature_index, turn_requester=request_agent_turn, max_tool_calls=4)`, and `ResearchAgent.run(patient, question, top_k=5) -> dict[str, Any]`.

- [ ] **Step 1: Write failing success-and-privacy tests**

In `tests/test_agent.py`, add `test_agent_calls_prediction_then_search_and_returns_grounded_result` and `test_agent_messages_never_contain_patient_id_or_raw_feature_values`. Use an injected fake turn requester that first selects `predict_risk`, then `search_literature`, then returns a final answer; assert ordered trace, mapped citations, and absence of every patient input value from serialized provider messages.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest tests.test_agent.AgentTests.test_agent_calls_prediction_then_search_and_returns_grounded_result tests.test_agent.AgentTests.test_agent_messages_never_contain_patient_id_or_raw_feature_values -v`

Expected: ERROR because `app.agent` does not exist.

- [ ] **Step 3: Implement tool schemas, safe projections, and the minimal loop**

Define the two tool schemas as constants. Inject the patient dictionary into the local prediction closure rather than into messages. Send only the research question initially; append only allowlisted prediction output or retrieved evidence after a tool executes. Map final `cited_source_ids` back to the evidence retrieved during this run and return `citations`, `tool_trace`, `provider`, and `model`.

- [ ] **Step 4: Run the two success tests and verify GREEN**

Run: the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Write failing safety-boundary tests**

Add `test_agent_rejects_unknown_tool`, `test_agent_rejects_invalid_search_arguments`, `test_agent_rejects_identical_repeated_call`, `test_agent_stops_after_four_tool_calls`, and `test_agent_rejects_unretrieved_citation`. Assert no forbidden tool executes and each case raises `AgentExecutionError` or `AgentLimitError` as appropriate.

- [ ] **Step 6: Run the safety tests and verify RED**

Run: `python -m unittest tests.test_agent -v`

Expected: FAIL on missing validation and limits.

- [ ] **Step 7: Implement dispatch validation and loop guards**

Use strict Pydantic argument models: `predict_risk` accepts no properties; `search_literature` requires a non-empty question and `top_k` from 1 through 5. Canonicalize each call as tool name plus sorted JSON arguments for duplicate detection. Reject an unknown tool, invalid arguments, a duplicate signature, a fifth tool call, or an unrecognized citation.

- [ ] **Step 8: Run the complete Agent test file**

Run: `python -m unittest tests.test_agent -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add app/agent.py tests/test_agent.py
git commit -m "feat: add bounded research agent loop"
```

### Task 3: FastAPI Agent Endpoint

**Files:**
- Modify: `app/api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `ResearchAgent.run(patient, question, top_k)` and the existing request-ID/error-response helpers.
- Produces: `AgentAnalyzeRequest` and `POST /agent/analyze`; `create_app(..., agent_service: Optional[object] = None)` supports dependency injection.

- [ ] **Step 1: Write failing endpoint tests**

Add `test_agent_analyze_returns_structured_result_from_injected_service`, `test_agent_analyze_uses_specific_validation_error`, `test_agent_analyze_maps_missing_resources`, and `test_agent_analyze_maps_provider_and_execution_failures`. Assert HTTP 200 for success, 422 `invalid_agent_request`, 503 resource/configuration codes, and 502 Agent failure codes, all with request IDs where applicable.

- [ ] **Step 2: Run endpoint tests and verify RED**

Run: `python -m unittest tests.test_api -v`

Expected: FAIL because `/agent/analyze` and injection do not exist.

- [ ] **Step 3: Implement lazy Agent construction and endpoint mapping**

Extend `PatientRequest` with `question` (2–500 characters) and `top_k` (1–5) in `AgentAnalyzeRequest`. Lazily load the same artifact and literature index already configured for other endpoints. Pass only patient fields to the Agent, and map known exceptions to the exact codes in the spec without logging sensitive payloads.

- [ ] **Step 4: Run API tests and verify GREEN**

Run: `python -m unittest tests.test_api -v`

Expected: PASS, with model-dependent tests still skipped when `PPMI_TEST_SOURCE_ROOT` is unset.

- [ ] **Step 5: Commit**

```powershell
git add app/api.py tests/test_api.py
git commit -m "feat: expose agent analysis API"
```

### Task 4: Streamlit Agent Experience

**Files:**
- Modify: `ui/dashboard.py`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `POST /agent/analyze` response fields from Task 3.
- Produces: `request_agent_analysis(payload, question, top_k) -> tuple[Optional[dict], Optional[str]]`, Agent error messages, and a rendered Agent section beneath an available prediction.

- [ ] **Step 1: Write failing dashboard helper tests**

Add `test_request_agent_analysis_posts_patient_question_and_top_k`, `test_format_api_error_explains_agent_failures`, and `test_agent_result_sections_preserve_fixed_order`. Assert a 90-second timeout, merged request JSON, request-ID display, and ordered headings for answer, prediction, evidence, and disclaimer.

- [ ] **Step 2: Run focused dashboard tests and verify RED**

Run: `python -m unittest tests.test_dashboard -v`

Expected: FAIL because the request helper and Agent formatting do not exist.

- [ ] **Step 3: Implement the helper and Streamlit section**

Require an existing patient payload, a non-empty research question, and explicit consent before calling the endpoint. Render the four summaries, citations with title/page/score/source ID, and a compact ordered tool trace containing only tool name and status.

- [ ] **Step 4: Run dashboard tests and verify GREEN**

Run: `python -m unittest tests.test_dashboard -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add ui/dashboard.py tests/test_dashboard.py
git commit -m "feat: add agent analysis dashboard"
```

### Task 5: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/demo-runbook.md`
- Modify: `理解项目.md`

**Interfaces:**
- Consumes: the completed endpoint and UI behavior.
- Produces: reproducible startup and demonstration instructions plus a learner-oriented explanation of the Agent loop.

- [ ] **Step 1: Update project documentation**

Document the Agent purpose, two tools, privacy boundary, four-call limit, `/agent/analyze` request/response role, UI steps, expected tool trace, and the distinction between an Agent-selected workflow and the existing fixed `/predict`, `/explain`, and `/literature/ask` endpoints.

- [ ] **Step 2: Run repository safety checks**

Run: `git diff --check` and `rg -n "sk-[A-Za-z0-9_-]{20,}" --glob '!.env' --glob '!*.joblib' .`

Expected: no whitespace error and no committed API key.

- [ ] **Step 3: Run the full automated suite**

Run: `python -m unittest discover -s tests -v`

Expected: all runnable tests PASS; only training-data-dependent tests may be skipped when `PPMI_TEST_SOURCE_ROOT` is unset.

- [ ] **Step 4: Run a local smoke test**

Start the existing one-click launcher, load example patient data, run Agent analysis with a research question, and confirm the page shows structured summaries, citations, tool trace, provider/model, and disclaimer without exposing raw patient values in logs.

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/demo-runbook.md '理解项目.md'
git commit -m "docs: explain agent tool workflow"
```
