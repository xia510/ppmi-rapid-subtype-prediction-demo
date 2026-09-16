# Structured LLM Output Design

## Goal

Replace the free-form DeepSeek explanation with three named Chinese text fields so the API and Streamlit page can display predictable research-only content.

## Options considered

1. Keep plain text and split it with headings on the page. This is simple, but a model can omit or reorder headings, so the program cannot reliably tell which sentence belongs where.
2. Use DeepSeek JSON mode and validate the returned JSON with a local Pydantic model. This is the selected approach: the provider enforces valid JSON, while this application enforces the exact three fields and non-empty strings it needs.
3. Use a JSON Schema capable endpoint. That adds API migration complexity and is unnecessary for three short strings in this existing Chat Completions client.

## Data flow

`Streamlit -> POST /explain -> local prediction -> de-identified summary -> DeepSeek -> JSON text -> Pydantic validation -> Streamlit sections`

The client will continue to send only derived probability, threshold, label, and top contribution summaries to DeepSeek. Patient ID and the 12 raw features remain local. The model remains an optional explainer; it never produces the prediction.

## Response contract

The `interpretation` object keeps its existing provider, model, and disclaimer fields and replaces `text` with:

```json
{
  "probability_summary": "…",
  "contribution_summary": "…",
  "research_disclaimer": "…"
}
```

All three values must be non-blank strings. Invalid JSON, missing keys, non-string values, or blank values become `DeepSeekRequestError`, which `/explain` maps to HTTP 502. The UI presents each field under a clear Chinese heading.

## Testing

Tests will first prove that the request asks for JSON mode, that valid JSON is parsed into the three fields, and that malformed or incomplete JSON is rejected. API and dashboard tests will then assert the new contract without a real DeepSeek call.
