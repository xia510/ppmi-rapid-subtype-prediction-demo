# API Observability and Error Contract Design

## Goal

Make local FastAPI failures easier to diagnose without placing API keys, patient identifiers, raw feature values, or full prediction output in logs.

## Selected approach

Use lightweight, terminal-only application logging. A middleware assigns every HTTP request a short random `request_id`, measures duration, adds `X-Request-ID` to the response header, and logs only request ID, method, path, status code, duration, and a safe error category.

The alternative of writing permanent local log files is intentionally excluded for this learning project: it increases disk cleanup and privacy responsibilities. External monitoring is also excluded because this project runs locally and would add service configuration unrelated to the learning goal.

## Error contract

Expected API failures return a predictable JSON shape:

```json
{
  "error": {
    "code": "deepseek_not_configured",
    "message": "DeepSeek API key is not configured."
  },
  "request_id": "a1b2c3d4e5f6"
}
```

Codes distinguish invalid input, missing model artifact, missing DeepSeek configuration, and unavailable DeepSeek responses. The message is safe for a user; the request ID is a bridge to the terminal log. Successful prediction payloads retain their existing shape to avoid breaking the dashboard.

## Flow

`Browser -> middleware generates request_id -> endpoint succeeds or creates a categorized error -> response includes request_id -> middleware logs safe metadata -> dashboard shows readable message plus request ID on failure.`

The existing consent gate and data-minimizing DeepSeek boundary stay unchanged.

## Testing

Tests assert that expected error responses include a stable code and request ID, successful responses include `X-Request-ID`, and dashboard formatting displays a request ID without exposing technical exception details. Tests do not write files or make real network calls.
