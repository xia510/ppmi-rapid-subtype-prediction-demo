# Configuration Management Design

## Goal

Move non-secret runtime settings out of hard-coded Python constants while preserving current local defaults and keeping all real secrets outside Git.

## Approach

Use environment variables only. The application will not auto-load a local `.env` file and will not add a dependency. This makes PowerShell configuration explicit and keeps the current local workflow intact. A tracked `.env.example` documents variable names with empty or safe sample values; `.env` remains ignored.

## Configuration contract

- `PPMI_MODEL_ARTIFACT`: optional model bundle path; default is `artifacts/model_bundle.joblib` under the project root.
- `PPMI_API_BASE_URL`: optional Streamlit-to-FastAPI address; default is `http://127.0.0.1:8000`.
- `DEEPSEEK_API_KEY`: required only for the optional explanation action; it has no example value and is never committed.
- `DEEPSEEK_MODEL`: optional model name; default stays `deepseek-flash`.

Environment variables take precedence over defaults. The API model path and UI base URL are the only newly configurable runtime values. Existing successful response contracts do not change.

## Safety

`.env.example` may be committed because it has no real secret. `.env` remains in `.gitignore`. Logs and error responses continue to avoid key values.
