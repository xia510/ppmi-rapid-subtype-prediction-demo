"""FastAPI endpoints for the PPMI Rapid-subtype prediction demonstration."""

import logging
from pathlib import Path
import time
from typing import Callable, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.deepseek import (
    DeepSeekConfigurationError,
    DeepSeekRequestError,
    request_interpretation,
)
from app.predictor import InputValidationError, predict_from_patient


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_DIR / "artifacts" / "model_bundle.joblib"
LOGGER = logging.getLogger("ppmi.api")


class PatientRequest(BaseModel):
    """The 12 raw features required by the exported research model."""

    patient_id: Optional[str] = None
    scopa: float
    NP1COG: float
    rem: float
    DVT_SFTANIM: float
    MIA_STRIATUM_mean: float
    LEDD: float
    SEX: float
    updrs3_score: float
    MSEADLG: float
    DVT_SDM: float
    upsit_pctl: float
    quip: float


def _prediction_for_external_explanation(prediction: dict) -> dict:
    """Keep only derived, de-identified fields needed by the external LLM."""
    return {
        "rapid_probability": prediction["rapid_probability"],
        "research_threshold": prediction["research_threshold"],
        "prediction_label": prediction["prediction_label"],
        "model_version": prediction["model_version"],
        "top_positive_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction["top_positive_contributors"]
        ],
        "top_negative_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction["top_negative_contributors"]
        ],
    }


def _request_id(request: Request) -> str:
    """Read the identifier assigned by the request middleware."""
    return str(getattr(request.state, "request_id", "unknown"))


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    """Return a stable, safe error body without exposing internal exceptions."""
    request_id = _request_id(request)
    LOGGER.warning("api_error request_id=%s code=%s", request_id, code)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": request_id,
        },
    )


def create_app(
    artifact_path: Optional[Path] = None,
    explainer: Optional[Callable[[dict], dict]] = None,
) -> FastAPI:
    """Create the API app, optionally using a test-specific model artifact."""
    resolved_artifact_path = Path(artifact_path or DEFAULT_ARTIFACT_PATH)
    resolved_explainer = explainer or request_interpretation
    api = FastAPI(
        title="PPMI Rapid Subtype Prediction Demo",
        description="Research demonstration API. Not for clinical diagnosis.",
        version="research-demo-v1",
    )

    @api.middleware("http")
    async def observe_request(request: Request, call_next):
        """Log safe request metadata and attach a correlation ID to each response."""
        request.state.request_id = uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        LOGGER.info(
            "api_request request_id=%s method=%s path=%s status_code=%s duration_ms=%.1f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @api.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        """Replace framework validation details with a stable user-safe contract."""
        return _error_response(
            request,
            status_code=422,
            code="invalid_input",
            message="Input is incomplete or has an invalid numeric value.",
        )

    @api.get("/health")
    def health() -> dict[str, object]:
        """Return service metadata without loading a model artifact."""
        return {
            "status": "ok",
            "service": "ppmi-rapid-subtype-prediction-demo",
            "feature_count": 12,
            "model_artifact_available": resolved_artifact_path.exists(),
        }

    @api.post("/predict")
    def predict(patient: PatientRequest, request: Request) -> object:
        """Return a calibrated Rapid-subtype probability for one raw patient payload."""
        try:
            return predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError:
            return _error_response(
                request,
                status_code=503,
                code="model_artifact_unavailable",
                message="Model artifact is unavailable. Export it before starting the API.",
            )
        except InputValidationError:
            return _error_response(
                request,
                status_code=422,
                code="invalid_input",
                message="Input is incomplete or has an invalid numeric value.",
            )

    @api.post("/explain")
    def explain(patient: PatientRequest, request: Request) -> object:
        """Predict locally, then ask DeepSeek to explain only de-identified model output."""
        try:
            prediction = predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError:
            return _error_response(
                request,
                status_code=503,
                code="model_artifact_unavailable",
                message="Model artifact is unavailable. Export it before starting the API.",
            )
        except InputValidationError:
            return _error_response(
                request,
                status_code=422,
                code="invalid_input",
                message="Input is incomplete or has an invalid numeric value.",
            )

        try:
            interpretation = resolved_explainer(_prediction_for_external_explanation(prediction))
        except DeepSeekConfigurationError:
            return _error_response(
                request,
                status_code=503,
                code="deepseek_not_configured",
                message="DeepSeek API key is not configured.",
            )
        except DeepSeekRequestError:
            return _error_response(
                request,
                status_code=502,
                code="deepseek_unavailable",
                message="DeepSeek could not return a usable research explanation.",
            )

        return {"prediction": prediction, "interpretation": interpretation}

    return api


app = create_app()
