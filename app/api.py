"""FastAPI endpoints for the PPMI Rapid-subtype prediction demonstration."""

from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.deepseek import (
    DeepSeekConfigurationError,
    DeepSeekRequestError,
    request_interpretation,
)
from app.predictor import InputValidationError, predict_from_patient


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_DIR / "artifacts" / "model_bundle.joblib"


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
    def predict(patient: PatientRequest) -> dict:
        """Return a calibrated Rapid-subtype probability for one raw patient payload."""
        try:
            return predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Model artifact is unavailable. Export it before starting the API.",
            ) from error
        except InputValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/explain")
    def explain(patient: PatientRequest) -> dict:
        """Predict locally, then ask DeepSeek to explain only de-identified model output."""
        try:
            prediction = predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Model artifact is unavailable. Export it before starting the API.",
            ) from error
        except InputValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            interpretation = resolved_explainer(_prediction_for_external_explanation(prediction))
        except DeepSeekConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except DeepSeekRequestError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return {"prediction": prediction, "interpretation": interpretation}

    return api


app = create_app()
