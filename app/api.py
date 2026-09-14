"""FastAPI endpoints for the PPMI Rapid-subtype prediction demonstration."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


def create_app(artifact_path: Optional[Path] = None) -> FastAPI:
    """Create the API app, optionally using a test-specific model artifact."""
    resolved_artifact_path = Path(artifact_path or DEFAULT_ARTIFACT_PATH)
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

    return api


app = create_app()
