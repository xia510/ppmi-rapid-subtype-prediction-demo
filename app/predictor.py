"""Predict the research Rapid-subtype probability from one raw patient JSON file."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd


class InputValidationError(ValueError):
    """Raised when a patient JSON payload does not meet the model contract."""


def _load_patient(input_path: Path) -> dict:
    try:
        patient = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputValidationError(f"Input file is not valid JSON: {error.msg}") from error

    if not isinstance(patient, dict):
        raise InputValidationError("Patient JSON must contain one object.")
    return patient


def _feature_frame(patient: dict, feature_names: list[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in patient]
    if missing:
        raise InputValidationError(f"Missing required feature(s): {', '.join(missing)}")

    values: dict[str, float] = {}
    for feature in feature_names:
        value = patient[feature]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise InputValidationError(f"Feature {feature} must be a finite number.")
        values[feature] = float(value)
    return pd.DataFrame([values], columns=feature_names)


def _top_contributors(
    patient_frame: pd.DataFrame,
    standardized_features,
    feature_names: list[str],
    base_model,
) -> tuple[list[dict], list[dict], list[dict]]:
    coefficients = base_model.coef_[0]
    contributions = standardized_features[0] * coefficients
    rows = [
        {
            "feature": feature,
            "raw_value": float(patient_frame.iloc[0][feature]),
            "standardized_value": float(standardized_features[0][index]),
            "coefficient": float(coefficients[index]),
            "contribution": float(contributions[index]),
        }
        for index, feature in enumerate(feature_names)
    ]
    positive = sorted(
        (row for row in rows if row["contribution"] > 0),
        key=lambda row: row["contribution"],
        reverse=True,
    )[:3]
    negative = sorted(
        (row for row in rows if row["contribution"] < 0),
        key=lambda row: row["contribution"],
    )[:3]
    return rows, positive, negative


def predict_from_patient(patient: dict, artifact_path: Path) -> dict:
    """Predict Rapid probability from one in-memory raw patient dictionary."""
    if not isinstance(patient, dict):
        raise InputValidationError("Patient data must contain one object.")

    bundle = joblib.load(artifact_path)
    patient_frame = _feature_frame(patient, bundle["feature_names"])

    standardized_features = bundle["scaler"].transform(patient_frame)
    rapid_probability = float(
        bundle["calibrated_model"].predict_proba(standardized_features)[0, 1]
    )
    all_contributions, top_positive, top_negative = _top_contributors(
        patient_frame,
        standardized_features,
        bundle["feature_names"],
        bundle["base_model"],
    )
    threshold = float(bundle["classification_threshold"])

    result = {
        "patient_id": patient.get("patient_id"),
        "rapid_probability": rapid_probability,
        "research_threshold": threshold,
        "prediction_label": (
            "Rapid risk above research threshold"
            if rapid_probability >= threshold
            else "Rapid risk below research threshold"
        ),
        "model_version": bundle["model_version"],
        "target_definition": bundle["target_definition"],
        "all_feature_contributions": all_contributions,
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "explanation_note": (
            "Contributions describe this model's linear score only; they are not causal "
            "effects or clinical advice."
        ),
        "disclaimer": "Research demonstration only. Not for clinical diagnosis.",
    }
    return result


def predict_from_json(input_path: Path, artifact_path: Path, output_path: Path) -> dict:
    """Load one raw patient JSON file, predict Rapid probability, and save result JSON."""
    patient = _load_patient(input_path)
    result = predict_from_patient(patient, artifact_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the research Rapid-subtype probability from a patient JSON file."
    )
    parser.add_argument("--input", required=True, help="Path to raw patient JSON")
    parser.add_argument("--artifact", required=True, help="Path to model_bundle.joblib")
    parser.add_argument("--output", required=True, help="Path for prediction JSON")
    args = parser.parse_args()

    result = predict_from_json(args.input, args.artifact, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
