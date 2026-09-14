"""Export a raw-input prediction bundle from the existing PPMI research outputs."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ID_COLUMN = "PATNO"
LABEL_COLUMN = "Target"
MODEL_PARAMS = {
    "penalty": "l2",
    "solver": "saga",
    "C": 0.9699634728610746,
    "class_weight": "balanced",
    "max_iter": 10000,
    "random_state": 42,
}
CLASSIFICATION_THRESHOLD = 0.20918217546039175
THRESHOLD_METHOD = "F1 median from 10-fold CV"


def export_model_bundle(source_root: Path, artifact_path: Path) -> None:
    """Train and save the scaler plus both research-model prediction objects."""
    source_root = Path(source_root)
    artifact_path = Path(artifact_path)

    raw_train = pd.read_csv(source_root / "PPMI_4_LASSO_train_raw.csv")
    selected_train = pd.read_csv(source_root / "PPMI_4_LASSO_train_1se.csv")
    feature_names = [
        column
        for column in selected_train.columns
        if column not in (ID_COLUMN, LABEL_COLUMN)
    ]

    X_raw = raw_train[feature_names]
    y = raw_train[LABEL_COLUMN]
    if X_raw.isna().any().any():
        raise ValueError("The exported first-version model requires complete input features.")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    base_model = LogisticRegression(**MODEL_PARAMS)
    base_model.fit(X, y)

    calibration_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260702,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(LogisticRegression(**MODEL_PARAMS)),
        method="sigmoid",
        cv=calibration_cv,
        ensemble=False,
        n_jobs=-1,
    )
    calibrated_model.fit(X, y)

    bundle = {
        "feature_names": feature_names,
        "scaler": scaler,
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "model_version": "research-demo-v1",
        "target_definition": "1 = Rapid subtype, 0 = Slow subtype",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_method": THRESHOLD_METHOD,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)


if __name__ == "__main__":
    default_source = Path(r"C:\Users\20284\Documents\trae_projects\code_xuexi_001")
    default_artifact = Path(__file__).resolve().parents[1] / "artifacts" / "model_bundle.joblib"
    export_model_bundle(default_source, default_artifact)
    print(f"Model bundle saved to: {default_artifact}")
