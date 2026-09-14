from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.export_model import export_model_bundle


SOURCE_ROOT = Path(r"C:\Users\20284\Documents\trae_projects\code_xuexi_001")


class ExportModelBundleTests(unittest.TestCase):
    def test_exported_bundle_predicts_probability_from_raw_patient_values(self):
        """The exported artifact must contain the full raw-input prediction contract."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "model_bundle.joblib"
            export_model_bundle(SOURCE_ROOT, artifact_path)

            bundle = joblib.load(artifact_path)
            required_keys = {
                "feature_names",
                "scaler",
                "base_model",
                "calibrated_model",
                "model_version",
                "target_definition",
                "classification_threshold",
                "threshold_method",
            }
            self.assertTrue(required_keys.issubset(bundle))
            self.assertEqual(len(bundle["feature_names"]), 12)
            self.assertEqual(bundle["threshold_method"], "F1 median from 10-fold CV")

            raw_train = pd.read_csv(SOURCE_ROOT / "PPMI_4_LASSO_train_raw.csv")
            patient = raw_train.iloc[[0]][bundle["feature_names"]]
            features = bundle["scaler"].transform(patient)
            probability = bundle["calibrated_model"].predict_proba(features)[0, 1]

            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
