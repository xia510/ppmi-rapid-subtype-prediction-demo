from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.export_model import export_model_bundle
from tests.data_support import authorized_source_root


SOURCE_ROOT = authorized_source_root()


class ExportModelBundleTests(unittest.TestCase):
    def test_command_line_requires_an_explicit_source_root(self):
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "export_model.py"),
        ]

        completed = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--source-root", completed.stderr)

    @unittest.skipUnless(SOURCE_ROOT, "Set PPMI_TEST_SOURCE_ROOT for model tests.")
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

    @unittest.skipUnless(SOURCE_ROOT, "Set PPMI_TEST_SOURCE_ROOT for model tests.")
    def test_command_line_accepts_source_and_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "nested" / "model_bundle.joblib"
            command = [
                sys.executable,
                str(PROJECT_DIR / "scripts" / "export_model.py"),
                "--source-root",
                str(SOURCE_ROOT),
                "--artifact",
                str(artifact_path),
            ]

            completed = subprocess.run(command, capture_output=True, text=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(artifact_path.exists())
            self.assertIn(str(artifact_path), completed.stdout)


if __name__ == "__main__":
    unittest.main()
