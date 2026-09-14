import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Optional

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.predictor import InputValidationError, predict_from_json
from scripts.export_model import export_model_bundle


SOURCE_ROOT = Path(r"C:\Users\20284\Documents\trae_projects\code_xuexi_001")


class PredictorTests(unittest.TestCase):
    def _write_patient_json(self, path: Path, remove_field: Optional[str] = None) -> None:
        raw_train = pd.read_csv(SOURCE_ROOT / "PPMI_4_LASSO_train_raw.csv")
        selected_train = pd.read_csv(SOURCE_ROOT / "PPMI_4_LASSO_train_1se.csv")
        features = [column for column in selected_train.columns if column not in ("PATNO", "Target")]
        row = raw_train.iloc[0]
        patient = {"patient_id": "demo-139982"}
        patient.update({feature: float(row[feature]) for feature in features})
        if remove_field:
            patient.pop(remove_field)
        path.write_text(json.dumps(patient), encoding="utf-8")

    def test_predicts_from_raw_patient_json_and_writes_result_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_path = temp_root / "model_bundle.joblib"
            input_path = temp_root / "patient.json"
            output_path = temp_root / "prediction.json"
            export_model_bundle(SOURCE_ROOT, artifact_path)
            self._write_patient_json(input_path)

            result = predict_from_json(input_path, artifact_path, output_path)

            self.assertEqual(result["patient_id"], "demo-139982")
            self.assertGreaterEqual(result["rapid_probability"], 0.0)
            self.assertLessEqual(result["rapid_probability"], 1.0)
            self.assertIn("research_threshold", result)
            self.assertIn("top_positive_contributors", result)
            self.assertIn("top_negative_contributors", result)
            self.assertIn("all_feature_contributions", result)
            self.assertIn("explanation_note", result)
            self.assertEqual(len(result["all_feature_contributions"]), 12)
            self.assertEqual(
                [item["feature"] for item in result["all_feature_contributions"]],
                [
                    "scopa", "NP1COG", "rem", "DVT_SFTANIM",
                    "MIA_STRIATUM_mean", "LEDD", "SEX", "updrs3_score",
                    "MSEADLG", "DVT_SDM", "upsit_pctl", "quip",
                ],
            )
            self.assertLessEqual(len(result["top_positive_contributors"]), 3)
            self.assertLessEqual(len(result["top_negative_contributors"]), 3)
            self.assertTrue(
                all(item["contribution"] > 0 for item in result["top_positive_contributors"])
            )
            self.assertTrue(
                all(item["contribution"] < 0 for item in result["top_negative_contributors"])
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), result)

    def test_rejects_patient_json_missing_a_required_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_path = temp_root / "model_bundle.joblib"
            input_path = temp_root / "patient.json"
            export_model_bundle(SOURCE_ROOT, artifact_path)
            self._write_patient_json(input_path, remove_field="LEDD")

            with self.assertRaisesRegex(InputValidationError, "LEDD"):
                predict_from_json(input_path, artifact_path, temp_root / "prediction.json")

    def test_command_line_entrypoint_writes_prediction_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_path = temp_root / "model_bundle.joblib"
            input_path = temp_root / "patient.json"
            output_path = temp_root / "prediction.json"
            export_model_bundle(SOURCE_ROOT, artifact_path)
            self._write_patient_json(input_path)

            command = [
                sys.executable,
                str(PROJECT_DIR / "app" / "predictor.py"),
                "--input",
                str(input_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
