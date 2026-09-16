import sys
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.api import app, create_app
from scripts.export_model import export_model_bundle


SOURCE_ROOT = Path(r"C:\Users\20284\Documents\trae_projects\code_xuexi_001")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifact_path = Path(cls.temp_dir.name) / "model_bundle.joblib"
        export_model_bundle(SOURCE_ROOT, cls.artifact_path)
        cls.client = TestClient(create_app(cls.artifact_path))

        raw_train = pd.read_csv(SOURCE_ROOT / "PPMI_4_LASSO_train_raw.csv")
        selected_train = pd.read_csv(SOURCE_ROOT / "PPMI_4_LASSO_train_1se.csv")
        feature_names = [
            column
            for column in selected_train.columns
            if column not in ("PATNO", "Target")
        ]
        row = raw_train.iloc[0]
        cls.valid_patient = {"patient_id": "api-demo-1"}
        cls.valid_patient.update({feature: float(row[feature]) for feature in feature_names})

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_health_returns_service_metadata(self):
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["service"], "ppmi-rapid-subtype-prediction-demo")
        self.assertEqual(response.json()["feature_count"], 12)

    def test_predict_returns_calibrated_probability_and_all_contributions(self):
        response = self.client.post("/predict", json=self.valid_patient)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["patient_id"], "api-demo-1")
        self.assertGreaterEqual(result["rapid_probability"], 0.0)
        self.assertLessEqual(result["rapid_probability"], 1.0)
        self.assertEqual(len(result["all_feature_contributions"]), 12)

    def test_predict_rejects_a_missing_required_feature(self):
        incomplete_patient = dict(self.valid_patient)
        incomplete_patient.pop("LEDD")

        response = self.client.post("/predict", json=incomplete_patient)

        self.assertEqual(response.status_code, 422)

    def test_explain_combines_existing_prediction_with_an_injected_llm_explanation(self):
        received_prediction = {}

        def fake_explainer(prediction):
            received_prediction.update(prediction)
            return {
                "provider": "DeepSeek",
                "model": "deepseek-flash",
                "probability_summary": "概率说明。",
                "contribution_summary": "贡献说明。",
                "research_disclaimer": "仅供科研演示。",
                "disclaimer": "AI-generated research explanation only. Not clinical advice.",
            }

        client = TestClient(create_app(self.artifact_path, fake_explainer))
        response = client.post("/explain", json=self.valid_patient)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"]["probability_summary"], "概率说明。"
        )
        self.assertEqual(
            response.json()["interpretation"]["contribution_summary"], "贡献说明。"
        )
        self.assertEqual(response.json()["prediction"]["patient_id"], "api-demo-1")
        self.assertNotIn("patient_id", received_prediction)


if __name__ == "__main__":
    unittest.main()
