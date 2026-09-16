import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


class SettingsTests(unittest.TestCase):
    def test_model_artifact_path_uses_project_default_when_unset(self):
        from app.settings import model_artifact_path

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                model_artifact_path(), PROJECT_DIR / "artifacts" / "model_bundle.joblib"
            )

    def test_model_artifact_path_uses_environment_override(self):
        from app.settings import model_artifact_path

        with patch.dict(os.environ, {"PPMI_MODEL_ARTIFACT": "D:/models/demo.joblib"}):
            self.assertEqual(model_artifact_path(), Path("D:/models/demo.joblib"))

    def test_api_base_url_uses_environment_override(self):
        from app.settings import api_base_url

        with patch.dict(os.environ, {"PPMI_API_BASE_URL": "http://api.example:9000"}):
            self.assertEqual(api_base_url(), "http://api.example:9000")

    def test_env_example_has_a_blank_key_placeholder(self):
        text = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")

        self.assertIn("DEEPSEEK_API_KEY=", text)
        self.assertNotIn("sk-", text)


if __name__ == "__main__":
    unittest.main()
