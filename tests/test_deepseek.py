import json
from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.deepseek import build_interpretation_request, request_interpretation


PREDICTION = {
    "patient_id": "must-not-leave-the-service",
    "rapid_probability": 0.087,
    "research_threshold": 0.2092,
    "prediction_label": "Rapid risk below research threshold",
    "model_version": "research-demo-v1",
    "top_positive_contributors": [
        {"feature": "quip", "contribution": 0.41},
    ],
    "top_negative_contributors": [
        {"feature": "SEX", "contribution": -0.30},
    ],
    "disclaimer": "Research demonstration only. Not for clinical diagnosis.",
}


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "chatcmpl-demo",
            "model": "deepseek-flash",
            "choices": [{"message": {"content": "这是科研结果的辅助解读。"}}],
        }


class DeepSeekExplanationTests(unittest.TestCase):
    def test_request_payload_excludes_patient_identifier_and_raw_features(self):
        request = build_interpretation_request(PREDICTION)
        serialized_messages = json.dumps(request["messages"], ensure_ascii=False)

        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertNotIn("must-not-leave-the-service", serialized_messages)
        self.assertIn("8.70%", serialized_messages)
        self.assertIn("quip", serialized_messages)
        self.assertIn("不得给出诊断", serialized_messages)

    def test_request_interpretation_returns_model_text_without_real_network_call(self):
        captured_request = {}

        def fake_post(url, **kwargs):
            captured_request["url"] = url
            captured_request.update(kwargs)
            return FakeResponse()

        result = request_interpretation(
            PREDICTION,
            api_key="test-key",
            http_post=fake_post,
        )

        self.assertEqual(captured_request["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured_request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(result["text"], "这是科研结果的辅助解读。")
        self.assertEqual(result["model"], "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
