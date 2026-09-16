import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ui.dashboard import (
    FEATURE_NAMES,
    build_patient_payload,
    format_api_error,
    request_explanation,
)
import ui.dashboard as dashboard


class DashboardHelperTests(unittest.TestCase):
    def test_build_patient_payload_preserves_feature_order_and_numeric_values(self):
        raw_values = {feature: str(index + 1) for index, feature in enumerate(FEATURE_NAMES)}

        payload = build_patient_payload("demo-ui-1", raw_values)

        self.assertEqual(payload["patient_id"], "demo-ui-1")
        self.assertEqual(list(payload.keys())[1:], FEATURE_NAMES)
        self.assertEqual(payload["scopa"], 1.0)
        self.assertEqual(payload["quip"], 12.0)

    def test_format_api_error_names_missing_feature_for_validation_response(self):
        detail = [{"type": "missing", "loc": ["body", "LEDD"]}]

        message = format_api_error(422, detail)

        self.assertIn("LEDD", message)
        self.assertIn("必填", message)

    def test_format_api_error_explains_missing_deepseek_key(self):
        message = format_api_error(503, "DEEPSEEK_API_KEY is not configured on the API service.")

        self.assertIn("DEEPSEEK_API_KEY", message)

    def test_format_api_error_includes_safe_request_id_for_structured_errors(self):
        message = format_api_error(
            503,
            {
                "error": {
                    "code": "deepseek_not_configured",
                    "message": "Do not render this internal text.",
                },
                "request_id": "abc123def456",
            },
        )

        self.assertIn("DEEPSEEK_API_KEY", message)
        self.assertIn("请求编号：abc123def456", message)
        self.assertNotIn("Do not render this internal text.", message)

    def test_request_explanation_posts_to_the_separate_explain_endpoint(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "interpretation": {
                "probability_summary": "概率说明。",
                "contribution_summary": "贡献说明。",
                "research_disclaimer": "仅供科研演示。",
            }
        }

        with patch("ui.dashboard.requests.post", return_value=response) as post:
            result, error = request_explanation({"scopa": 1.0})

        self.assertIsNone(error)
        self.assertEqual(result["interpretation"]["contribution_summary"], "贡献说明。")
        self.assertTrue(post.call_args.args[0].endswith("/explain"))

    def test_structured_interpretation_sections_have_fixed_chinese_headings(self):
        self.assertTrue(hasattr(dashboard, "structured_interpretation_sections"))

        sections = dashboard.structured_interpretation_sections(
            {
                "probability_summary": "概率说明。",
                "contribution_summary": "贡献说明。",
                "research_disclaimer": "仅供科研演示。",
            }
        )

        self.assertEqual(
            sections,
            [
                ("概率与阈值", "概率说明。"),
                ("特征贡献说明", "贡献说明。"),
                ("科研使用说明", "仅供科研演示。"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
