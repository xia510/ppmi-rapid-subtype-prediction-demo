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
    request_agent_analysis,
    request_literature_answer,
    request_explanation,
)
import ui.dashboard as dashboard


class DashboardHelperTests(unittest.TestCase):
    def test_request_agent_analysis_posts_patient_question_and_top_k(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "answer_summary": "完成研究分析。",
            "tool_trace": [{"tool": "predict_risk", "status": "success"}],
        }
        patient = {"patient_id": "demo", "scopa": 1.0}

        with patch("ui.dashboard.requests.post", return_value=response) as post:
            result, error = request_agent_analysis(patient, "结合模型与文献", top_k=4)

        self.assertIsNone(error)
        self.assertEqual(result["answer_summary"], "完成研究分析。")
        self.assertTrue(post.call_args.args[0].endswith("/agent/analyze"))
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"patient_id": "demo", "scopa": 1.0, "question": "结合模型与文献", "top_k": 4},
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 90)

    def test_format_api_error_explains_agent_failures(self):
        cases = {
            "invalid_agent_request": "Agent 输入",
            "agent_upstream_unavailable": "DeepSeek",
            "agent_execution_failed": "安全限制",
        }
        for code, expected_text in cases.items():
            with self.subTest(code=code):
                message = format_api_error(
                    502,
                    {
                        "error": {"code": code, "message": "Internal detail."},
                        "request_id": "agent12345678",
                    },
                )
                self.assertIn(expected_text, message)
                self.assertIn("请求编号：agent12345678", message)
                self.assertNotIn("Internal detail.", message)

    def test_agent_result_sections_preserve_fixed_order(self):
        sections = dashboard.agent_result_sections(
            {
                "answer_summary": "回答。",
                "prediction_summary": "预测。",
                "evidence_summary": "证据。",
                "research_disclaimer": "声明。",
            }
        )

        self.assertEqual(
            sections,
            [
                ("Agent 回答", "回答。"),
                ("模型预测摘要", "预测。"),
                ("文献证据摘要", "证据。"),
                ("科研使用说明", "声明。"),
            ],
        )

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

    def test_request_literature_answer_posts_question_and_top_k(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "answer": "基于文献的回答。",
            "evidence_limitations": "群体证据不能证明个体结局。",
            "citations": [],
        }

        with patch("ui.dashboard.requests.post", return_value=response) as post:
            result, error = request_literature_answer("什么是SCOPA-AUT？", top_k=4)

        self.assertIsNone(error)
        self.assertEqual(result["answer"], "基于文献的回答。")
        self.assertTrue(post.call_args.args[0].endswith("/literature/ask"))
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"question": "什么是SCOPA-AUT？", "top_k": 4},
        )

    def test_format_api_error_explains_missing_literature_index(self):
        message = format_api_error(
            503,
            {
                "error": {
                    "code": "literature_index_unavailable",
                    "message": "Internal detail.",
                },
                "request_id": "abc123def456",
            },
        )

        self.assertIn("文献索引", message)
        self.assertIn("请求编号：abc123def456", message)


if __name__ == "__main__":
    unittest.main()
