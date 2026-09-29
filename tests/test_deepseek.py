import json
from pathlib import Path
import sys
import unittest

import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.deepseek import (
    DeepSeekRequestError,
    build_agent_turn_request,
    build_grounded_answer_request,
    build_interpretation_request,
    request_agent_turn,
    request_grounded_answer,
    request_interpretation,
)


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

EVIDENCE = [
    {
        "source_id": "paper-a-p2-c1",
        "title": "Autonomic dysfunction in Parkinson disease",
        "source": "autonomic.pdf",
        "page": 2,
        "text": "Autonomic symptoms were associated with longitudinal outcomes.",
        "score": 0.91,
    }
]

AGENT_MESSAGES = [
    {"role": "user", "content": "研究问题：哪些因素与进展有关？"},
]
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_risk",
            "description": "Run the local model.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
]


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "chatcmpl-demo",
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "probability_summary": "概率低于研究阈值。",
                                "contribution_summary": "quip 提高线性得分。",
                                "research_disclaimer": "仅供科研演示，不构成临床建议。",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
        }


class IncompleteJsonResponse(FakeResponse):
    def json(self):
        response = super().json()
        response["choices"][0]["message"]["content"] = '{"probability_summary": "概率说明。"}'
        return response


class ExtraFieldJsonResponse(FakeResponse):
    def json(self):
        response = super().json()
        response["choices"][0]["message"]["content"] = json.dumps(
            {
                "probability_summary": "概率说明。",
                "contribution_summary": "贡献说明。",
                "research_disclaimer": "仅供科研演示。",
                "unexpected_field": "不应被接受。",
            },
            ensure_ascii=False,
        )
        return response


class GroundedAnswerResponse(FakeResponse):
    cited_source_id = "paper-a-p2-c1"

    def json(self):
        return {
            "id": "chatcmpl-rag-demo",
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "检索证据提示自主神经症状与纵向结局存在群体层面关联。",
                                "cited_source_ids": [self.cited_source_id],
                                "evidence_limitations": "该证据不能证明个体因果关系。",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
        }


class UnknownCitationResponse(GroundedAnswerResponse):
    cited_source_id = "unknown-source"


class EmptyGroundedAnswerResponse(GroundedAnswerResponse):
    def json(self):
        response = super().json()
        response["choices"][0]["message"]["content"] = "   "
        return response


class InvalidJsonGroundedAnswerResponse(GroundedAnswerResponse):
    def json(self):
        response = super().json()
        response["choices"][0]["message"]["content"] = "not-json"
        return response


class InvalidSchemaGroundedAnswerResponse(GroundedAnswerResponse):
    def json(self):
        response = super().json()
        response["choices"][0]["message"]["content"] = json.dumps(
            {"answer": "缺少引用字段。"}, ensure_ascii=False
        )
        return response


class HttpErrorGroundedAnswerResponse(GroundedAnswerResponse):
    status_code = 402

    def raise_for_status(self):
        raise requests.HTTPError(response=self)


class AgentToolCallResponse(FakeResponse):
    arguments = "{}"

    def json(self):
        return {
            "id": "chatcmpl-agent-tool",
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-predict-1",
                                "type": "function",
                                "function": {
                                    "name": "predict_risk",
                                    "arguments": self.arguments,
                                },
                            }
                        ],
                    }
                }
            ],
        }


class MalformedAgentToolCallResponse(AgentToolCallResponse):
    arguments = "{not-json"


class AgentFinalResponse(FakeResponse):
    def json(self):
        return {
            "id": "chatcmpl-agent-final",
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer_summary": "研究任务已经完成。",
                                "prediction_summary": "模型概率低于研究阈值。",
                                "evidence_summary": "本地证据支持群体层面的相关性。",
                                "cited_source_ids": ["paper-a-p2-c1"],
                                "research_disclaimer": "仅供科研演示，不构成临床建议。",
                            },
                            ensure_ascii=False,
                        ),
                    }
                }
            ],
        }


class DeepSeekExplanationTests(unittest.TestCase):
    def test_agent_request_contains_only_supplied_messages_and_allowlisted_tools(self):
        request = build_agent_turn_request(AGENT_MESSAGES, AGENT_TOOLS)

        self.assertEqual(request["messages"][1:], AGENT_MESSAGES)
        self.assertEqual(request["tools"], AGENT_TOOLS)
        self.assertEqual(request["tool_choice"], "auto")
        serialized = json.dumps(request, ensure_ascii=False)
        self.assertNotIn("patient_id", serialized)
        self.assertNotIn("scopa", serialized)
        self.assertIn("不得提供诊断", serialized)

    def test_request_agent_turn_parses_tool_calls(self):
        result = request_agent_turn(
            AGENT_MESSAGES,
            AGENT_TOOLS,
            api_key="test-key",
            http_post=lambda url, **kwargs: AgentToolCallResponse(),
        )

        self.assertEqual(result["kind"], "tool_calls")
        self.assertEqual(result["tool_calls"], [
            {"id": "call-predict-1", "name": "predict_risk", "arguments": {}}
        ])
        self.assertEqual(result["assistant_message"]["tool_calls"][0]["id"], "call-predict-1")
        self.assertEqual(result["model"], "deepseek-flash")

    def test_request_agent_turn_parses_valid_final_json(self):
        result = request_agent_turn(
            AGENT_MESSAGES,
            AGENT_TOOLS,
            api_key="test-key",
            http_post=lambda url, **kwargs: AgentFinalResponse(),
        )

        self.assertEqual(result["kind"], "final")
        self.assertEqual(result["final"]["cited_source_ids"], ["paper-a-p2-c1"])
        self.assertEqual(result["final"]["answer_summary"], "研究任务已经完成。")
        self.assertEqual(result["provider"], "DeepSeek")

    def test_request_agent_turn_rejects_malformed_tool_arguments(self):
        with self.assertRaisesRegex(DeepSeekRequestError, "tool arguments"):
            request_agent_turn(
                AGENT_MESSAGES,
                AGENT_TOOLS,
                api_key="test-key",
                http_post=lambda url, **kwargs: MalformedAgentToolCallResponse(),
            )

    def test_request_payload_excludes_patient_identifier_and_raw_features(self):
        request = build_interpretation_request(PREDICTION)
        serialized_messages = json.dumps(request["messages"], ensure_ascii=False)

        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertIn("response_format", request)
        self.assertEqual(request.get("response_format"), {"type": "json_object"})
        self.assertNotIn("must-not-leave-the-service", serialized_messages)
        self.assertIn("8.70%", serialized_messages)
        self.assertIn("quip", serialized_messages)
        self.assertIn("不得给出诊断", serialized_messages)

    def test_request_interpretation_parses_required_json_fields_without_real_network_call(self):
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
        self.assertIn("probability_summary", result)
        self.assertEqual(result.get("probability_summary"), "概率低于研究阈值。")
        self.assertEqual(result.get("contribution_summary"), "quip 提高线性得分。")
        self.assertEqual(result.get("research_disclaimer"), "仅供科研演示，不构成临床建议。")
        self.assertEqual(result["model"], "deepseek-flash")

    def test_request_interpretation_rejects_json_missing_required_fields(self):
        def incomplete_post(url, **kwargs):
            return IncompleteJsonResponse()

        with self.assertRaises(DeepSeekRequestError):
            request_interpretation(
                PREDICTION,
                api_key="test-key",
                http_post=incomplete_post,
            )

    def test_request_interpretation_rejects_json_with_extra_fields(self):
        def extra_field_post(url, **kwargs):
            return ExtraFieldJsonResponse()

        with self.assertRaises(DeepSeekRequestError):
            request_interpretation(
                PREDICTION,
                api_key="test-key",
                http_post=extra_field_post,
            )

    def test_grounded_request_contains_numbered_evidence_and_research_boundaries(self):
        request = build_grounded_answer_request(
            "自主神经症状是否与帕金森病进展有关？",
            EVIDENCE,
        )
        serialized_messages = json.dumps(request["messages"], ensure_ascii=False)

        self.assertIn("paper-a-p2-c1", serialized_messages)
        self.assertIn("Autonomic symptoms", serialized_messages)
        self.assertIn("只能依据", serialized_messages)
        self.assertIn("不可信引用材料", serialized_messages)
        self.assertIn("群体研究", serialized_messages)
        self.assertNotIn("patient_id", serialized_messages)

    def test_request_grounded_answer_accepts_only_retrieved_source_ids(self):
        result = request_grounded_answer(
            "自主神经症状是否与帕金森病进展有关？",
            EVIDENCE,
            api_key="test-key",
            http_post=lambda url, **kwargs: GroundedAnswerResponse(),
        )

        self.assertEqual(result["cited_source_ids"], ["paper-a-p2-c1"])
        self.assertIn("个体因果", result["evidence_limitations"])

    def test_request_grounded_answer_rejects_unknown_source_id(self):
        with self.assertRaises(DeepSeekRequestError):
            request_grounded_answer(
                "自主神经症状是否与帕金森病进展有关？",
                EVIDENCE,
                api_key="test-key",
                http_post=lambda url, **kwargs: UnknownCitationResponse(),
            )

    def test_request_grounded_answer_reports_safe_parse_failure_category(self):
        cases = [
            (EmptyGroundedAnswerResponse(), "empty content"),
            (InvalidJsonGroundedAnswerResponse(), "invalid JSON"),
            (InvalidSchemaGroundedAnswerResponse(), "invalid response schema"),
            (HttpErrorGroundedAnswerResponse(), "upstream HTTP 402"),
        ]

        for response, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                with self.assertRaisesRegex(DeepSeekRequestError, expected_reason):
                    request_grounded_answer(
                        "自主神经症状是否与帕金森病进展有关？",
                        EVIDENCE,
                        api_key="test-key",
                        http_post=lambda url, **kwargs: response,
                    )


if __name__ == "__main__":
    unittest.main()
