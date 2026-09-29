import copy
import json
from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.agent import AgentExecutionError, AgentLimitError, ResearchAgent


PATIENT = {
    "patient_id": "private-patient-987654",
    "scopa": 10101.125,
    "NP1COG": 20202.25,
    "rem": 30303.375,
    "DVT_SFTANIM": 40404.5,
    "MIA_STRIATUM_mean": 50505.625,
    "LEDD": 60606.75,
    "SEX": 70707.875,
    "updrs3_score": 80808.125,
    "MSEADLG": 90909.25,
    "DVT_SDM": 11111.375,
    "upsit_pctl": 22222.5,
    "quip": 33333.625,
}

PREDICTION = {
    "patient_id": PATIENT["patient_id"],
    "rapid_probability": 0.087,
    "research_threshold": 0.2092,
    "prediction_label": "Rapid risk below research threshold",
    "model_version": "research-demo-v1",
    "top_positive_contributors": [{"feature": "quip", "contribution": 0.41}],
    "top_negative_contributors": [{"feature": "SEX", "contribution": -0.30}],
    "all_feature_contributions": [
        {"feature": "quip", "raw_value": PATIENT["quip"], "contribution": 0.41}
    ],
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


def tool_turn(call_id, name, arguments):
    return {
        "kind": "tool_calls",
        "provider": "DeepSeek",
        "model": "deepseek-flash",
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


def final_turn(cited_source_ids=None):
    return {
        "kind": "final",
        "provider": "DeepSeek",
        "model": "deepseek-flash",
        "final": {
            "answer_summary": "模型与文献分析已经完成。",
            "prediction_summary": "模型概率低于研究阈值。",
            "evidence_summary": "本地文献提供群体层面证据。",
            "cited_source_ids": list(cited_source_ids or []),
            "research_disclaimer": "仅供科研演示，不构成临床建议。",
        },
    }


class FakeIndex:
    def __init__(self):
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return copy.deepcopy(EVIDENCE)


class QueuedTurns:
    def __init__(self, turns):
        self.turns = list(turns)
        self.messages_seen = []
        self.tools_seen = []

    def __call__(self, messages, tools):
        self.messages_seen.append(copy.deepcopy(messages))
        self.tools_seen.append(copy.deepcopy(tools))
        return self.turns.pop(0)


class AgentTests(unittest.TestCase):
    def test_agent_calls_prediction_then_search_and_returns_grounded_result(self):
        requester = QueuedTurns(
            [
                tool_turn("call-1", "predict_risk", {}),
                tool_turn(
                    "call-2",
                    "search_literature",
                    {"question": "自主神经症状与疾病进展", "top_k": 3},
                ),
                final_turn(["paper-a-p2-c1"]),
            ]
        )
        index = FakeIndex()
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=index,
            turn_requester=requester,
        )

        result = agent.run(PATIENT, "结合模型与文献解释研究结果", top_k=3)

        self.assertEqual(
            result["tool_trace"],
            [
                {"tool": "predict_risk", "status": "success"},
                {"tool": "search_literature", "status": "success"},
            ],
        )
        self.assertEqual(index.calls, [("自主神经症状与疾病进展", 3)])
        self.assertEqual(result["citations"][0]["source_id"], "paper-a-p2-c1")
        self.assertEqual(result["provider"], "DeepSeek")

    def test_agent_messages_never_contain_patient_id_or_raw_feature_values(self):
        requester = QueuedTurns(
            [tool_turn("call-1", "predict_risk", {}), final_turn()]
        )
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=FakeIndex(),
            turn_requester=requester,
        )

        agent.run(PATIENT, "解释本地模型结果")

        serialized = json.dumps(requester.messages_seen, ensure_ascii=False)
        self.assertNotIn(PATIENT["patient_id"], serialized)
        for feature, value in PATIENT.items():
            if feature != "patient_id":
                self.assertNotIn(str(value), serialized)
        self.assertNotIn("all_feature_contributions", serialized)
        self.assertIn("rapid_probability", serialized)

    def test_agent_rejects_unknown_tool(self):
        requester = QueuedTurns([tool_turn("call-1", "run_shell", {})])
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=FakeIndex(),
            turn_requester=requester,
        )

        with self.assertRaisesRegex(AgentExecutionError, "unknown tool"):
            agent.run(PATIENT, "执行研究分析")

    def test_agent_rejects_invalid_search_arguments(self):
        requester = QueuedTurns(
            [
                tool_turn(
                    "call-1",
                    "search_literature",
                    {"question": "进展", "top_k": 9},
                )
            ]
        )
        index = FakeIndex()
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=index,
            turn_requester=requester,
        )

        with self.assertRaisesRegex(AgentExecutionError, "invalid tool arguments"):
            agent.run(PATIENT, "执行研究分析")
        self.assertEqual(index.calls, [])

    def test_agent_rejects_identical_repeated_call(self):
        call = {"question": "自主神经症状", "top_k": 3}
        requester = QueuedTurns(
            [
                tool_turn("call-1", "search_literature", call),
                tool_turn("call-2", "search_literature", call),
            ]
        )
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=FakeIndex(),
            turn_requester=requester,
        )

        with self.assertRaisesRegex(AgentExecutionError, "repeated tool call"):
            agent.run(PATIENT, "执行研究分析")

    def test_agent_stops_after_four_tool_calls(self):
        turns = [
            tool_turn(
                f"call-{index}",
                "search_literature",
                {"question": f"不同问题 {index}", "top_k": 1},
            )
            for index in range(1, 6)
        ]
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=FakeIndex(),
            turn_requester=QueuedTurns(turns),
            max_tool_calls=4,
        )

        with self.assertRaisesRegex(AgentLimitError, "tool-call limit"):
            agent.run(PATIENT, "执行研究分析")

    def test_agent_rejects_unretrieved_citation(self):
        requester = QueuedTurns(
            [
                tool_turn(
                    "call-1",
                    "search_literature",
                    {"question": "自主神经症状", "top_k": 1},
                ),
                final_turn(["source-not-retrieved"]),
            ]
        )
        agent = ResearchAgent(
            predictor=lambda patient: copy.deepcopy(PREDICTION),
            literature_index=FakeIndex(),
            turn_requester=requester,
        )

        with self.assertRaisesRegex(AgentExecutionError, "unretrieved source"):
            agent.run(PATIENT, "执行研究分析")


if __name__ == "__main__":
    unittest.main()
