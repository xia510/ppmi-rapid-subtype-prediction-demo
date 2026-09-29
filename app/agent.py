"""Privacy-bounded tool loop for the research demonstration Agent."""

import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, constr

from app.deepseek import request_agent_turn


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_risk",
            "description": (
                "Run the local calibrated PPMI Rapid-subtype model on the patient "
                "data held privately by the backend. This tool takes no arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "Search the local Parkinson research-literature index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 2},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["question", "top_k"],
                "additionalProperties": False,
            },
        },
    },
]


class AgentExecutionError(RuntimeError):
    """Raised when a provider request violates the local Agent contract."""


class AgentLimitError(AgentExecutionError):
    """Raised when an Agent run exceeds a bounded execution limit."""


class PredictRiskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchLiteratureArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: constr(strip_whitespace=True, min_length=2, max_length=500)
    top_k: int = Field(ge=1, le=5)


def _safe_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Project a local prediction onto the fields allowed to leave the service."""
    return {
        "rapid_probability": prediction["rapid_probability"],
        "research_threshold": prediction["research_threshold"],
        "prediction_label": prediction["prediction_label"],
        "model_version": prediction["model_version"],
        "top_positive_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction.get("top_positive_contributors", [])
        ],
        "top_negative_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction.get("top_negative_contributors", [])
        ],
    }


def _safe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row["source_id"],
            "title": row["title"],
            "source": row["source"],
            "page": row["page"],
            "text": row["text"],
            "score": row["score"],
        }
        for row in evidence
    ]


class ResearchAgent:
    """Run a finite, allowlisted local-tool loop controlled by DeepSeek turns."""

    def __init__(
        self,
        predictor: Callable[[dict[str, Any]], dict[str, Any]],
        literature_index: object,
        turn_requester: Callable = request_agent_turn,
        max_tool_calls: int = 4,
    ):
        self.predictor = predictor
        self.literature_index = literature_index
        self.turn_requester = turn_requester
        self.max_tool_calls = int(max_tool_calls)

    def run(
        self,
        patient: dict[str, Any],
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        question = str(question).strip()
        if not question:
            raise AgentExecutionError("Agent question must not be empty.")
        if not 1 <= int(top_k) <= 5:
            raise AgentExecutionError("Agent top_k must be between 1 and 5.")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "科研任务：{question}\n"
                    "患者数据仅保存在后端；需要模型结果时调用 predict_risk。"
                    "需要文献证据时调用 search_literature，最多检索 {top_k} 条。"
                ).format(question=question, top_k=int(top_k)),
            }
        ]
        tool_trace: list[dict[str, str]] = []
        retrieved_by_id: dict[str, dict[str, Any]] = {}
        executed_signatures: set[str] = set()
        call_count = 0

        while True:
            turn = self.turn_requester(messages, AGENT_TOOLS)
            if turn.get("kind") == "final":
                final = dict(turn["final"])
                cited_source_ids = list(dict.fromkeys(final["cited_source_ids"]))
                if any(source_id not in retrieved_by_id for source_id in cited_source_ids):
                    raise AgentExecutionError("Agent cited an unretrieved source.")
                return {
                    **final,
                    "cited_source_ids": cited_source_ids,
                    "citations": [retrieved_by_id[source_id] for source_id in cited_source_ids],
                    "tool_trace": tool_trace,
                    "provider": turn.get("provider", "unknown"),
                    "model": turn.get("model", "unknown"),
                }
            if turn.get("kind") != "tool_calls" or not turn.get("tool_calls"):
                raise AgentExecutionError("Agent returned neither tool calls nor a final answer.")

            messages.append(turn["assistant_message"])
            for call in turn["tool_calls"]:
                if call_count >= self.max_tool_calls:
                    raise AgentLimitError("Agent exceeded the tool-call limit.")
                name = str(call.get("name", ""))
                arguments = call.get("arguments")
                if name not in {"predict_risk", "search_literature"}:
                    raise AgentExecutionError("Agent requested an unknown tool.")
                try:
                    if name == "predict_risk":
                        validated = PredictRiskArguments.model_validate(arguments)
                    else:
                        validated = SearchLiteratureArguments.model_validate(arguments)
                except (TypeError, ValidationError) as error:
                    raise AgentExecutionError("Agent supplied invalid tool arguments.") from error

                normalized_arguments = validated.model_dump()
                signature = "{name}:{arguments}".format(
                    name=name,
                    arguments=json.dumps(
                        normalized_arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                if signature in executed_signatures:
                    raise AgentExecutionError("Agent requested a repeated tool call.")
                executed_signatures.add(signature)
                call_count += 1

                if name == "predict_risk":
                    output = _safe_prediction(self.predictor(patient))
                else:
                    evidence = _safe_evidence(
                        self.literature_index.search(
                            normalized_arguments["question"],
                            top_k=normalized_arguments["top_k"],
                        )
                    )
                    for row in evidence:
                        retrieved_by_id[str(row["source_id"])] = row
                    output = {"evidence": evidence}

                tool_trace.append({"tool": name, "status": "success"})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call["id"]),
                        "name": name,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
