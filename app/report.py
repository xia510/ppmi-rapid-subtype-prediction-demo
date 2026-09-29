"""Deterministic Markdown reports built from validated Agent results."""

from datetime import datetime, timezone
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, constr


NonEmptyText = constr(strip_whitespace=True, min_length=1)


class ReportGenerationError(RuntimeError):
    """Raised when an Agent result cannot be rendered as a safe report."""


class ReportCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: NonEmptyText
    title: NonEmptyText
    source: NonEmptyText
    page: int = Field(ge=1)
    text: NonEmptyText
    score: float


class ReportToolTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: NonEmptyText
    status: NonEmptyText


class ReportAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_summary: NonEmptyText
    prediction_summary: NonEmptyText
    evidence_summary: NonEmptyText
    cited_source_ids: list[NonEmptyText]
    citations: list[ReportCitation]
    tool_trace: list[ReportToolTrace]
    research_disclaimer: NonEmptyText
    provider: NonEmptyText
    model: NonEmptyText


_MARKDOWN_INLINE_CHARACTERS = re.compile(r"([\\`#|])")


def _inline_text(value: object) -> str:
    """Collapse untrusted text to one escaped Markdown line."""
    collapsed = " ".join(str(value).split())
    return _MARKDOWN_INLINE_CHARACTERS.sub(r"\\\1", collapsed)


def _utc_timestamp(value: Optional[datetime]) -> datetime:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def render_research_report(
    agent_result: dict[str, Any],
    question: str,
    patient_id: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Validate an Agent result and render a fixed research-only report."""
    try:
        validated = ReportAgentResult.model_validate(agent_result)
    except (TypeError, ValidationError) as error:
        raise ReportGenerationError("Agent result is invalid for reporting.") from error

    normalized_question = str(question).strip()
    if not normalized_question:
        raise ReportGenerationError("Report question must not be empty.")

    cited_ids = list(validated.cited_source_ids)
    citation_ids = [citation.source_id for citation in validated.citations]
    if cited_ids != citation_ids or len(citation_ids) != len(set(citation_ids)):
        raise ReportGenerationError("Report citations do not match verified source IDs.")

    timestamp = _utc_timestamp(generated_at)
    generated_label = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    filename = f"ppmi-research-report-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.md"

    metadata = [f"- 生成时间：{generated_label}"]
    if patient_id is not None and str(patient_id).strip():
        metadata.append(f"- 样本标识：{_inline_text(patient_id)}")
    metadata.extend(
        [
            f"- 分析服务：{_inline_text(validated.provider)}",
            f"- 模型：{_inline_text(validated.model)}",
        ]
    )

    if validated.citations:
        citation_blocks = []
        for index, citation in enumerate(validated.citations, start=1):
            citation_blocks.append(
                "\n".join(
                    [
                        f"### [{index}] {_inline_text(citation.title)}",
                        (
                            f"- 来源：{_inline_text(citation.source)}；页码：{citation.page}；"
                            f"相关性分数：{citation.score:.4f}"
                        ),
                        f"- source_id: `{_inline_text(citation.source_id)}`",
                        f"- 证据片段：{_inline_text(citation.text)}",
                    ]
                )
            )
        citations_markdown = "\n\n".join(citation_blocks)
    else:
        citations_markdown = "本次分析未引用本地文献证据。"

    if validated.tool_trace:
        tool_markdown = "\n".join(
            f"{index}. `{_inline_text(row.tool)}` — {_inline_text(row.status)}"
            for index, row in enumerate(validated.tool_trace, start=1)
        )
    else:
        tool_markdown = "本次分析未记录工具调用。"

    markdown = "\n\n".join(
        [
            "# PPMI Rapid 亚型科研分析报告",
            "\n".join(metadata),
            "## 1. 研究问题\n\n" + _inline_text(normalized_question),
            "## 2. 综合结论\n\n" + _inline_text(validated.answer_summary),
            "## 3. 模型预测摘要\n\n" + _inline_text(validated.prediction_summary),
            "## 4. 文献证据摘要\n\n" + _inline_text(validated.evidence_summary),
            "## 5. 主要参考证据\n\n" + citations_markdown,
            "## 6. 工具执行记录\n\n" + tool_markdown,
            "## 7. 科研用途声明\n\n" + _inline_text(validated.research_disclaimer),
        ]
    ) + "\n"

    return {
        "filename": filename,
        "media_type": "text/markdown; charset=utf-8",
        "markdown": markdown,
        "citation_count": len(validated.citations),
        "tool_trace": [row.model_dump() for row in validated.tool_trace],
        "provider": validated.provider,
        "model": validated.model,
    }
