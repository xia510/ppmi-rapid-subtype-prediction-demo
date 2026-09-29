from datetime import datetime, timezone
import re
import unittest

from app.report import ReportGenerationError, render_research_report


class EvidenceReportTests(unittest.TestCase):
    def setUp(self):
        self.agent_result = {
            "answer_summary": "综合分析支持谨慎解释。",
            "prediction_summary": "模型概率低于研究阈值。",
            "evidence_summary": "文献仅支持群体层面的相关性。",
            "cited_source_ids": ["paper-a-p2-c1"],
            "citations": [
                {
                    "source_id": "paper-a-p2-c1",
                    "title": "Autonomic Study",
                    "source": "autonomic.pdf",
                    "page": 2,
                    "text": "Autonomic burden was associated with progression.",
                    "score": 0.91234,
                }
            ],
            "tool_trace": [
                {"tool": "predict_risk", "status": "success"},
                {"tool": "search_literature", "status": "success"},
            ],
            "research_disclaimer": "仅供科研演示，不用于临床决策。",
            "provider": "DeepSeek",
            "model": "deepseek-flash",
        }

    def test_renders_fixed_sections_verified_citations_and_safe_metadata(self):
        result = render_research_report(
            self.agent_result,
            question="哪些因素与进展相关？",
            patient_id="demo-001",
            generated_at=datetime(2026, 9, 29, 12, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(result["filename"], "ppmi-research-report-20260929T123000Z.md")
        self.assertEqual(result["media_type"], "text/markdown; charset=utf-8")
        self.assertEqual(result["citation_count"], 1)
        self.assertEqual(result["tool_trace"], self.agent_result["tool_trace"])
        self.assertEqual(result["provider"], "DeepSeek")
        self.assertEqual(result["model"], "deepseek-flash")

        markdown = result["markdown"]
        headings = [
            "# PPMI Rapid 亚型科研分析报告",
            "## 1. 研究问题",
            "## 2. 综合结论",
            "## 3. 模型预测摘要",
            "## 4. 文献证据摘要",
            "## 5. 主要参考证据",
            "## 6. 工具执行记录",
            "## 7. 科研用途声明",
        ]
        positions = [markdown.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("样本标识：demo-001", markdown)
        self.assertIn("[1] Autonomic Study", markdown)
        self.assertIn("source_id: `paper-a-p2-c1`", markdown)
        self.assertIn("相关性分数：0.9123", markdown)
        self.assertNotIn("scopa", markdown)
        self.assertNotIn("LEDD", markdown)

    def test_rejects_empty_summary_and_unverified_citation_mapping(self):
        empty_summary = dict(self.agent_result)
        empty_summary["answer_summary"] = "   "
        with self.assertRaises(ReportGenerationError):
            render_research_report(empty_summary, "研究问题")

        mismatched = dict(self.agent_result)
        mismatched["cited_source_ids"] = ["not-retrieved"]
        with self.assertRaises(ReportGenerationError):
            render_research_report(mismatched, "研究问题")

    def test_normalizes_markdown_control_characters_inside_citation_fields(self):
        result_data = dict(self.agent_result)
        result_data["citations"] = [
            {
                **self.agent_result["citations"][0],
                "title": "# Injected heading\nSecond line",
                "text": "Evidence\n## Fake section | value",
            }
        ]

        markdown = render_research_report(result_data, "研究问题")["markdown"]

        self.assertNotRegex(markdown, r"(?m)^# Injected heading$")
        self.assertNotRegex(markdown, r"(?m)^## Fake section")
        self.assertIn(r"\# Injected heading Second line", markdown)
        self.assertIn(r"Evidence \#\# Fake section \| value", markdown)

    def test_filename_contains_only_safe_ascii_characters(self):
        filename = render_research_report(
            self.agent_result,
            question="研究问题",
            patient_id="患者/../../secret",
            generated_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        )["filename"]

        self.assertRegex(filename, r"^[A-Za-z0-9._-]+\.md$")
        self.assertNotIn("患者", filename)
        self.assertNotIn("..", filename)


if __name__ == "__main__":
    unittest.main()
