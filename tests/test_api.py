import sys
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.api import app, create_app
from app.agent import AgentExecutionError
from app.deepseek import DeepSeekConfigurationError, DeepSeekRequestError
from app.rag import LiteratureIndexNotFoundError, RAGError
from app.report import ReportGenerationError
from scripts.export_model import export_model_bundle
from tests.data_support import authorized_source_root


SOURCE_ROOT = authorized_source_root()


@unittest.skipUnless(SOURCE_ROOT, "Set PPMI_TEST_SOURCE_ROOT for API model tests.")
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
        self.assertRegex(response.headers.get("X-Request-ID", ""), r"^[0-9a-f]{12}$")
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
        payload = response.json()
        self.assertIn("error", payload)
        self.assertEqual(payload.get("error", {}).get("code"), "invalid_input")
        self.assertRegex(payload.get("request_id", ""), r"^[0-9a-f]{12}$")

    def test_explain_reports_a_missing_key_with_a_safe_error_code(self):
        def missing_key_explainer(prediction):
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured.")

        client = TestClient(create_app(self.artifact_path, missing_key_explainer))
        response = client.post("/explain", json=self.valid_patient)

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertIn("error", payload)
        self.assertEqual(payload.get("error", {}).get("code"), "deepseek_not_configured")
        self.assertRegex(payload.get("request_id", ""), r"^[0-9a-f]{12}$")

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


class LiteratureApiTests(unittest.TestCase):
    def test_literature_ask_logs_safe_upstream_failure_reason(self):
        class FailingLiteratureService:
            def ask(self, question, top_k=5):
                raise DeepSeekRequestError("DeepSeek returned empty content.")

        client = TestClient(
            create_app(
                Path("missing-model.joblib"),
                literature_service=FailingLiteratureService(),
            )
        )

        with self.assertLogs("ppmi.api", level="WARNING") as captured:
            response = client.post(
                "/literature/ask",
                json={"question": "自主神经症状是否与进展有关？", "top_k": 3},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("DeepSeek returned empty content.", "\n".join(captured.output))

    def test_literature_ask_returns_grounded_answer_without_model_artifact(self):
        class FakeLiteratureService:
            def ask(self, question, top_k=5):
                return {
                    "question": question,
                    "answer": "检索证据支持群体层面的相关性描述。",
                    "evidence_limitations": "不能推断个体因果关系。",
                    "citations": [
                        {
                            "source_id": "paper-a-p2-c1",
                            "title": "Autonomic Study",
                            "source": "autonomic.pdf",
                            "page": 2,
                            "text": "Evidence text.",
                            "score": 0.9,
                        }
                    ],
                    "provider": "DeepSeek",
                    "model": "test-model",
                    "disclaimer": "仅供科研文献辅助阅读。",
                }

        client = TestClient(
            create_app(
                Path("missing-model.joblib"),
                literature_service=FakeLiteratureService(),
            )
        )
        response = client.post(
            "/literature/ask",
            json={"question": "自主神经症状是否与进展有关？", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["page"], 2)
        self.assertRegex(response.headers.get("X-Request-ID", ""), r"^[0-9a-f]{12}$")

    def test_literature_ask_reports_missing_local_index(self):
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(
                create_app(
                    Path("missing-model.joblib"),
                    literature_index_path=Path(directory),
                )
            )
            response = client.post(
                "/literature/ask",
                json={"question": "What evidence is available?", "top_k": 3},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "literature_index_unavailable")

    def test_literature_ask_rejects_empty_question_with_specific_error(self):
        client = TestClient(create_app(Path("missing-model.joblib")))
        response = client.post(
            "/literature/ask",
            json={"question": "   ", "top_k": 3},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_literature_question")


class AgentApiTests(unittest.TestCase):
    patient = {
        "patient_id": "agent-api-demo",
        "scopa": 1.0,
        "NP1COG": 2.0,
        "rem": 3.0,
        "DVT_SFTANIM": 4.0,
        "MIA_STRIATUM_mean": 5.0,
        "LEDD": 6.0,
        "SEX": 1.0,
        "updrs3_score": 8.0,
        "MSEADLG": 9.0,
        "DVT_SDM": 10.0,
        "upsit_pctl": 11.0,
        "quip": 12.0,
    }

    def test_agent_analyze_returns_structured_result_from_injected_service(self):
        received = {}

        class FakeAgentService:
            def run(self, patient, question, top_k=5):
                received.update({"patient": patient, "question": question, "top_k": top_k})
                return {
                    "answer_summary": "完成研究分析。",
                    "prediction_summary": "概率低于阈值。",
                    "evidence_summary": "证据为群体层面。",
                    "cited_source_ids": ["paper-a-p2-c1"],
                    "citations": [{"source_id": "paper-a-p2-c1", "page": 2}],
                    "tool_trace": [{"tool": "predict_risk", "status": "success"}],
                    "research_disclaimer": "仅供科研演示。",
                    "provider": "DeepSeek",
                    "model": "test-model",
                }

        client = TestClient(
            create_app(Path("missing-model.joblib"), agent_service=FakeAgentService())
        )
        response = client.post(
            "/agent/analyze",
            json={**self.patient, "question": "结合模型和文献分析", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer_summary"], "完成研究分析。")
        self.assertEqual(received["question"], "结合模型和文献分析")
        self.assertEqual(received["top_k"], 3)
        self.assertNotIn("question", received["patient"])
        self.assertNotIn("top_k", received["patient"])
        self.assertRegex(response.headers.get("X-Request-ID", ""), r"^[0-9a-f]{12}$")

    def test_agent_analyze_uses_specific_validation_error(self):
        client = TestClient(
            create_app(Path("missing-model.joblib"), agent_service=object())
        )
        response = client.post(
            "/agent/analyze",
            json={**self.patient, "question": " ", "top_k": 6},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_agent_request")

    def test_agent_analyze_maps_missing_resources(self):
        cases = [
            (FileNotFoundError(), "model_artifact_unavailable"),
            (LiteratureIndexNotFoundError("missing"), "literature_index_unavailable"),
            (RAGError("dimension mismatch"), "literature_rag_unavailable"),
            (DeepSeekConfigurationError("missing"), "deepseek_not_configured"),
        ]
        for error, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                class FailingAgent:
                    def run(self, patient, question, top_k=5):
                        raise error

                client = TestClient(
                    create_app(Path("missing-model.joblib"), agent_service=FailingAgent())
                )
                response = client.post(
                    "/agent/analyze",
                    json={**self.patient, "question": "执行研究分析", "top_k": 3},
                )

                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], expected_code)
                self.assertRegex(response.json()["request_id"], r"^[0-9a-f]{12}$")

    def test_agent_analyze_maps_provider_and_execution_failures(self):
        cases = [
            (DeepSeekRequestError("upstream"), "agent_upstream_unavailable"),
            (AgentExecutionError("unknown tool"), "agent_execution_failed"),
        ]
        for error, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                class FailingAgent:
                    def run(self, patient, question, top_k=5):
                        raise error

                client = TestClient(
                    create_app(Path("missing-model.joblib"), agent_service=FailingAgent())
                )
                response = client.post(
                    "/agent/analyze",
                    json={**self.patient, "question": "执行研究分析", "top_k": 3},
                )

                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["error"]["code"], expected_code)
                self.assertRegex(response.json()["request_id"], r"^[0-9a-f]{12}$")


class ReportApiTests(unittest.TestCase):
    patient = AgentApiTests.patient

    def test_report_generate_uses_agent_result_and_returns_markdown_metadata(self):
        received = {}
        agent_result = {
            "answer_summary": "完成研究分析。",
            "prediction_summary": "概率低于阈值。",
            "evidence_summary": "证据为群体层面。",
            "cited_source_ids": [],
            "citations": [],
            "tool_trace": [{"tool": "predict_risk", "status": "success"}],
            "research_disclaimer": "仅供科研演示。",
            "provider": "DeepSeek",
            "model": "test-model",
        }

        class FakeAgentService:
            def run(self, patient, question, top_k=5):
                received["agent"] = {
                    "patient": patient,
                    "question": question,
                    "top_k": top_k,
                }
                return agent_result

        def fake_renderer(result, question, patient_id=None):
            received["renderer"] = {
                "result": result,
                "question": question,
                "patient_id": patient_id,
            }
            return {
                "filename": "ppmi-research-report-test.md",
                "media_type": "text/markdown; charset=utf-8",
                "markdown": "# 报告\n",
                "citation_count": 0,
                "tool_trace": result["tool_trace"],
                "provider": result["provider"],
                "model": result["model"],
            }

        client = TestClient(
            create_app(
                Path("missing-model.joblib"),
                agent_service=FakeAgentService(),
                report_renderer=fake_renderer,
            )
        )
        response = client.post(
            "/report/generate",
            json={**self.patient, "question": "生成科研报告", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["markdown"], "# 报告\n")
        self.assertEqual(response.json()["filename"], "ppmi-research-report-test.md")
        self.assertNotIn("question", received["agent"]["patient"])
        self.assertNotIn("top_k", received["agent"]["patient"])
        self.assertEqual(received["agent"]["top_k"], 3)
        self.assertIs(received["renderer"]["result"], agent_result)
        self.assertEqual(received["renderer"]["patient_id"], "agent-api-demo")
        self.assertNotIn("scopa", received["renderer"])
        self.assertRegex(response.headers.get("X-Request-ID", ""), r"^[0-9a-f]{12}$")

    def test_report_generate_uses_specific_validation_error(self):
        client = TestClient(
            create_app(Path("missing-model.joblib"), agent_service=object())
        )

        response = client.post(
            "/report/generate",
            json={**self.patient, "question": " ", "top_k": 7},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_report_request")

    def test_report_generate_reuses_safe_agent_error_codes(self):
        cases = [
            (FileNotFoundError(), 503, "model_artifact_unavailable"),
            (LiteratureIndexNotFoundError("missing"), 503, "literature_index_unavailable"),
            (RAGError("dimension mismatch"), 503, "literature_rag_unavailable"),
            (DeepSeekConfigurationError("missing"), 503, "deepseek_not_configured"),
            (DeepSeekRequestError("upstream"), 502, "agent_upstream_unavailable"),
            (AgentExecutionError("unknown tool"), 502, "agent_execution_failed"),
        ]
        for error, expected_status, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                class FailingAgent:
                    def run(self, patient, question, top_k=5):
                        raise error

                client = TestClient(
                    create_app(
                        Path("missing-model.joblib"),
                        agent_service=FailingAgent(),
                    )
                )
                response = client.post(
                    "/report/generate",
                    json={**self.patient, "question": "生成科研报告", "top_k": 3},
                )

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json()["error"]["code"], expected_code)
                self.assertRegex(response.json()["request_id"], r"^[0-9a-f]{12}$")

    def test_report_generate_maps_renderer_failure(self):
        class FakeAgentService:
            def run(self, patient, question, top_k=5):
                return {"answer_summary": "invalid for report"}

        def failing_renderer(result, question, patient_id=None):
            raise ReportGenerationError("missing citations")

        client = TestClient(
            create_app(
                Path("missing-model.joblib"),
                agent_service=FakeAgentService(),
                report_renderer=failing_renderer,
            )
        )
        response = client.post(
            "/report/generate",
            json={**self.patient, "question": "生成科研报告", "top_k": 3},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "report_generation_failed")
        self.assertRegex(response.json()["request_id"], r"^[0-9a-f]{12}$")

if __name__ == "__main__":
    unittest.main()
