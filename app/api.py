"""PPMI Rapid 亚型预测演示项目的 FastAPI 后端接口。"""

import logging
import time
from typing import Callable, Optional
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr

from app.agent import AgentExecutionError, ResearchAgent
from app.deepseek import (
    DeepSeekConfigurationError,
    DeepSeekRequestError,
    request_interpretation,
)
from app.predictor import InputValidationError, predict_from_patient
from app.rag import (
    LiteratureIndex,
    LiteratureIndexNotFoundError,
    LiteratureRAG,
    RAGError,
)
from app.report import ReportGenerationError, render_research_report
from app.settings import (
    literature_index_path as configured_literature_index_path,
    model_artifact_path,
)


# 创建本模块专用的日志记录器，用于记录请求编号、接口路径、状态码和耗时。
LOGGER = logging.getLogger("ppmi.api")


class PatientRequest(BaseModel):
    """定义 /predict 和 /explain 接收的患者 JSON 格式。"""

    # patient_id 只用于标识和结果展示，不作为模型特征，因此允许为空。
    patient_id: Optional[str] = None
    # 以下 12 项是模型必需的原始特征，Pydantic 会检查它们能否转换为浮点数。
    scopa: float
    NP1COG: float
    rem: float
    DVT_SFTANIM: float
    MIA_STRIATUM_mean: float
    LEDD: float
    SEX: float
    updrs3_score: float
    MSEADLG: float
    DVT_SDM: float
    upsit_pctl: float
    quip: float


class LiteratureQuestionRequest(BaseModel):
    """Define a bounded research-literature question."""

    question: constr(strip_whitespace=True, min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class AgentAnalyzeRequest(PatientRequest):
    """Define patient inputs plus the bounded research task for the Agent."""

    question: constr(strip_whitespace=True, min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=5)


class ReportGenerateRequest(AgentAnalyzeRequest):
    """Define patient inputs and research options for a Markdown report."""


def _prediction_for_external_explanation(prediction: dict) -> dict:
    """只保留外部大模型解释所需的去标识化派生结果。"""
    # 使用“允许列表”重新构造字典，避免 patient_id、原始特征值和全部贡献明细外传。
    return {
        "rapid_probability": prediction["rapid_probability"],
        "research_threshold": prediction["research_threshold"],
        "prediction_label": prediction["prediction_label"],
        "model_version": prediction["model_version"],
        "top_positive_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction["top_positive_contributors"]
        ],
        "top_negative_contributors": [
            {"feature": row["feature"], "contribution": row["contribution"]}
            for row in prediction["top_negative_contributors"]
        ],
    }


def _request_id(request: Request) -> str:
    """读取中间件写入当前请求状态的请求编号。"""
    return str(getattr(request.state, "request_id", "unknown"))


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    """生成结构稳定且不暴露内部异常细节的 JSON 错误响应。"""
    request_id = _request_id(request)
    # 日志只记录请求编号和安全错误码，不记录患者原始数据或 API Key。
    LOGGER.warning("api_error request_id=%s code=%s", request_id, code)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": request_id,
        },
    )


def create_app(
    artifact_path: Optional[Path] = None,
    explainer: Optional[Callable[[dict], dict]] = None,
    literature_service: Optional[object] = None,
    literature_index_path: Optional[Path] = None,
    agent_service: Optional[object] = None,
    report_renderer: Optional[Callable[..., dict]] = None,
) -> FastAPI:
    """创建 API 应用，并允许测试注入临时模型路径和假解释器。"""
    # 正常运行时从 settings.py 读取模型位置；测试时可以传入临时模型文件。
    resolved_artifact_path = Path(artifact_path or model_artifact_path())
    # 正常运行时调用 DeepSeek；测试时可注入假函数，避免真实联网和消耗余额。
    resolved_explainer = explainer or request_interpretation
    resolved_literature_index_path = Path(
        literature_index_path or configured_literature_index_path()
    )
    literature_service_holder = {"service": literature_service}
    agent_service_holder = {"service": agent_service}
    resolved_report_renderer = report_renderer or render_research_report
    api = FastAPI(
        title="PPMI Rapid Subtype Prediction Demo",
        description="Research demonstration API. Not for clinical diagnosis.",
        version="research-demo-v1",
    )

    @api.middleware("http")
    async def observe_request(request: Request, call_next):
        """为每个请求添加编号，并记录安全的请求元数据和处理耗时。"""
        # uuid4 生成随机编号，只截取前 12 位，方便网页错误与后端日志对应。
        request.state.request_id = uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        # call_next 会把请求继续交给 /health、/predict 或 /explain 等具体接口。
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started_at) * 1000
        # 请求编号同时放入响应头，客户端无需解析正文也能读取。
        response.headers["X-Request-ID"] = request.state.request_id
        LOGGER.info(
            "api_request request_id=%s method=%s path=%s status_code=%s duration_ms=%.1f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @api.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        """将 Pydantic/FastAPI 输入校验错误转换为统一的 422 响应。"""
        # 不直接返回复杂的框架异常，网页只依赖稳定的 invalid_input 错误码。
        is_literature_request = request.url.path == "/literature/ask"
        is_agent_request = request.url.path == "/agent/analyze"
        is_report_request = request.url.path == "/report/generate"
        return _error_response(
            request,
            status_code=422,
            code=(
                "invalid_report_request"
                if is_report_request
                else "invalid_agent_request"
                if is_agent_request
                else "invalid_literature_question"
                if is_literature_request
                else "invalid_input"
            ),
            message=(
                "Report patient input, question, or top_k is invalid."
                if is_report_request
                else "Agent patient input, question, or top_k is invalid."
                if is_agent_request
                else (
                    "Literature question or top_k is invalid."
                    if is_literature_request
                    else "Input is incomplete or has an invalid numeric value."
                )
            ),
        )

    @api.get("/health")
    def health() -> dict[str, object]:
        """返回服务状态，并检查模型文件是否存在，但不真正加载模型。"""
        return {
            "status": "ok",
            "service": "ppmi-rapid-subtype-prediction-demo",
            "feature_count": 12,
            "model_artifact_available": resolved_artifact_path.exists(),
            "literature_index_available": (
                (resolved_literature_index_path / "metadata.json").exists()
                and (resolved_literature_index_path / "embeddings.npy").exists()
            ),
        }

    def resolve_literature_service():
        service = literature_service_holder["service"]
        if service is None:
            service = LiteratureRAG(LiteratureIndex.load(resolved_literature_index_path))
            literature_service_holder["service"] = service
        return service

    def resolve_agent_service():
        service = agent_service_holder["service"]
        if service is None:
            index = LiteratureIndex.load(resolved_literature_index_path)
            service = ResearchAgent(
                predictor=lambda patient: predict_from_patient(
                    patient, resolved_artifact_path
                ),
                literature_index=index,
            )
            agent_service_holder["service"] = service
        return service

    def agent_failure_response(
        error: Exception,
        request: Request,
        invalid_code: str,
        context: str,
    ) -> JSONResponse:
        """Map shared Agent dependencies to stable, privacy-safe API errors."""
        if isinstance(error, FileNotFoundError):
            return _error_response(
                request,
                status_code=503,
                code="model_artifact_unavailable",
                message="Model artifact is unavailable. Export it before using the Agent.",
            )
        if isinstance(error, LiteratureIndexNotFoundError):
            return _error_response(
                request,
                status_code=503,
                code="literature_index_unavailable",
                message="Local literature index is unavailable.",
            )
        if isinstance(error, RAGError):
            return _error_response(
                request,
                status_code=503,
                code="literature_rag_unavailable",
                message="The local literature RAG service is unavailable.",
            )
        if isinstance(error, DeepSeekConfigurationError):
            return _error_response(
                request,
                status_code=503,
                code="deepseek_not_configured",
                message="DeepSeek API key is not configured.",
            )
        if isinstance(error, DeepSeekRequestError):
            LOGGER.warning(
                "%s_upstream_failure request_id=%s reason=%s",
                context,
                _request_id(request),
                error,
            )
            return _error_response(
                request,
                status_code=502,
                code="agent_upstream_unavailable",
                message="DeepSeek could not continue the Agent run.",
            )
        if isinstance(error, AgentExecutionError):
            LOGGER.warning(
                "%s_execution_failure request_id=%s reason=%s",
                context,
                _request_id(request),
                error,
            )
            return _error_response(
                request,
                status_code=502,
                code="agent_execution_failed",
                message="The Agent stopped at a local safety boundary.",
            )
        return _error_response(
            request,
            status_code=422,
            code=invalid_code,
            message="Patient input contains an invalid value.",
        )

    @api.post("/literature/ask")
    def ask_literature(payload: LiteratureQuestionRequest, request: Request) -> object:
        """Retrieve local PDF evidence and request a citation-checked DeepSeek answer."""
        try:
            service = resolve_literature_service()
            return service.ask(payload.question, top_k=payload.top_k)
        except LiteratureIndexNotFoundError:
            return _error_response(
                request,
                status_code=503,
                code="literature_index_unavailable",
                message="Local literature index is unavailable.",
            )
        except DeepSeekConfigurationError:
            return _error_response(
                request,
                status_code=503,
                code="deepseek_not_configured",
                message="DeepSeek API key is not configured.",
            )
        except DeepSeekRequestError as error:
            # 仅在服务端记录经过代码控制的失败原因，便于用请求编号排查；
            # 不记录 API Key、完整提示词、文献正文或 DeepSeek 原始响应。
            LOGGER.warning(
                "literature_answer_failure request_id=%s reason=%s",
                _request_id(request),
                error,
            )
            return _error_response(
                request,
                status_code=502,
                code="literature_answer_unavailable",
                message="DeepSeek could not return a grounded literature answer.",
            )
        except (RAGError, ValueError):
            return _error_response(
                request,
                status_code=503,
                code="literature_rag_unavailable",
                message="The local literature RAG service is unavailable.",
            )

    @api.post("/agent/analyze")
    def analyze_with_agent(payload: AgentAnalyzeRequest, request: Request) -> object:
        """Run the bounded Agent over local prediction and literature tools."""
        patient = payload.model_dump(exclude={"question", "top_k"})
        try:
            service = resolve_agent_service()
            return service.run(patient, payload.question, top_k=payload.top_k)
        except (
            FileNotFoundError,
            LiteratureIndexNotFoundError,
            RAGError,
            DeepSeekConfigurationError,
            DeepSeekRequestError,
            AgentExecutionError,
            InputValidationError,
        ) as error:
            return agent_failure_response(
                error,
                request,
                invalid_code="invalid_agent_request",
                context="agent",
            )

    @api.post("/report/generate")
    def generate_report(payload: ReportGenerateRequest, request: Request) -> object:
        """Run the bounded Agent and render its verified result as Markdown."""
        patient = payload.model_dump(exclude={"question", "top_k"})
        try:
            service = resolve_agent_service()
            agent_question = (
                "生成最终科研报告前，必须调用 predict_risk 和 search_literature，"
                f"然后完成以下研究任务：{payload.question}"
            )
            agent_result = service.run(
                patient,
                agent_question,
                top_k=payload.top_k,
            )
            return resolved_report_renderer(
                agent_result,
                payload.question,
                patient_id=payload.patient_id,
            )
        except ReportGenerationError as error:
            LOGGER.warning(
                "report_generation_failure request_id=%s reason=%s",
                _request_id(request),
                error,
            )
            return _error_response(
                request,
                status_code=502,
                code="report_generation_failed",
                message="A safe research report could not be generated.",
            )
        except (
            FileNotFoundError,
            LiteratureIndexNotFoundError,
            RAGError,
            DeepSeekConfigurationError,
            DeepSeekRequestError,
            AgentExecutionError,
            InputValidationError,
        ) as error:
            return agent_failure_response(
                error,
                request,
                invalid_code="invalid_report_request",
                context="report_agent",
            )

    @api.post("/predict")
    def predict(patient: PatientRequest, request: Request) -> object:
        """接收一名患者的原始特征，返回校准概率、阈值和特征贡献度。"""
        try:
            # model_dump() 把 Pydantic 对象转换为 predictor.py 接受的普通字典。
            return predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError:
            # 服务仍在运行，但模型包不存在，因此使用 503 表示当前无法完成预测。
            return _error_response(
                request,
                status_code=503,
                code="model_artifact_unavailable",
                message="Model artifact is unavailable. Export it before starting the API.",
            )
        except InputValidationError:
            # predictor.py 的第二层输入校验失败，例如缺失特征或存在非有限数值。
            return _error_response(
                request,
                status_code=422,
                code="invalid_input",
                message="Input is incomplete or has an invalid numeric value.",
            )

    @api.post("/explain")
    def explain(patient: PatientRequest, request: Request) -> object:
        """先在本地重新预测，再让 DeepSeek 解释去标识化的模型结果。"""
        try:
            # 后端不信任客户端传来的中间结果，必须使用原始输入重新生成预测。
            prediction = predict_from_patient(patient.model_dump(), resolved_artifact_path)
        except FileNotFoundError:
            return _error_response(
                request,
                status_code=503,
                code="model_artifact_unavailable",
                message="Model artifact is unavailable. Export it before starting the API.",
            )
        except InputValidationError:
            return _error_response(
                request,
                status_code=422,
                code="invalid_input",
                message="Input is incomplete or has an invalid numeric value.",
            )

        try:
            # 先通过允许列表脱敏，再调用解释器；DeepSeek 不参与概率计算。
            interpretation = resolved_explainer(_prediction_for_external_explanation(prediction))
        except DeepSeekConfigurationError:
            # 没有在启动 FastAPI 的终端配置 DEEPSEEK_API_KEY。
            return _error_response(
                request,
                status_code=503,
                code="deepseek_not_configured",
                message="DeepSeek API key is not configured.",
            )
        except DeepSeekRequestError:
            # 外部请求失败，或 DeepSeek 返回内容未通过 JSON/Pydantic 校验。
            return _error_response(
                request,
                status_code=502,
                code="deepseek_unavailable",
                message="DeepSeek could not return a usable research explanation.",
            )

        # 同时返回本地原始预测和经过结构校验的中文辅助解释。
        return {"prediction": prediction, "interpretation": interpretation}

    # 把已注册中间件、异常处理器和所有接口的 FastAPI 对象返回。
    return api


# Uvicorn 使用 app.api:app 启动时，会读取这里创建好的应用对象。
app = create_app()
