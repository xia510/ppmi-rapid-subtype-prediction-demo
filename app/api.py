"""PPMI Rapid 亚型预测演示项目的 FastAPI 后端接口。"""

import logging
import time
from typing import Callable, Optional
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.deepseek import (
    DeepSeekConfigurationError,
    DeepSeekRequestError,
    request_interpretation,
)
from app.predictor import InputValidationError, predict_from_patient
from app.settings import model_artifact_path


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
) -> FastAPI:
    """创建 API 应用，并允许测试注入临时模型路径和假解释器。"""
    # 正常运行时从 settings.py 读取模型位置；测试时可以传入临时模型文件。
    resolved_artifact_path = Path(artifact_path or model_artifact_path())
    # 正常运行时调用 DeepSeek；测试时可注入假函数，避免真实联网和消耗余额。
    resolved_explainer = explainer or request_interpretation
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
        return _error_response(
            request,
            status_code=422,
            code="invalid_input",
            message="Input is incomplete or has an invalid numeric value.",
        )

    @api.get("/health")
    def health() -> dict[str, object]:
        """返回服务状态，并检查模型文件是否存在，但不真正加载模型。"""
        return {
            "status": "ok",
            "service": "ppmi-rapid-subtype-prediction-demo",
            "feature_count": 12,
            "model_artifact_available": resolved_artifact_path.exists(),
        }

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

    # 把已注册中间件、异常处理器和三个接口的 FastAPI 对象返回。
    return api


# Uvicorn 使用 app.api:app 启动时，会读取这里创建好的应用对象。
app = create_app()
