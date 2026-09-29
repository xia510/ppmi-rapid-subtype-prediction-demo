"""Streamlit dashboard for the PPMI Rapid-subtype prediction demonstration."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

from app.settings import api_base_url


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_DIR / "examples" / "sample_patient.json"

FEATURE_NAMES: list[str] = [
    "scopa",
    "NP1COG",
    "rem",
    "DVT_SFTANIM",
    "MIA_STRIATUM_mean",
    "LEDD",
    "SEX",
    "updrs3_score",
    "MSEADLG",
    "DVT_SDM",
    "upsit_pctl",
    "quip",
]


def build_patient_payload(patient_id: str, raw_values: dict[str, object]) -> dict:
    """Build the fixed-order JSON payload accepted by FastAPI POST /predict."""
    payload = {"patient_id": patient_id}
    for feature in FEATURE_NAMES:
        payload[feature] = float(raw_values[feature])
    return payload


def format_api_error(status_code: int, detail: Any) -> str:
    """Turn expected FastAPI errors into a concise, user-facing message."""
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        error_code = detail["error"].get("code")
        messages = {
            "invalid_input": "输入格式不正确，请检查 12 项特征是否均为有效数字。",
            "model_artifact_unavailable": "后端未找到模型文件。请先运行 scripts/export_model.py。",
            "deepseek_not_configured": (
                "后端尚未配置 DEEPSEEK_API_KEY，请在启动 FastAPI 的终端设置该环境变量后重启服务。"
            ),
            "deepseek_unavailable": "DeepSeek 暂时未能返回可用的科研辅助解读，请稍后重试。",
            "literature_index_unavailable": (
                "后端尚未建立本地文献索引，请先运行 scripts/build_literature_index.py。"
            ),
            "literature_answer_unavailable": "DeepSeek 暂时未能返回有文献依据的回答，请稍后重试。",
            "literature_rag_unavailable": "本地文献检索暂时不可用，请检查索引和Embedding模型。",
            "invalid_literature_question": "文献问题不能为空，Top-K 必须在 1 到 10 之间。",
            "invalid_agent_request": "Agent 输入不正确，请检查患者特征、研究问题和证据数量。",
            "agent_upstream_unavailable": "DeepSeek 暂时无法继续 Agent 分析，请稍后重试。",
            "agent_execution_failed": "Agent 触发了本地安全限制，已停止本次分析。",
            "invalid_report_request": "报告输入不正确，请检查患者特征、研究问题和证据数量。",
            "report_generation_failed": "后端未能生成安全、完整的科研报告，请稍后重试。",
        }
        message = messages.get(error_code, "后端发生了未分类错误，请稍后重试。")
        request_id = detail.get("request_id")
        if request_id:
            return f"{message} 请求编号：{request_id}"
        return message
    if status_code == 422:
        if isinstance(detail, list):
            missing_features = [
                str(item["loc"][-1])
                for item in detail
                if item.get("type") == "missing" and item.get("loc")
            ]
            if missing_features:
                return f"输入不完整：{', '.join(missing_features)} 为必填特征。"
        return "输入格式不正确，请检查 12 项特征是否均为有效数字。"
    if status_code == 503:
        if "DEEPSEEK_API_KEY" in str(detail):
            return "后端尚未配置 DEEPSEEK_API_KEY，请在启动 FastAPI 的终端设置该环境变量后重启服务。"
        return "后端未找到模型文件。请先运行 scripts/export_model.py。"
    return f"预测接口返回错误（HTTP {status_code}）：{detail}"


def health_status() -> tuple[bool, Any]:
    """Query FastAPI health without making a failed local service crash the page."""
    try:
        response = requests.get(f"{api_base_url()}/health", timeout=3)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as error:
        return False, str(error)


def request_prediction(payload: dict) -> tuple[Optional[dict], Optional[str]]:
    """Send a validated page payload to FastAPI and return result or readable error."""
    try:
        response = requests.post(
            f"{api_base_url()}/predict",
            json=payload,
            timeout=10,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None

    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


def request_explanation(payload: dict) -> tuple[Optional[dict], Optional[str]]:
    """Ask the backend to predict locally and request a de-identified DeepSeek explanation."""
    try:
        response = requests.post(
            f"{api_base_url()}/explain",
            json=payload,
            timeout=40,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None

    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


def request_literature_answer(
    question: str,
    top_k: int = 5,
) -> tuple[Optional[dict], Optional[str]]:
    """Ask the FastAPI backend to retrieve local literature and answer from evidence."""
    try:
        response = requests.post(
            f"{api_base_url()}/literature/ask",
            json={"question": question, "top_k": int(top_k)},
            timeout=60,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


def request_agent_analysis(
    payload: dict,
    question: str,
    top_k: int = 5,
) -> tuple[Optional[dict], Optional[str]]:
    """Send patient context and a research task to the bounded backend Agent."""
    request_payload = {
        **payload,
        "question": str(question).strip(),
        "top_k": int(top_k),
    }
    try:
        response = requests.post(
            f"{api_base_url()}/agent/analyze",
            json=request_payload,
            timeout=90,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


def request_research_report(
    payload: dict,
    question: str,
    top_k: int = 5,
) -> tuple[Optional[dict], Optional[str]]:
    """Run the bounded Agent and request a deterministic Markdown report."""
    request_payload = {
        **payload,
        "question": str(question).strip(),
        "top_k": int(top_k),
    }
    try:
        response = requests.post(
            f"{api_base_url()}/report/generate",
            json=request_payload,
            timeout=120,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


def structured_interpretation_sections(interpretation: dict) -> list[tuple[str, str]]:
    """Map validated DeepSeek fields to the stable Chinese headings shown on the page."""
    return [
        ("概率与阈值", interpretation["probability_summary"]),
        ("特征贡献说明", interpretation["contribution_summary"]),
        ("科研使用说明", interpretation["research_disclaimer"]),
    ]


def agent_result_sections(result: dict) -> list[tuple[str, str]]:
    """Map validated Agent fields to stable Chinese dashboard headings."""
    return [
        ("Agent 回答", result["answer_summary"]),
        ("模型预测摘要", result["prediction_summary"]),
        ("文献证据摘要", result["evidence_summary"]),
        ("科研使用说明", result["research_disclaimer"]),
    ]


def load_example() -> dict:
    """Load the tracked, non-sensitive demonstration payload."""
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def _initialize_form_state() -> None:
    example = load_example()
    st.session_state.setdefault("patient_id", example.get("patient_id", "demo-patient"))
    for feature in FEATURE_NAMES:
        st.session_state.setdefault(f"input_{feature}", float(example[feature]))


def _load_example_into_form() -> None:
    example = load_example()
    st.session_state["patient_id"] = example.get("patient_id", "demo-patient")
    for feature in FEATURE_NAMES:
        st.session_state[f"input_{feature}"] = float(example[feature])


def _render_result(result: dict) -> None:
    st.divider()
    st.subheader("预测结果")
    probability, threshold, label = st.columns(3)
    probability.metric("Rapid 概率", f"{result['rapid_probability']:.2%}")
    threshold.metric("研究阈值", f"{result['research_threshold']:.2%}")
    label.metric("研究标签", "低于阈值" if "below" in result["prediction_label"] else "高于阈值")

    if "below" in result["prediction_label"]:
        st.info(result["prediction_label"])
    else:
        st.warning(result["prediction_label"])

    positive, negative = st.columns(2)
    positive.markdown("#### 正向贡献 Top 3")
    positive.dataframe(
        pd.DataFrame(result["top_positive_contributors"])[["feature", "contribution"]],
        hide_index=True,
        width="stretch",
    )
    negative.markdown("#### 负向贡献 Top 3")
    negative.dataframe(
        pd.DataFrame(result["top_negative_contributors"])[["feature", "contribution"]],
        hide_index=True,
        width="stretch",
    )

    with st.expander("查看全部 12 项特征贡献度"):
        st.dataframe(
            pd.DataFrame(result["all_feature_contributions"]),
            hide_index=True,
            width="stretch",
        )
    st.caption(result["explanation_note"])
    st.warning(result["disclaimer"])


def main() -> None:
    st.set_page_config(
        page_title="PPMI Rapid Prediction",
        page_icon="◈",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f7f4ed; color: #14213d; }
        h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #123047; }
        [data-testid="stMetric"] { background: #ffffff; border-left: 4px solid #198f8a;
            padding: 14px; border-radius: 6px; }
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
            background: #123047; color: #ffffff; border: 0; border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _initialize_form_state()

    st.title("PPMI Rapid Subtype Prediction")
    st.caption("Research interface · Calibrated logistic regression · Not for clinical diagnosis")

    available, health = health_status()
    if available and health.get("model_artifact_available"):
        st.success("后端已连接，模型文件可用。")
    elif available:
        st.warning("后端已连接，但尚未找到 model_bundle.joblib。")
    else:
        st.error("无法连接 FastAPI 后端。请先启动 app.api 服务。")

    action_column, note_column = st.columns([1, 3])
    with action_column:
        if st.button("加载示例数据"):
            _load_example_into_form()
            st.rerun()
    with note_column:
        st.caption("示例数据仅用于演示，不代表真实临床建议。")

    st.subheader("受试者原始特征")
    with st.form("prediction_form"):
        patient_id = st.text_input("Patient ID", key="patient_id")
        columns = st.columns(3)
        raw_values: dict[str, float] = {}
        for index, feature in enumerate(FEATURE_NAMES):
            with columns[index % 3]:
                raw_values[feature] = st.number_input(
                    feature,
                    key=f"input_{feature}",
                    format="%.6f",
                )
        submitted = st.form_submit_button("开始预测")

    if submitted:
        payload = build_patient_payload(patient_id, raw_values)
        with st.spinner("正在调用校准模型..."):
            result, error = request_prediction(payload)
        if error:
            st.error(error)
        else:
            st.session_state["prediction_result"] = result
            st.session_state["last_prediction_payload"] = payload
            st.session_state.pop("deepseek_interpretation", None)
            st.session_state.pop("agent_result", None)
            st.session_state.pop("report_result", None)

    if "prediction_result" in st.session_state:
        _render_result(st.session_state["prediction_result"])

        st.subheader("DeepSeek 科研辅助解读")
        st.caption(
            "点击后，后端仅会向 DeepSeek 发送去标识化的概率、阈值和贡献度摘要；"
            "不会发送 Patient ID 或 12 项原始特征。"
        )
        consent = st.checkbox(
            "我理解该操作会调用外部 DeepSeek API，结果仅用于科研演示。",
            key="deepseek_consent",
        )
        if st.button("生成 DeepSeek 辅助解读"):
            if not consent:
                st.warning("请先确认外部 API 调用说明。")
            else:
                with st.spinner("正在请求 DeepSeek 辅助解读..."):
                    explanation, error = request_explanation(
                        st.session_state["last_prediction_payload"]
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state["deepseek_interpretation"] = explanation[
                        "interpretation"
                    ]

        if "deepseek_interpretation" in st.session_state:
            interpretation = st.session_state["deepseek_interpretation"]
            for heading, content in structured_interpretation_sections(interpretation):
                st.markdown(f"#### {heading}")
                if heading == "科研使用说明":
                    st.warning(content)
                else:
                    st.info(content)
            st.caption(
                f"提供方：{interpretation['provider']} · 模型：{interpretation['model']}"
            )
            st.warning(interpretation["disclaimer"])

        st.subheader("Agent 智能分析")
        st.caption(
            "DeepSeek只负责选择本地预测和文献检索工具并整理结果；"
            "Patient ID 与 12 项原始特征不会发送给 DeepSeek。"
        )
        agent_question = st.text_area(
            "Agent 研究任务",
            placeholder="例如：结合模型贡献因素和本地文献，说明这个结果有哪些科研层面的依据与局限？",
            key="agent_question",
        )
        agent_top_k = st.slider(
            "Agent 最多使用的文献证据数量",
            min_value=1,
            max_value=5,
            value=3,
            key="agent_top_k",
        )
        agent_consent = st.checkbox(
            "我理解去标识化模型结果、研究问题和检索文献片段会发送给 DeepSeek。",
            key="agent_consent",
        )
        if st.button("运行 Agent 智能分析"):
            if not agent_question.strip():
                st.warning("请先输入 Agent 研究任务。")
            elif not agent_consent:
                st.warning("请先确认外部 API 调用说明。")
            else:
                with st.spinner("Agent 正在选择并调用本地工具..."):
                    agent_result, error = request_agent_analysis(
                        st.session_state["last_prediction_payload"],
                        agent_question,
                        agent_top_k,
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state["agent_result"] = agent_result

        if "agent_result" in st.session_state:
            agent_result = st.session_state["agent_result"]
            for heading, content in agent_result_sections(agent_result):
                st.markdown(f"#### {heading}")
                if heading == "科研使用说明":
                    st.warning(content)
                else:
                    st.info(content)

            st.markdown("#### Agent 工具轨迹")
            trace_rows = [
                {"顺序": index, "工具": row["tool"], "状态": row["status"]}
                for index, row in enumerate(agent_result["tool_trace"], start=1)
            ]
            if trace_rows:
                st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")
            else:
                st.caption("本次分析未调用本地工具。")

            if agent_result["citations"]:
                st.markdown("#### Agent 使用的文献依据")
                for citation in agent_result["citations"]:
                    label = "{title} · 第 {page} 页 · 相似度 {score:.3f}".format(
                        **citation
                    )
                    with st.expander(label):
                        st.write(citation["text"])
                        st.caption(
                            f"本地文件：{citation['source']} · 来源编号：{citation['source_id']}"
                        )
            st.caption(
                f"提供方：{agent_result['provider']} · 模型：{agent_result['model']}"
            )

        st.subheader("带文献依据的最终报告")
        st.caption(
            "后端会运行受限制 Agent，再用本地固定模板生成 Markdown；"
            "报告引用只能来自本次实际检索并校验过的文献。"
        )
        report_question = st.text_area(
            "报告研究问题",
            placeholder="例如：结合模型结果和本地文献，总结主要科研依据、局限和需要谨慎解释之处。",
            key="report_question",
        )
        report_top_k = st.slider(
            "报告最多使用的文献证据数量",
            min_value=1,
            max_value=5,
            value=3,
            key="report_top_k",
        )
        report_consent = st.checkbox(
            "我理解去标识化模型结果、研究问题和检索文献片段会发送给 DeepSeek。",
            key="report_consent",
        )
        if st.button("生成最终 Markdown 报告"):
            if not report_question.strip():
                st.warning("请先输入报告研究问题。")
            elif not report_consent:
                st.warning("请先确认外部 API 调用说明。")
            else:
                with st.spinner("正在运行 Agent 并生成科研报告..."):
                    report_result, error = request_research_report(
                        st.session_state["last_prediction_payload"],
                        report_question,
                        report_top_k,
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state["report_result"] = report_result

        if "report_result" in st.session_state:
            report_result = st.session_state["report_result"]
            st.markdown("#### 报告预览")
            st.markdown(report_result["markdown"])
            st.caption(
                "引用数量：{count} · 提供方：{provider} · 模型：{model}".format(
                    count=report_result["citation_count"],
                    provider=report_result["provider"],
                    model=report_result["model"],
                )
            )
            st.download_button(
                "下载 Markdown 报告",
                data=report_result["markdown"].encode("utf-8"),
                file_name=report_result["filename"],
                mime=report_result["media_type"],
            )

    st.divider()
    st.subheader("帕金森病医学文献助手")
    st.caption(
        "问题会先检索本地开放文献索引，再把问题与相关证据片段发送给 DeepSeek。"
        "请勿输入姓名、病历号等个人身份信息；回答是群体研究背景，不是对个体患者的诊断或因果判断。"
    )
    question = st.text_area(
        "文献问题",
        placeholder="例如：自主神经功能异常与帕金森病进展有什么关系？",
        key="literature_question",
    )
    top_k = st.slider("检索证据数量", min_value=1, max_value=10, value=5)
    literature_consent = st.checkbox(
        "我理解问题和检索到的公开文献片段会发送给 DeepSeek。",
        key="literature_consent",
    )
    if st.button("检索文献并回答"):
        if not question.strip():
            st.warning("请先输入文献问题。")
        elif not literature_consent:
            st.warning("请先确认外部 API 调用说明。")
        else:
            with st.spinner("正在检索本地文献并生成有依据的回答..."):
                literature_result, error = request_literature_answer(question, top_k)
            if error:
                st.error(error)
            else:
                st.session_state["literature_result"] = literature_result

    if "literature_result" in st.session_state:
        literature_result = st.session_state["literature_result"]
        st.markdown("#### 回答")
        st.info(literature_result["answer"])
        st.markdown("#### 文献依据")
        for citation in literature_result["citations"]:
            label = "{title} · 第 {page} 页 · 相似度 {score:.3f}".format(**citation)
            with st.expander(label):
                st.write(citation["text"])
                st.caption(f"本地文件：{citation['source']} · 来源编号：{citation['source_id']}")
        st.warning(literature_result["evidence_limitations"])
        st.caption(
            f"提供方：{literature_result['provider']} · 模型：{literature_result['model']}"
        )
        st.warning(literature_result["disclaimer"])


if __name__ == "__main__":
    main()
