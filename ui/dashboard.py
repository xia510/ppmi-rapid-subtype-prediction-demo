"""Streamlit dashboard for the PPMI Rapid-subtype prediction demonstration."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_DIR / "examples" / "sample_patient.json"
API_BASE_URL = "http://127.0.0.1:8000"

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
        return "后端未找到模型文件。请先运行 scripts/export_model.py。"
    return f"预测接口返回错误（HTTP {status_code}）：{detail}"


def health_status() -> tuple[bool, Any]:
    """Query FastAPI health without making a failed local service crash the page."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as error:
        return False, str(error)


def request_prediction(payload: dict) -> tuple[Optional[dict], Optional[str]]:
    """Send a validated page payload to FastAPI and return result or readable error."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10,
        )
    except requests.RequestException:
        return None, "无法连接预测后端。请确认 FastAPI 服务正在 127.0.0.1:8000 运行。"

    if response.ok:
        return response.json(), None

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, format_api_error(response.status_code, detail)


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

    if "prediction_result" in st.session_state:
        _render_result(st.session_state["prediction_result"])


if __name__ == "__main__":
    main()
