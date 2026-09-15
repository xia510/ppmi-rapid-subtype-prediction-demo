"""Minimal DeepSeek client for research-only interpretation of model outputs."""

import json
import os
from typing import Any, Callable, Dict, Optional

import requests


DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
SYSTEM_PROMPT = """你是科研模型输出的辅助解释助手。只根据提供的模型结果进行中文转述。
不得给出诊断、治疗、用药或个人医疗建议；不得将特征贡献解释为因果关系；必须提醒结果仅供科研演示。"""


class DeepSeekConfigurationError(RuntimeError):
    """Raised when the local API key has not been configured."""


class DeepSeekRequestError(RuntimeError):
    """Raised when DeepSeek cannot return a usable completion."""


def _contributor_lines(contributors: list[dict]) -> str:
    if not contributors:
        return "无"
    return "；".join(
        "{feature}（贡献度 {contribution:.4f}）".format(
            feature=row["feature"], contribution=float(row["contribution"])
        )
        for row in contributors
    )


def build_interpretation_request(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Create a de-identified, bounded DeepSeek request from model output only."""
    user_prompt = """以下是一个科研演示模型的去标识化输出：
- Rapid 概率：{probability:.2%}
- 研究阈值：{threshold:.2%}
- 研究标签：{label}
- 正向贡献 Top 特征：{positive}
- 负向贡献 Top 特征：{negative}

请用中文生成三段简短内容：
1. 概率与研究阈值的关系；
2. 正负贡献特征如何影响该模型的线性得分；
3. 明确说明这不是临床诊断、贡献度不是因果关系。
不得补充未提供的数据，不得给出医疗建议。""".format(
        probability=float(prediction["rapid_probability"]),
        threshold=float(prediction["research_threshold"]),
        label=prediction["prediction_label"],
        positive=_contributor_lines(prediction.get("top_positive_contributors", [])),
        negative=_contributor_lines(prediction.get("top_negative_contributors", [])),
    )
    return {
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
        "thinking": {"type": "disabled"},
        "stream": False,
    }


def request_interpretation(
    prediction: Dict[str, Any],
    api_key: Optional[str] = None,
    http_post: Callable = requests.post,
) -> Dict[str, str]:
    """Request a concise research interpretation without sending raw inputs or patient ID."""
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_key:
        raise DeepSeekConfigurationError(
            "DEEPSEEK_API_KEY is not configured on the API service."
        )

    payload = build_interpretation_request(prediction)
    try:
        response = http_post(
            DEEPSEEK_CHAT_URL,
            headers={
                "Authorization": "Bearer {key}".format(key=resolved_key),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        response_data = response.json()
        text = response_data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        raise DeepSeekRequestError("DeepSeek did not return a usable interpretation.") from error

    if not text:
        raise DeepSeekRequestError("DeepSeek returned an empty interpretation.")

    return {
        "provider": "DeepSeek",
        "model": str(response_data.get("model", payload["model"])),
        "text": text,
        "disclaimer": "AI-generated research explanation only. Not clinical advice.",
    }


def request_payload_for_audit(prediction: Dict[str, Any]) -> str:
    """Return serialised request content for tests and local privacy review only."""
    return json.dumps(build_interpretation_request(prediction), ensure_ascii=False)
