"""Teacher 输出解析器：从 XML 标签格式中抽取 reasoning / score / risk_level / confidence。

设计成宽容解析（容忍空白、大小写、缺失某段），失败时返回 None 让调用方决定重试。
"""

from __future__ import annotations

import re
from typing import Optional, Dict


_RISK_LEVELS = {"low", "medium", "high", "critical"}


def _extract(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse(text: str) -> Optional[Dict]:
    """解析 Teacher 输出。失败返回 None。"""
    reasoning = _extract(text, "reasoning")
    score_raw = _extract(text, "score")
    level_raw = _extract(text, "risk_level")
    conf_raw = _extract(text, "confidence")

    if reasoning is None or score_raw is None:
        return None

    # score
    try:
        score = float(re.findall(r"-?\d+\.?\d*", score_raw)[0])
    except (IndexError, ValueError):
        return None
    score = max(0.0, min(100.0, score))

    # risk_level
    level = (level_raw or "").strip().lower()
    if level not in _RISK_LEVELS:
        # 用 score 推断兜底
        if score < 25:
            level = "low"
        elif score < 50:
            level = "medium"
        elif score < 75:
            level = "high"
        else:
            level = "critical"

    # confidence
    confidence = 0.7  # 默认值
    if conf_raw:
        try:
            confidence = float(re.findall(r"-?\d+\.?\d*", conf_raw)[0])
            confidence = max(0.0, min(1.0, confidence))
        except (IndexError, ValueError):
            pass

    return {
        "reasoning": reasoning,
        "score": score,
        "risk_level": level,
        "confidence": confidence,
    }
