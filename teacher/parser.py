"""Teacher 输出解析器。

从 XML 标签格式中抽取：event_category / reasoning / score / risk_level / confidence。
设计成宽容解析（容忍空白、大小写、缺失某段），失败时返回 None 让调用方决定重试。
"""

from __future__ import annotations

import re
from typing import Optional, Dict


_RISK_LEVELS = {"一般", "较大", "重大", "特别重大"}
_EVENT_CATEGORIES = {"自然灾害", "事故灾难", "公共卫生事件", "社会安全事件"}


def _extract(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _level_from_score(score: float) -> str:
    if score < 25:
        return "一般"
    if score < 50:
        return "较大"
    if score < 75:
        return "重大"
    return "特别重大"


def _normalize_level(raw: Optional[str], score: float) -> str:
    """风险等级强制按 score 25/50/75 反查（与 prompt 系统提示一致）。

    Teacher 输出的 risk_level 仅作完整性标记；任何不一致的情况，
    都以 score-derived 为准——这是 PoC 期定下的设计：等级 = 风险严重度，
    不是按伤亡人数事后分级。
    """
    derived = _level_from_score(score)
    if not raw:
        return derived
    s = raw.strip()
    # 兼容旧 enum 与变体写法
    aliases = {
        "low": "一般", "medium": "较大", "high": "重大", "critical": "特别重大",
        "一般事故": "一般", "较大事故": "较大", "重大事故": "重大",
        "特别重大事故": "特别重大", "特别严重": "特别重大",
    }
    canonical = aliases.get(s) or aliases.get(s.lower()) or s
    # 注意：即便 Teacher 写了 canonical 等级名，也必须服从 score 决定的 derived
    return derived if derived else canonical


def _normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if s in _EVENT_CATEGORIES:
        return s
    # 兼容含括号、空格等噪声
    for cat in _EVENT_CATEGORIES:
        if cat in s:
            return cat
    return None


def parse(text: str) -> Optional[Dict]:
    """解析 Teacher 输出。失败返回 None。"""
    reasoning = _extract(text, "reasoning")
    score_raw = _extract(text, "score")
    level_raw = _extract(text, "risk_level")
    conf_raw = _extract(text, "confidence")
    cat_raw = _extract(text, "event_category")

    if reasoning is None or score_raw is None:
        return None

    # score
    try:
        score = float(re.findall(r"-?\d+\.?\d*", score_raw)[0])
    except (IndexError, ValueError):
        return None
    score = max(0.0, min(100.0, score))

    # risk_level
    risk_level = _normalize_level(level_raw, score)

    # event_category
    event_category = _normalize_category(cat_raw)
    # 注意：event_category 缺失不算解析失败，仅留空——由质量过滤阶段决定是否丢弃

    # confidence
    confidence = 0.7
    if conf_raw:
        try:
            confidence = float(re.findall(r"-?\d+\.?\d*", conf_raw)[0])
            confidence = max(0.0, min(1.0, confidence))
        except (IndexError, ValueError):
            pass

    return {
        "event_category": event_category,
        "reasoning": reasoning,
        "score": score,
        "risk_level": risk_level,
        "confidence": confidence,
    }
