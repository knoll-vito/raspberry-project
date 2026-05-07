"""Teacher Prompt 模板（4 个版本）。

设计要点（与方案 §5.2 一致）：
- 强制结构化输出（XML 标签包裹），方便后续解析
- 要求 reasoning 不少于 100 字符
- 要求 confidence ∈ [0,1]
- score ∈ [0,100]，risk_level ∈ {low, medium, high, critical}

调用方在 deepseek_client / mock_labeler 中选用模板。
"""

from __future__ import annotations

from typing import Dict

SYSTEM_PROMPT = """你是一名资深应急管理与灾害评估专家，长期参与地震、洪水、火灾、泥石流等灾后人员风险评估。
你的任务是基于结构化灾情特征，输出客观、可解释的人员伤亡风险评分。
你必须严格按指定的 XML 标签格式输出，绝不输出标签之外的解释性内容。
"""


_OUTPUT_FORMAT = """请严格按以下格式输出，不要输出任何额外内容：

<reasoning>
（在此分步阐述你的推理过程：先识别关键风险因子，再分析各因子叠加效应，最后给出评分依据。不少于 100 字。）
</reasoning>
<score>0~100 的浮点数，越高代表伤亡风险越大</score>
<risk_level>low | medium | high | critical 之一</risk_level>
<confidence>0~1 的浮点数，代表你对本次评分的确信度</confidence>
"""


def _format_features(sample: Dict) -> str:
    return (
        f"- 灾种 (disaster_type): {sample['disaster_type']}\n"
        f"- 强度/震级 (magnitude): {sample['magnitude']}\n"
        f"- 建筑倒塌率 (building_collapse_rate): {sample['building_collapse_rate']}\n"
        f"- 估计被困人数 (estimated_trapped): {sample['estimated_trapped']}\n"
        f"- 环境温度 (temperature_c): {sample['temperature_c']} ℃\n"
        f"- 灾后已过小时数 (hours_since_disaster): {sample['hours_since_disaster']} h\n"
        f"- 救援预计抵达时间 (rescue_eta_hours): {sample['rescue_eta_hours']} h\n"
        f"- 道路可通行性 (road_accessibility): {sample['road_accessibility']}\n"
    )


# v1：基础版（适用于 60% 普通样本）
def prompt_v1(sample: Dict) -> str:
    return (
        "请基于以下灾情特征，评估人员伤亡风险：\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v2：强制推理链（适用于复杂多因素样本）
def prompt_v2(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。本样本涉及多个高强度风险因子，"
        "你必须分步推理，列出至少 3 条关键判断，并解释它们如何相互放大或抵消。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v3：对比推理（适用于风险等级边界模糊样本）
def prompt_v3(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。本样本可能位于风险等级边界上，"
        "请同时讨论『为什么是 high 而不是 medium』或『为什么是 critical 而不是 high』，"
        "再给出最终评分。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v4：不确定性感知（适用于数据缺失或异常样本）
def prompt_v4(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。注意：部分特征可能不完整或处于异常区间，"
        "请显式说明你对哪些信息不确定，并将这种不确定性反映在 confidence 字段中。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


PROMPTS = {
    "v1": prompt_v1,
    "v2": prompt_v2,
    "v3": prompt_v3,
    "v4": prompt_v4,
}


def build(sample: Dict, version: str = "v1") -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version: {version}; expected one of {list(PROMPTS)}")
    return PROMPTS[version](sample)
