"""Teacher Prompt 模板（4 个版本）。

设计要点（与方案 §5.2 + 用户领域要求一致）：
- 评估框架基于《国家突发事件总体应急预案》：
  · 4 大类：自然灾害 / 事故灾难 / 公共卫生事件 / 社会安全事件
  · 4 等级：一般 / 较大 / 重大 / 特别重大（按伤亡人员或经济损失划分）
- 强制结构化输出（XML 标签包裹），方便后续解析
- 必答项：黄金 72h 衰减分析（reasoning 内必须显式分析 hours_since_disaster
  对生还率的非线性影响）
- score ∈ [0,100]，confidence ∈ [0,1]，reasoning ≥ 100 字符
- risk_level ∈ {一般, 较大, 重大, 特别重大}

调用方在 deepseek_client / mock_labeler 中选用模板。
"""

from __future__ import annotations

from typing import Dict


SYSTEM_PROMPT = """你是一名熟悉《国家突发事件总体应急预案》的应急管理与灾后人员风险评估专家。
本任务输出的是【人员伤亡风险严重度（risk severity）】，用于救援资源调度的预警决策，
而不是事后官方分级——所以等级名虽沿用四级标准的词汇，但映射规则按风险严重度，详见【二】。
你的评估框架严格遵循以下标准：

【一、突发事件四大类】
1. 自然灾害：地震、洪水、滑坡、泥石流、台风、森林火灾等
2. 事故灾难：城市火灾、交通事故、化工事故、矿难等
3. 公共卫生事件：传染病疫情、群体不明原因疾病等
4. 社会安全事件：恐怖袭击、群体性事件、刑事案件等

【二、风险等级（risk_level）映射规则——必须严格遵守！】
本任务的 risk_level 由 score 决定，而非按死亡人数推断：
- score <  25  →  一般
- 25 ≤ score < 50  →  较大
- 50 ≤ score < 75  →  重大
- score ≥ 75  →  特别重大

⚠️ 这与官方"按伤亡人数事后分级"不同。即便被困人数较少（例如 2 人），
   只要场景具备高致死风险（极端温度、超过 72h、道路阻断等），
   仍可对应"特别重大"。reasoning 中可以提及死亡人数估算辅助说理，
   但等级 enum 必须按上述 score 阈值反查。

四级官方标准（仅作为评分锚点参考，不参与 enum 决策）：
- 一般：典型场景死亡 < 3、经济损失 < 1000 万
- 较大：典型场景死亡 3~10、经济损失 1000~5000 万
- 重大：典型场景死亡 10~30、经济损失 5000 万~1 亿
- 特别重大：典型场景死亡 ≥ 30、经济损失 ≥ 1 亿

【三、黄金 72 小时生还率非线性衰减经验曲线（必须代入推理）】
- 0~24h：生还率 ≈ 90%（黄金救援早期）
- 24~48h：生还率 ≈ 50~70%（生命支持开始衰竭）
- 48~72h：生还率 ≈ 20~40%（不可逆损伤累积）
- 72h 之后：生还率急剧下降至 < 10%（除特殊条件如水源、气流）
- 极端温度（< 0℃ 或 > 35℃）会让上述曲线整体提前 6~12h

你必须严格按指定 XML 标签格式输出，绝不输出标签之外的解释性内容。
"""


_OUTPUT_FORMAT = """请严格按以下格式输出，不要输出任何额外内容：

<event_category>
（在四大类中选一个：自然灾害 | 事故灾难 | 公共卫生事件 | 社会安全事件）
</event_category>
<reasoning>
（分步推理，至少包含三段，且第二段必须是 72h 衰减分析：
 1) 关键风险因子识别（灾种、强度、被困人数、温度、道路等）；
 2) 黄金 72 小时衰减分析（基于 hours_since_disaster + rescue_eta_hours 的总暴露时长，
    对照系统提示中的非线性曲线，给出当前生还率粗估区间，并说明极端温度对曲线的偏移影响）；
 3) 多因子叠加效应与最终评分依据。不少于 100 字。）
</reasoning>
<score>0~100 的浮点数，越高代表伤亡风险越大</score>
<risk_level>一般 | 较大 | 重大 | 特别重大 之一（必须严格按 score 阈值 25/50/75 反查，不要按死亡人数推断）</risk_level>
<confidence>0~1 的浮点数，代表你对本次评分的确信度</confidence>
"""


def _v(sample: Dict, key: str, unit: str = "") -> str:
    """缺失值统一显示为「未知/缺失」，让 V4 自己处理不确定性。"""
    if key not in sample or sample[key] is None or sample[key] == "":
        return "未知/缺失"
    return f"{sample[key]}{unit}"


def _format_features(sample: Dict) -> str:
    extra = ""
    if sample.get("event_name"):
        extra += f"- 事件标识 (event_name): {sample['event_name']}\n"
    return (
        extra
        + f"- 灾种 (disaster_type): {_v(sample, 'disaster_type')}\n"
        + f"- 强度/震级 (magnitude): {_v(sample, 'magnitude')}\n"
        + f"- 建筑倒塌率 (building_collapse_rate): {_v(sample, 'building_collapse_rate')}\n"
        + f"- 估计被困人数 (estimated_trapped): {_v(sample, 'estimated_trapped')}\n"
        + f"- 环境温度 (temperature_c): {_v(sample, 'temperature_c', ' ℃')}\n"
        + f"- 灾后已过小时数 (hours_since_disaster): {_v(sample, 'hours_since_disaster', ' h')}\n"
        + f"- 救援预计抵达时间 (rescue_eta_hours): {_v(sample, 'rescue_eta_hours', ' h')}\n"
        + f"- 道路可通行性 (road_accessibility): {_v(sample, 'road_accessibility')}\n"
    )


# v1：基础版（适用于 60% 普通样本）
def prompt_v1(sample: Dict) -> str:
    return (
        "请基于以下灾情特征，按系统提示中的四大类/四等级标准评估人员伤亡风险，"
        "并务必在 <reasoning> 第二段内显式分析黄金 72h 衰减曲线。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v2：强制推理链（适用于复杂多因素样本）
def prompt_v2(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。本样本涉及多个高强度风险因子，"
        "你必须在 <reasoning> 中分步推理，列出至少 3 条关键判断，并解释它们如何相互放大或抵消。"
        "其中黄金 72h 衰减分析为必答项，需明确代入非线性曲线给出当前生还率区间。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v3：对比推理（适用于风险等级边界模糊样本）
def prompt_v3(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。本样本可能位于风险等级边界上，"
        "请同时讨论『为什么是 重大 而不是 较大』或『为什么是 特别重大 而不是 重大』，"
        "再给出最终评分。同样必须在 <reasoning> 中包含黄金 72h 衰减分析段。\n\n"
        f"{_format_features(sample)}\n"
        f"{_OUTPUT_FORMAT}"
    )


# v4：不确定性感知（适用于数据缺失或异常样本）
def prompt_v4(sample: Dict) -> str:
    return (
        "请基于以下灾情特征评估人员伤亡风险。注意：部分特征可能不完整或处于异常区间，"
        "请显式说明你对哪些信息不确定，并将这种不确定性反映在 confidence 字段中。"
        "黄金 72h 衰减分析为必答项。\n\n"
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
