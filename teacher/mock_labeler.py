"""Mock Teacher：基于规则的可解释打分器。

设计目的：
1. 没有 DeepSeek API key 时让整个 Pipeline 端到端跑通
2. 输出格式与真实 Teacher 完全一致
   （event_category + reasoning + score + risk_level + confidence）
3. 规则基于救援领域常识，足以让 Student 学到合理的特征排序

⚠️ Mock 仅用于链路验证，正式数据集必须用真实 Teacher 重新标注。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from teacher.recommendations import build_recommendations, format_for_reasoning


# 5 灾种基线风险（0~30）
DISASTER_BASELINE = {
    "earthquake": 25,
    "flood": 18,
    "urban_fire": 22,
    "forest_fire": 16,   # 森林火灾通常人员密度低、可疏散
    "landslide": 20,
}

# 灾种 → 突发事件大类（与 prompt_templates 中的 4 大类一致）
DISASTER_TO_CATEGORY = {
    "earthquake": "自然灾害",
    "flood": "自然灾害",
    "forest_fire": "自然灾害",
    "landslide": "自然灾害",
    "urban_fire": "事故灾难",
}


def _level_from_score(score: float) -> str:
    if score < 25:
        return "一般"
    if score < 50:
        return "较大"
    if score < 75:
        return "重大"
    return "特别重大"


def _survival_rate_from_hours(hours: float, temp_c: float) -> tuple[str, float]:
    """根据黄金 72h 非线性衰减经验曲线返回 (区间描述, 中位生还率)。

    与 prompt_templates 系统提示中的曲线对齐。
    """
    # 极端温度让曲线整体提前 ~9h
    eff_h = hours
    if temp_c < 0 or temp_c > 35:
        eff_h += 9
    if eff_h <= 24:
        return ("≥85%（黄金救援早期）", 0.90)
    if eff_h <= 48:
        return ("50~70%（生命支持开始衰竭）", 0.60)
    if eff_h <= 72:
        return ("20~40%（不可逆损伤累积）", 0.30)
    return ("<10%（已超黄金窗口，生还概率急剧下降）", 0.05)


def _score_components(s: Dict) -> List[tuple]:
    """返回 (因子名, 贡献分, 解释) 三元组列表。"""
    out: List[tuple] = []

    base = DISASTER_BASELINE.get(s["disaster_type"], 20)
    out.append(("灾种基线", base, f"{s['disaster_type']} 基线风险约 {base}"))

    mag = float(s["magnitude"])
    if s["disaster_type"] == "earthquake":
        mag_score = max(0, (mag - 3.0) / 5.5) * 25
    else:
        mag_score = max(0, (mag - 3.0) / 5.5) * 18
    out.append(("强度", mag_score, f"magnitude={mag} 贡献 {mag_score:.1f}"))

    cr = float(s["building_collapse_rate"])
    cr_score = cr * 20
    out.append(("建筑倒塌率", cr_score, f"倒塌率 {cr:.2f} 贡献 {cr_score:.1f}"))

    trapped = int(s["estimated_trapped"])
    trapped_score = min(12, math.log1p(trapped) * 2.2)
    out.append(("被困人数", trapped_score, f"被困 {trapped} 人 贡献 {trapped_score:.1f}"))

    t = float(s["temperature_c"])
    temp_score = min(10, abs(t - 20) * 0.4)
    out.append(("环境温度", temp_score, f"{t}℃ 偏离舒适区 贡献 {temp_score:.1f}"))

    hours = float(s["hours_since_disaster"])
    eta = float(s["rescue_eta_hours"])
    delay = max(0, hours - 24) * 0.15 + eta * 0.6
    delay_score = min(20, delay)
    out.append(("时间窗", delay_score, f"灾后 {hours}h, 救援 ETA {eta}h 贡献 {delay_score:.1f}"))

    road = float(s["road_accessibility"])
    road_score = (1 - road) * 12
    out.append(("道路可达性", road_score, f"可达性 {road:.2f} 贡献 {road_score:.1f}"))

    return out


def _confidence(s: Dict, components: List[tuple]) -> float:
    total = sum(c[1] for c in components)
    if total <= 0:
        return 0.5
    leads = max(c[1] for c in components) / total
    base = 0.65 + min(0.25, leads * 0.4)
    if s["building_collapse_rate"] == 0 and s["estimated_trapped"] == 0:
        base -= 0.15
    return round(max(0.3, min(0.98, base)), 3)


def _format_reasoning(s: Dict, components: List[tuple], score: float) -> str:
    hours = float(s["hours_since_disaster"])
    eta = float(s["rescue_eta_hours"])
    temp = float(s["temperature_c"])
    total_exposure = hours + eta
    surv_band, _ = _survival_rate_from_hours(total_exposure, temp)

    sections = []

    # 第一段：关键风险因子识别
    sections.append("【关键风险因子识别】")
    for name, val, expl in components:
        sections.append(f"  - {name}：{expl}")

    # 第二段：黄金 72h 衰减分析（必答项）
    extreme_note = ""
    if temp < 0 or temp > 35:
        extreme_note = f"环境温度 {temp}℃ 处于极端段，曲线整体提前约 9h，"
    sections.append("【黄金 72h 衰减分析】")
    sections.append(
        f"  当前已过 {hours}h、救援预计还需 {eta}h，总暴露时长约 {total_exposure:.1f}h。"
        f"{extreme_note}对照非线性衰减曲线，被困人员生还率粗估为 {surv_band}。"
    )

    # 第三段：叠加与评分依据（不再有"建议优先调度…"那句固定 boilerplate）
    sections.append("【多因子叠加与评分依据】")
    sections.append(
        f"  以上因子线性叠加得到风险评分 {score:.1f}，归类为 {_level_from_score(score)}；"
        "下一段按等级 + 主导因子 + 极端阈值给出针对性救援建议。"
    )

    # 第四段：针对性救援建议（按 level + 灾种 + 阈值动态生成，1~4 条）
    recs = build_recommendations(s, _level_from_score(score), max_items=4)
    sections.append("【针对性救援建议】")
    sections.append(format_for_reasoning(recs))

    return "\n".join(sections)


def label(sample: Dict, *, jitter: float = 0.0, seed: int | None = None) -> Dict:
    """对单条样本打标。

    Args:
        sample: 原始特征 dict
        jitter: 给 score 加入 ±jitter 的随机扰动（模拟 Teacher 多次采样的不确定性）
        seed: 复现性
    """
    components = _score_components(sample)
    score = sum(c[1] for c in components)
    score = max(0.0, min(100.0, score))

    if jitter > 0:
        rng = random.Random(seed)
        score = max(0.0, min(100.0, score + rng.uniform(-jitter, jitter)))

    return {
        "sample_id": sample.get("sample_id"),
        "input": sample,
        "event_category": DISASTER_TO_CATEGORY.get(sample["disaster_type"], "自然灾害"),
        "reasoning": _format_reasoning(sample, components, score),
        "score": round(score, 2),
        "risk_level": _level_from_score(score),
        "confidence": _confidence(sample, components),
        "teacher": "mock-rule-v2",
        "prompt_version": "mock",
    }
