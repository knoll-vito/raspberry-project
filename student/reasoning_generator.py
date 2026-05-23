"""Phase 3：Student 动态 Reasoning 生成器。

基于 SHAP top features + score + sample 原始值，动态生成【多因子叠加与评分依据】段落，
替代原来任何输入都雷同的 boilerplate。

生成的 reasoning 分 4 段（与 mock_labeler / prompt_templates 格式对齐）：
  1) 关键风险因子识别
  2) 黄金 72h 衰减分析
  3) 多因子叠加与评分依据（动态生成，核心改进）
  4) 针对性救援建议（委托 recommendations.py）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from teacher.recommendations import build_recommendations, format_for_reasoning

# display name (from FEATURE_DISPLAY_NAME in inference.py) → sample dict key
_DISPLAY_TO_INTERNAL: dict = {
    "collapse_rate": "building_collapse_rate",
    "trapped_count": "estimated_trapped",
    "temp_c": "temperature_c",
    "hours_since_event": "hours_since_disaster",
    "rescue_eta": "rescue_eta_hours",
    "road_access": "road_accessibility",
}

# ── 72h 生还率曲线（与 mock_labeler 对齐）─────────────────────────────────


def _survival_band(hours: float, temp_c: float) -> Tuple[str, float]:
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


def _score_to_level(score: float) -> str:
    if score < 25:
        return "一般"
    if score < 50:
        return "较大"
    if score < 75:
        return "重大"
    return "特别重大"


# ── 核心：动态生成多因子叠加段落 ─────────────────────────────────────────


def _describe_factor_direction(internal_name: str, raw_shap: float,
                                sample: Dict) -> str:
    """将单个因子生成一句有方向感的描述。"""
    val = sample.get(internal_name, 0)
    direction = "↑" if raw_shap >= 0 else "↓"

    if internal_name == "building_collapse_rate":
        level = "极高" if val >= 0.7 else "高" if val >= 0.4 else "低"
        return f"建筑倒塌率 {val:.0%}（{level}）{direction}"

    if internal_name == "estimated_trapped":
        return f"被困人数 {int(val)} 人 {direction}"

    if internal_name == "hours_since_disaster":
        window = "黄金窗口早期" if val <= 6 else "黄金窗口中后段" if val <= 48 else "黄金窗口末期" if val <= 72 else "已超黄金窗口"
        return f"灾后 {val:.0f}h（{window}）{direction}"

    if internal_name == "rescue_eta_hours":
        eta_label = "短" if val <= 2 else "中等" if val <= 8 else "长" if val <= 24 else "极长"
        return f"救援 ETA {val:.1f}h（{eta_label}）{direction}"

    if internal_name == "road_accessibility":
        road_label = "严重阻断" if val < 0.2 else "部分阻断" if val < 0.5 else "可达"
        return f"道路可达性 {val:.2f}（{road_label}）{direction}"

    if internal_name == "temperature_c":
        temp_label = "低温" if val < 5 else "高温" if val > 35 else "常温"
        return f"环境温度 {val:.1f}℃（{temp_label}）{direction}"

    if internal_name == "magnitude":
        return f"强度 mag {val:.1f} {direction}"

    if internal_name == "medical_accessibility":
        label = "极低" if val < 0.2 else "低" if val < 0.5 else "中等" if val < 0.8 else "充足"
        return f"医疗可及性 {val:.2f}（{label}）{direction}"

    if internal_name == "rescue_skill_level":
        label = "低" if val < 0.3 else "中等" if val < 0.7 else "高"
        return f"救援技能水平 {val:.2f}（{label}）{direction}"

    if internal_name == "night_time":
        return f"夜间事件 {'是' if val else '否'} {direction}"

    if internal_name == "holiday_event":
        return f"节假日期间 {'是' if val else '否'} {direction}"

    if internal_name.startswith("is_"):
        dt = internal_name[3:]
        return f"灾种={dt} {'阳性' if val else '阴性'} {direction}"

    return f"{internal_name}={val} {direction}"


def build_multi_factor_text(sample: Dict, score: float,
                            top_features: List[Dict]) -> str:
    """动态生成【多因子叠加与评分依据】段落（核心改进）。"""
    if not top_features:
        return (
            f"  综合以上因子加权融合计算，风险评分 {score:.1f}，"
            f"归类为 {_score_to_level(score)}。"
        )

    # 按 raw_shap 绝对值排序，取前 3~5 个主导因子
    sorted_feats = sorted(top_features, key=lambda f: abs(f.get("raw_shap", 0)), reverse=True)
    leading = sorted_feats[:4]

    lines = []
    lines.append(f"  风险评分 {score:.1f}（{_score_to_level(score)}），")
    lines.append("  主导因子（按贡献度排序）：")

    contributions = []
    for i, f in enumerate(leading, 1):
        display = f.get("feature", "")
        internal = _DISPLAY_TO_INTERNAL.get(display, display)
        raw = f.get("raw_shap", 0)
        desc = _describe_factor_direction(internal, raw, sample)
        contributions.append(f"    第{i}因子：{desc}（贡献 {'+' if raw >= 0 else ''}{raw:.1f} 分）")

    lines.extend(contributions)
    lines.append(f"  以上因子加权叠加，归类为 {_score_to_level(score)}；")
    lines.append("  下一段按等级+主导因子给出针对性救援建议。")

    return "\n".join(lines)


def generate_student_reasoning(sample: Dict, score: float,
                                 top_features: List[Dict],
                                 level: str | None = None) -> str:
    """生成完整的 4 段 reasoning 文本（供 inference.py 使用）。"""
    if level is None:
        level = _score_to_level(score)

    sections = []

    # 第一段：关键风险因子识别
    sections.append("【关键风险因子识别】")
    dtype = sample.get("disaster_type", "未知")
    sections.append(f"  - 灾种：{dtype}（{_disaster_baseline(dtype)} 分基线）")
    sections.append(f"  - 强度：mag={sample.get('magnitude', '?')}（{_magnitude_contrib(sample)} 分）")
    sections.append(f"  - 建筑倒塌率：{sample.get('building_collapse_rate', 0):.2f}（{_collapse_contrib(sample)} 分）")
    sections.append(f"  - 被困人数：{int(sample.get('estimated_trapped', 0))} 人（{_trapped_contrib(sample)} 分）")
    sections.append(f"  - 环境温度：{sample.get('temperature_c', 0):.1f}℃（{_temp_contrib(sample)} 分）")
    sections.append(f"  - 时间窗口：灾后 {sample.get('hours_since_disaster', 0):.1f}h，ETA {sample.get('rescue_eta_hours', 0):.1f}h（{_time_contrib(sample)} 分）")
    sections.append(f"  - 道路可达性：{sample.get('road_accessibility', 0):.2f}（{_road_contrib(sample)} 分）")
    if sample.get("medical_accessibility") is not None:
        sections.append(f"  - 医疗可及性：{sample.get('medical_accessibility', 0):.2f}")
    if sample.get("rescue_skill_level") is not None:
        sections.append(f"  - 救援技能水平：{sample.get('rescue_skill_level', 0):.2f}")

    # 第二段：黄金 72h 衰减分析
    hours = float(sample.get("hours_since_disaster", 0))
    eta = float(sample.get("rescue_eta_hours", 0))
    temp = float(sample.get("temperature_c", 20))
    total_exposure = hours + eta
    surv_band, _ = _survival_band(total_exposure, temp)

    sections.append("【黄金 72h 衰减分析】")
    extreme_note = ""
    if temp < 0 or temp > 35:
        extreme_note = f"环境温度 {temp}℃ 处于极端段，曲线整体提前约 9h，"
    sections.append(
        f"  当前已过 {hours:.1f}h、救援预计还需 {eta:.1f}h，总暴露时长约 {total_exposure:.1f}h。"
        f"{extreme_note}对照非线性衰减曲线，被困人员生还率粗估为 {surv_band}。"
    )

    # 第三段：多因子叠加与评分依据（动态生成）
    sections.append("【多因子叠加与评分依据】")
    sections.append(build_multi_factor_text(sample, score, top_features))

    # 第四段：针对性救援建议
    recs = build_recommendations(sample, level, max_items=4)
    sections.append("【针对性救援建议】")
    sections.append(format_for_reasoning(recs))

    return "\n".join(sections)


# ── 各因子贡献辅助函数 ────────────────────────────────────────────────────


def _disaster_baseline(dt: str) -> float:
    baselines = {
        "earthquake": 25, "flood": 18, "urban_fire": 22,
        "forest_fire": 16, "landslide": 20,
    }
    return baselines.get(dt, 20)


def _magnitude_contrib(s: Dict) -> float:
    import math
    mag = float(s.get("magnitude", 0))
    dt = s.get("disaster_type", "")
    if dt == "earthquake":
        return max(0, (mag - 3.0) / 5.5) * 25
    return max(0, (mag - 3.0) / 5.5) * 18


def _collapse_contrib(s: Dict) -> float:
    cr = float(s.get("building_collapse_rate", 0))
    return cr * 20


def _trapped_contrib(s: Dict) -> float:
    import math
    t = int(s.get("estimated_trapped", 0))
    return min(12, math.log1p(t) * 2.2)


def _temp_contrib(s: Dict) -> float:
    t = float(s.get("temperature_c", 20))
    return min(10, abs(t - 20) * 0.4)


def _time_contrib(s: Dict) -> float:
    h = float(s.get("hours_since_disaster", 0))
    eta = float(s.get("rescue_eta_hours", 0))
    delay = max(0, h - 24) * 0.15 + eta * 0.6
    return min(20, delay)


def _road_contrib(s: Dict) -> float:
    road = float(s.get("road_accessibility", 0.5))
    return (1 - road) * 12
