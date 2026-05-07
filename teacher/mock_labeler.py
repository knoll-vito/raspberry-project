"""Mock Teacher：基于规则的可解释打分器。

设计目的：
1. 在没有 DeepSeek API key 的情况下让整个 Pipeline 端到端跑通
2. 输出格式与真实 Teacher 完全一致（reasoning + score + risk_level + confidence）
3. 规则基于救援领域常识，足以支撑 Student 模型学到合理的特征排序

⚠️ Mock 仅用于链路验证，正式数据集必须用真实 Teacher 重新标注。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List


# 灾种基线风险（0~30）
DISASTER_BASELINE = {
    "earthquake": 25,
    "flood": 18,
    "fire": 22,
    "landslide": 20,
}


def _level_from_score(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "medium"
    if score < 75:
        return "high"
    return "critical"


def _score_components(s: Dict) -> List[tuple]:
    """返回 (因子名, 贡献分, 解释) 三元组列表。"""
    out: List[tuple] = []

    base = DISASTER_BASELINE.get(s["disaster_type"], 20)
    out.append(("灾种基线", base, f"{s['disaster_type']} 基线风险约 {base}"))

    # 强度：3.0~8.5 → 0~25 分（地震尤其敏感）
    mag = float(s["magnitude"])
    if s["disaster_type"] == "earthquake":
        mag_score = max(0, (mag - 3.0) / 5.5) * 25
    else:
        mag_score = max(0, (mag - 3.0) / 5.5) * 18
    out.append(("强度", mag_score, f"magnitude={mag} 贡献 {mag_score:.1f}"))

    # 倒塌率：线性 0~20
    cr = float(s["building_collapse_rate"])
    cr_score = cr * 20
    out.append(("建筑倒塌率", cr_score, f"倒塌率 {cr:.2f} 贡献 {cr_score:.1f}"))

    # 被困人数：log scale 0~12
    trapped = int(s["estimated_trapped"])
    trapped_score = min(12, math.log1p(trapped) * 2.2)
    out.append(("被困人数", trapped_score, f"被困 {trapped} 人 贡献 {trapped_score:.1f}"))

    # 极端温度：偏离 20℃ 越远越高，0~10
    t = float(s["temperature_c"])
    temp_score = min(10, abs(t - 20) * 0.4)
    out.append(("环境温度", temp_score, f"{t}℃ 偏离舒适区 贡献 {temp_score:.1f}"))

    # 黄金 72h：超过越久风险递增；同时 rescue_eta 越长越糟
    hours = float(s["hours_since_disaster"])
    eta = float(s["rescue_eta_hours"])
    delay = max(0, hours - 24) * 0.15 + eta * 0.6  # 0~~20
    delay_score = min(20, delay)
    out.append(("时间窗", delay_score, f"灾后 {hours}h, 救援 ETA {eta}h 贡献 {delay_score:.1f}"))

    # 道路可达性：越差扣分越多（这里"扣分"取负）
    road = float(s["road_accessibility"])
    road_score = (1 - road) * 12  # 0~12
    out.append(("道路可达性", road_score, f"可达性 {road:.2f} 贡献 {road_score:.1f}"))

    return out


def _confidence(s: Dict, components: List[tuple]) -> float:
    """规则版置信度：分散贡献越均衡越自信；存在极端值时也更自信。"""
    total = sum(c[1] for c in components)
    if total <= 0:
        return 0.5
    # 主导因子占比越高，越确信结论；但太低也降信心
    leads = max(c[1] for c in components) / total
    base = 0.65 + min(0.25, leads * 0.4)
    # 缺失/异常时降低
    if s["building_collapse_rate"] == 0 and s["estimated_trapped"] == 0:
        base -= 0.15
    return round(max(0.3, min(0.98, base)), 3)


def _format_reasoning(s: Dict, components: List[tuple], score: float) -> str:
    lines = [
        f"对样本 {s.get('sample_id','?')} 做分步推理：",
    ]
    for name, val, expl in components:
        lines.append(f"- {name}：{expl}")
    lines.append(f"以上因子线性叠加得到风险评分 {score:.1f}，归类为 {_level_from_score(score)}。")
    lines.append("结论：在该灾情条件下，应优先调度搜救与医疗资源，特别关注主导风险因子。")
    return "\n".join(lines)


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
        "reasoning": _format_reasoning(sample, components, score),
        "score": round(score, 2),
        "risk_level": _level_from_score(score),
        "confidence": _confidence(sample, components),
        "teacher": "mock-rule-v1",
        "prompt_version": "mock",
    }
