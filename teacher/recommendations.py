"""灾后伤亡风险 · 救援建议规则引擎（共享给 mock_labeler 和 Student inference）。

设计目标：避免每个样本输出相同的 boilerplate，例如旧版 mock_labeler 末尾那句
"建议优先调度搜救与医疗资源，重点关注主导风险因子" —— 任何输入都返回这一句。

工作原理：按 (风险等级 + 灾种 + 极端阈值组合) 多条件触发，每条建议带：
  - priority: critical / high / medium / low（指示调度紧急程度）
  - tag:      触发器标识，便于下游程序识别（不靠文本匹配）
  - text:     人类可读建议（中文）

所有规则都是确定性的（不带随机性），所以 Mock 与 Student 共享同一引擎，
保证两者输出风格一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ── 优先级常量 ────────────────────────────────────────────────────────────
PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

_PRIORITY_ORDER = {
    PRIORITY_CRITICAL: 0,
    PRIORITY_HIGH: 1,
    PRIORITY_MEDIUM: 2,
    PRIORITY_LOW: 3,
}


# ── 等级基线建议（每条样本必触发一条） ──────────────────────────────────
_LEVEL_BASELINE: Dict[str, Dict] = {
    "特别重大": {
        "priority": PRIORITY_CRITICAL,
        "tag": "level_extreme",
        "text": "极端伤亡风险，立即启动一级响应；调集省级跨区救援力量 + 武警/部队介入；省级指挥部前置。",
    },
    "重大": {
        "priority": PRIORITY_HIGH,
        "tag": "level_major",
        "text": "重大伤亡风险，启动二级响应；市级救援力量优先到位 + 建立现场指挥所 + 24h 内伤员转运通道。",
    },
    "较大": {
        "priority": PRIORITY_MEDIUM,
        "tag": "level_large",
        "text": "区域性伤亡风险，县级应急 + 主管部门联动；专业救援队 12h 内到场。",
    },
    "一般": {
        "priority": PRIORITY_LOW,
        "tag": "level_general",
        "text": "风险可控，常规调度 + 维持现场秩序 + 定时信息通报。",
    },
}


# ── 灾种专项建议 ─────────────────────────────────────────────────────────
_DISASTER_SPECIFIC: Dict[str, Dict] = {
    "earthquake": {
        "priority": PRIORITY_MEDIUM,
        "tag": "earthquake_specific",
        "text": "地震专项：警惕余震 + 危房标识 + 燃气电力切断 + 防止次生火灾。",
    },
    "flood": {
        "priority": PRIORITY_MEDIUM,
        "tag": "flood_specific",
        "text": "洪涝专项：防溺水 + 水土传染病预防 + 电力设施漏电隐患排查 + 防汛物资前置。",
    },
    "forest_fire": {
        "priority": PRIORITY_MEDIUM,
        "tag": "forest_fire_specific",
        "text": "森林火灾专项：下风向疏散 + 防火带开辟 + 直升机洒水 + 防烟吸入伤。",
    },
    "landslide": {
        "priority": PRIORITY_MEDIUM,
        "tag": "landslide_specific",
        "text": "滑坡专项：警惕二次滑动 + 上游水文监测 + 堰塞湖排查 + 撤离下游居民。",
    },
    "urban_fire": {
        "priority": PRIORITY_MEDIUM,
        "tag": "urban_fire_specific",
        "text": "城市火灾/爆炸专项：化学品评估 + 周边人员疏散 + 烟雾医学监测 + 呼吸保护。",
    },
}


# ── 主引擎 ───────────────────────────────────────────────────────────────


def build_recommendations(sample: Dict, level_zh: str, *,
                          max_items: int = 4) -> List[Dict]:
    """根据样本特征 + 风险等级生成针对性救援建议清单。

    Args:
        sample:    含 disaster_type / temperature_c / road_accessibility /
                   building_collapse_rate / estimated_trapped /
                   hours_since_disaster / rescue_eta_hours 的字典
        level_zh:  '特别重大' / '重大' / '较大' / '一般'
        max_items: 输出条数上限（默认 4，避免噪音过多）

    Returns:
        [{priority, tag, text}, ...] —— 按优先级（critical → low）排序，
        触发器去重（同 tag 只出现一次）。
    """
    recs: List[Dict] = []

    # 1. 等级基线（必触发）
    baseline = _LEVEL_BASELINE.get(level_zh, _LEVEL_BASELINE["较大"])
    recs.append(dict(baseline))

    # 2. 道路阻断
    road = sample.get("road_accessibility")
    if isinstance(road, (int, float)):
        if road < 0.2:
            recs.append({
                "priority": PRIORITY_CRITICAL,
                "tag": "road_isolation",
                "text": f"道路严重阻断（可达性 {road:.2f}），建议直升机/无人机空中投送 + 工程兵紧急破障。",
            })
        elif road < 0.5:
            recs.append({
                "priority": PRIORITY_HIGH,
                "tag": "road_partial",
                "text": f"道路部分阻断（可达性 {road:.2f}），规划备用进出路线 + 重型清障车辆先行。",
            })

    # 3. 极端温度
    temp = sample.get("temperature_c")
    if isinstance(temp, (int, float)):
        if temp < 0:
            recs.append({
                "priority": PRIORITY_HIGH,
                "tag": "extreme_cold",
                "text": f"低温环境（{temp:g}°C），失温窗口快速收紧，前置保暖物资（保温毯/睡袋）+ 加快开挖。",
            })
        elif temp < 5:
            recs.append({
                "priority": PRIORITY_MEDIUM,
                "tag": "cold_weather",
                "text": f"寒冷环境（{temp:g}°C），注意被困者保暖 + 救援人员防冻装备到位。",
            })
        elif temp > 35:
            recs.append({
                "priority": PRIORITY_HIGH,
                "tag": "extreme_heat",
                "text": f"极端高温（{temp:g}°C），中暑/脱水风险，前置医疗补水 + 遮阳轮换 + 救援人员轮岗 30 分钟。",
            })

    # 4. 时间窗（黄金 72h）
    hours = sample.get("hours_since_disaster")
    if isinstance(hours, (int, float)):
        if hours > 72:
            recs.append({
                "priority": PRIORITY_HIGH,
                "tag": "golden_window_expired",
                "text": f"已过黄金 72h（已 {hours:g}h），搜救转遗体处理为主 + 心理援助 + 防疫消杀。",
            })
        elif hours > 48:
            recs.append({
                "priority": PRIORITY_MEDIUM,
                "tag": "golden_window_late",
                "text": f"黄金窗口收尾期（已 {hours:g}h），未来 24h 是最后高产搜救窗，加强生命探测仪 + 多班轮替。",
            })
        elif hours <= 6:
            recs.append({
                "priority": PRIORITY_HIGH,
                "tag": "golden_window_early",
                "text": f"早期黄金窗口（仅 {hours:g}h），快速集结主力搜救 + 优先抢通生命通道。",
            })

    # 5. 高倒塌率（结构性救援）
    cr = sample.get("building_collapse_rate")
    if isinstance(cr, (int, float)) and cr >= 0.7:
        recs.append({
            "priority": PRIORITY_HIGH,
            "tag": "high_collapse",
            "text": f"高倒塌率（{cr:.2f}），结构性救援为主，调集生命探测仪 + 工程吊机 + 切割破拆器械。",
        })
    elif isinstance(cr, (int, float)) and cr >= 0.3:
        recs.append({
            "priority": PRIORITY_MEDIUM,
            "tag": "moderate_collapse",
            "text": f"中等倒塌率（{cr:.2f}），分区分段排查 + 关注次生坍塌风险。",
        })

    # 6. 被困规模
    trapped = sample.get("estimated_trapped")
    if isinstance(trapped, (int, float)) and trapped >= 1000:
        recs.append({
            "priority": PRIORITY_HIGH,
            "tag": "mass_casualty",
            "text": f"大规模被困（{int(trapped)} 人），多点同时作业 + 三级伤员分流 + 野战医院前置。",
        })
    elif isinstance(trapped, (int, float)) and trapped >= 100:
        recs.append({
            "priority": PRIORITY_MEDIUM,
            "tag": "moderate_casualty",
            "text": f"显著伤亡规模（{int(trapped)} 人），医疗力量配套（救护车 ≥10 辆 + 流动血库）。",
        })

    # 7. 救援 ETA 长
    eta = sample.get("rescue_eta_hours")
    if isinstance(eta, (int, float)) and eta > 12:
        recs.append({
            "priority": PRIORITY_MEDIUM,
            "tag": "long_eta",
            "text": f"救援 ETA 长（{eta:g}h），考虑空中投送维生包先行 + 卫星通信前置 + 卫生装备空运。",
        })

    # 8. 灾种专项
    dt = sample.get("disaster_type")
    if dt in _DISASTER_SPECIFIC:
        recs.append(dict(_DISASTER_SPECIFIC[dt]))

    # 排序 + 去重 + 截断
    recs.sort(key=lambda r: _PRIORITY_ORDER.get(r["priority"], 99))
    seen_tags = set()
    deduped: List[Dict] = []
    for r in recs:
        if r["tag"] in seen_tags:
            continue
        seen_tags.add(r["tag"])
        deduped.append(r)
        if len(deduped) >= max_items:
            break

    return deduped


def format_for_reasoning(recs: List[Dict]) -> str:
    """把 recs 格式化成多行 reasoning 文本（供 mock_labeler 第三段使用）。"""
    if not recs:
        return "  无特殊建议。"
    lines = []
    for r in recs:
        lines.append(f"  [{r['priority'].upper()}] {r['text']}")
    return "\n".join(lines)
