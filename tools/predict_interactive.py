"""交互式推理：逐字段输入 → 显示 score/risk_level + mock baseline 对比。

用法：
    python3 tools/predict_interactive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DISASTER_TYPES = ["earthquake", "flood", "urban_fire", "forest_fire", "landslide"]


def _ask(prompt: str, *, choices: list = None, lo: float = None, hi: float = None,
         default: str = None, is_int: bool = False) -> object:
    while True:
        hint = ""
        if choices:
            hint = f" [{'/'.join(choices)}]"
        elif lo is not None and hi is not None:
            hint = f" [{lo}~{hi}]"
        if default is not None:
            hint += f" (默认 {default})"
        raw = input(f"  {prompt}{hint}: ").strip()
        if not raw and default is not None:
            raw = default
        if choices:
            if raw in choices:
                return raw
            print(f"  ⚠️  必须是 {choices} 之一，重输")
            continue
        try:
            v = int(raw) if is_int else float(raw)
        except ValueError:
            print(f"  ⚠️  '{raw}' 不是合法数字，重输")
            continue
        if lo is not None and v < lo:
            print(f"  ⚠️  {v} < {lo}，重输")
            continue
        if hi is not None and v > hi:
            print(f"  ⚠️  {v} > {hi}，重输")
            continue
        return v


def main() -> int:
    from deploy.inference import RiskScorer
    from teacher import mock_labeler

    print("=" * 60)
    print("  灾后伤亡风险 · 本地推理（LightGBM Student）")
    print("=" * 60)
    print("  按 Ctrl+C 随时退出；带「默认 X」的字段直接回车即用默认\n")

    sample = {
        "disaster_type": _ask("灾种", choices=DISASTER_TYPES, default="earthquake"),
        "magnitude": _ask("强度（地震=里氏；其他按映射）", lo=3.0, hi=8.5, default="6.5"),
        "building_collapse_rate": _ask("建筑倒塌率", lo=0.0, hi=1.0, default="0.4"),
        "estimated_trapped": _ask("估计被困人数", lo=0, hi=10000, default="50", is_int=True),
        "temperature_c": _ask("环境温度 ℃", lo=-40, hi=50, default="15"),
        "hours_since_disaster": _ask("灾后已过小时数", lo=0, hi=720, default="6"),
        "rescue_eta_hours": _ask("救援预计 ETA 小时", lo=0, hi=168, default="2"),
        "road_accessibility": _ask("道路可通行性", lo=0.0, hi=1.0, default="0.5"),
        "medical_accessibility": _ask("医疗资源可及性", lo=0.0, hi=1.0, default="0.5"),
        "rescue_skill_level": _ask("救援队伍技能水平", lo=0.0, hi=1.0, default="0.5"),
        "night_time": _ask("是否夜间事件 (0/1)", lo=0, hi=1, default="0", is_int=True),
        "holiday_event": _ask("是否节假日/大型活动 (0/1)", lo=0, hi=1, default="0", is_int=True),
    }

    scorer = RiskScorer()
    student = scorer.predict_one(sample)
    mock = mock_labeler.label(sample)

    print()
    print("=" * 60)
    print(f"  Student (LightGBM 232KB):  score={student['score']:6.2f}  level={student['risk_level']}")
    print(f"  Mock baseline (规则):       score={mock['score']:6.2f}  level={mock['risk_level']}")
    print(f"  diff (Student - Mock):     {student['score'] - mock['score']:+.2f}")
    print("=" * 60)

    print("\n  Mock 给出的推理路径（仅作对照参考）:")
    for line in mock["reasoning"].split("\n"):
        print(f"    {line}")

    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[interactive] bye")
        sys.exit(0)
