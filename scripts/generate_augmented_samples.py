"""Phase 1：使用 Latin Hypercube Sampling 生成多样化合成样本。

基于 starter_real.csv 的边缘分布，用 scipy.stats.qmc.LatinHypercube
生成均匀填充的高维特征空间，覆盖夜/昼、节假日/工作日、高低温 等极端组合。

输出：data/augmented_samples.jsonl（300 条，区别于 raw_samples.jsonl）
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 特征空间定义 ──────────────────────────────────────────────────────────
# each entry: (name, low, high, dtype)
# dtype: "cont" = continuous float, "binary" = 0/1

FEATURE_SPACE = [
    # disaster_type 由 one-hot 单独处理
    ("magnitude",               3.0,  9.5,  "cont"),
    ("building_collapse_rate", 0.0,  1.0,  "cont"),
    ("estimated_trapped",      0,    10000, "int"),
    ("temperature_c",         -20.0, 45.0, "cont"),
    ("hours_since_disaster",   0.0,  168.0,"cont"),
    ("rescue_eta_hours",       0.0,  72.0, "cont"),
    ("road_accessibility",     0.0,  1.0,  "cont"),
    ("medical_accessibility",  0.0,  1.0,  "cont"),
    ("rescue_skill_level",      0.0,  1.0,  "cont"),
    ("night_time",             0,    1,    "binary"),
    ("holiday_event",          0,    1,    "binary"),
]

# 固定每个灾种在总样本中的比例（近似 starter_real.csv 的灾种分布）
DISASTER_WEIGHTS = {
    "earthquake":  0.30,
    "flood":       0.25,
    "landslide":   0.20,
    "urban_fire":  0.15,
    "forest_fire": 0.10,
}
DISASTER_TYPES = list(DISASTER_WEIGHTS.keys())

N_SAMPLES = 300
SEED = 42


def _disaster_type_for_index(idx: int, total: int = N_SAMPLES) -> str:
    """按 DISASTER_WEIGHTS 的累积分布分配灾种"""
    r = idx / total
    cum = 0.0
    for dt, w in DISASTER_WEIGHTS.items():
        cum += w
        if r < cum:
            return dt
    return DISASTER_TYPES[-1]


def _build_sample(idx: int, raw: dict, total: int = N_SAMPLES) -> dict:
    """把连续采样的 raw 值转成带 sample_id / disaster_type 的完整样本"""
    sample_id = f"AUG{idx:05d}"
    disaster_type = _disaster_type_for_index(idx, total)

    # magnitude → 地震用里氏，其他用 Venshurst 映射
    mag = raw["magnitude"]
    if disaster_type != "earthquake":
        # Venshurst: 3.0→轻微, 5.5→中等, 7.0→严重, 8.5→灾难
        mag = 3.0 + (raw["magnitude"] / 9.5) * 5.5

    return {
        "sample_id": sample_id,
        "disaster_type": disaster_type,
        "magnitude": round(mag, 2),
        "building_collapse_rate": round(raw["building_collapse_rate"], 3),
        "estimated_trapped": int(raw["estimated_trapped"]),
        "temperature_c": round(raw["temperature_c"], 1),
        "hours_since_disaster": round(raw["hours_since_disaster"], 1),
        "rescue_eta_hours": round(raw["rescue_eta_hours"], 2),
        "road_accessibility": round(raw["road_accessibility"], 3),
        "medical_accessibility": round(raw["medical_accessibility"], 3),
        "rescue_skill_level": round(raw["rescue_skill_level"], 3),
        "night_time": int(round(raw["night_time"])),
        "holiday_event": int(round(raw["holiday_event"])),
    }


def generate(output_path: Path, n_samples: int = N_SAMPLES, seed: int = SEED) -> None:
    sampler = qmc.LatinHypercube(d=len(FEATURE_SPACE), seed=seed)

    # 各维度的 bounds
    lows  = [s[1] for s in FEATURE_SPACE]
    highs = [s[2] for s in FEATURE_SPACE]

    # 生成 LHS 矩阵
    latin_raw = sampler.random(n=n_samples)
    samples_norm = qmc.scale(latin_raw, lows, highs)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(samples_norm):
            raw = {FEATURE_SPACE[j][0]: row[j] for j in range(len(FEATURE_SPACE))}
            # 离散化
            raw["estimated_trapped"] = int(round(raw["estimated_trapped"]))
            sample = _build_sample(i, raw, total=n_samples)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[generate_augmented] wrote {n_samples} samples → {out_path}")
    print(f"  features: {[s[0] for s in FEATURE_SPACE]}")
    print(f"  disaster distribution:")
    dt_counts: dict = {}
    for i in range(n_samples):
        dt = _disaster_type_for_index(i, n_samples)
        dt_counts[dt] = dt_counts.get(dt, 0) + 1
    for dt, cnt in sorted(dt_counts.items()):
        print(f"    {dt}: {cnt} ({cnt/n_samples*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/augmented_samples.jsonl")
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(ROOT / args.output, n_samples=args.n, seed=args.seed)
