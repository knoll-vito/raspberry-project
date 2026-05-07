"""阶段 2：模拟灾情样本生成。

按方案 §阶段2 的 8 维特征矩阵生成均衡分布的样本。
- disaster_type: 类别均衡
- magnitude: 0.5 一档均匀采样
- building_collapse_rate, road_accessibility: 5 区间均衡
- estimated_trapped: 对数采样
- temperature_c: 极端温度加权
- hours_since_disaster: 黄金 72h 高密度
- rescue_eta_hours: 偏向短时间

输出：data/raw_samples.jsonl（无标签）
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


DISASTER_TYPES = ["earthquake", "flood", "fire", "landslide"]


def _sample_magnitude(rng: random.Random) -> float:
    # 3.0 ~ 8.5，每 0.5 一档均匀采样
    bins = [3.0 + 0.5 * i for i in range(12)]  # 3.0,3.5,...,8.5
    return rng.choice(bins)


def _sample_collapse_rate(rng: random.Random) -> float:
    # 5 个区间均衡：[0,0.2),...,[0.8,1.0]
    bucket = rng.randint(0, 4)
    return round(rng.uniform(bucket * 0.2, bucket * 0.2 + 0.2), 3)


def _sample_trapped(rng: random.Random) -> int:
    # 对数采样：log(1) ~ log(500)
    u = rng.uniform(0, 1)
    return int(math.exp(u * math.log(501)) - 1)


def _sample_temperature(rng: random.Random) -> float:
    # 极端温度加权：30% 概率落入极端段
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            return round(rng.uniform(-10, 0), 1)  # 严寒
        return round(rng.uniform(35, 45), 1)  # 酷热
    return round(rng.uniform(0, 35), 1)


def _sample_hours_since(rng: random.Random) -> float:
    # 0~72h 高密度，72~168h 长尾
    if rng.random() < 0.75:
        return round(rng.uniform(0, 72), 1)
    return round(rng.uniform(72, 168), 1)


def _sample_rescue_eta(rng: random.Random) -> float:
    # 偏向短时间：[0.5, 24]，对数采样
    u = rng.uniform(0, 1)
    return round(0.5 + (24 - 0.5) * (u ** 2), 2)


def _sample_road(rng: random.Random) -> float:
    bucket = rng.randint(0, 4)
    return round(rng.uniform(bucket * 0.2, bucket * 0.2 + 0.2), 3)


def make_sample(idx: int, rng: random.Random) -> dict:
    disaster_type = DISASTER_TYPES[idx % len(DISASTER_TYPES)]  # 均衡
    return {
        "sample_id": f"S{idx:05d}",
        "disaster_type": disaster_type,
        "magnitude": _sample_magnitude(rng),
        "building_collapse_rate": _sample_collapse_rate(rng),
        "estimated_trapped": _sample_trapped(rng),
        "temperature_c": _sample_temperature(rng),
        "hours_since_disaster": _sample_hours_since(rng),
        "rescue_eta_hours": _sample_rescue_eta(rng),
        "road_accessibility": _sample_road(rng),
    }


def main() -> None:
    cfg = load_config()
    n = int(cfg["sample_generation"]["n_samples"])
    seed = int(cfg["sample_generation"]["seed"])
    out_path = ROOT / cfg["paths"]["raw_samples"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            sample = make_sample(i, rng)  # i 用于均衡 disaster_type
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[generate_samples] wrote {n} samples → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
