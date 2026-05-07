"""特征工程：清洗后的 jsonl → numpy 数组。

把 disaster_type 做 one-hot，其它数值特征保持原值（LightGBM 不需要标准化）。
特征顺序固化在 FEATURE_ORDER，训练 & 推理共用，避免顺序错位。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DISASTER_TYPES = ["earthquake", "flood", "fire", "landslide"]

NUMERIC_FEATURES = [
    "magnitude",
    "building_collapse_rate",
    "estimated_trapped",
    "temperature_c",
    "hours_since_disaster",
    "rescue_eta_hours",
    "road_accessibility",
]

FEATURE_ORDER = [f"is_{t}" for t in DISASTER_TYPES] + NUMERIC_FEATURES


def vectorize(sample: Dict) -> List[float]:
    dtype = sample.get("disaster_type", "")
    onehot = [1.0 if dtype == t else 0.0 for t in DISASTER_TYPES]
    nums = [float(sample.get(k, 0)) for k in NUMERIC_FEATURES]
    return onehot + nums


def load_clean(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 clean_dataset.jsonl，返回 X, y_soft, y_hard_level, weights。

    y_soft: float 风险评分
    y_hard_level: int 0~3 对应 low/medium/high/critical（备用，可用于多任务）
    weights: confidence
    """
    LEVEL_TO_INT = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    X, y_soft, y_hard, w = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            X.append(vectorize(rec["input"]))
            y_soft.append(float(rec["score"]))
            y_hard.append(LEVEL_TO_INT.get(rec.get("risk_level", "medium"), 1))
            w.append(float(rec.get("confidence", 1.0)))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_soft, dtype=np.float32),
        np.asarray(y_hard, dtype=np.int8),
        np.asarray(w, dtype=np.float32),
    )
