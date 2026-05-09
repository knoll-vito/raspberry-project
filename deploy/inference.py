"""阶段 6：树莓派离线推理脚本（含完整 JSON 决策包格式）。

设计原则：
- 进程启动时一次性加载 model.pkl（避免每次推理重新加载）
- 单条 / 批量推理 API 一致
- 不依赖 openai / pandas，只需 lightgbm + joblib + numpy
- predict_full() 输出完整决策 JSON：header + inference + SHAP + CoT trace

CLI 用法：
    # 简洁输出
    python deploy/inference.py --json '{"disaster_type":"earthquake", ...}'

    # 完整决策包（含 SHAP / 优先级 / 置信度 / GPS 等）
    python deploy/inference.py --json '...' --full --device-id RPI5-EM-01 \\
        --lat 31.284 --lon 121.503
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import FEATURE_ORDER, vectorize, score_to_level as _level  # noqa: E402
from teacher import mock_labeler  # noqa: E402

DEFAULT_MODEL_PATH = ROOT / "student" / "model.pkl"

# 中文官方 4 等级 ↔ 英文枚举（用于上游对接消费）
LEVEL_ZH_TO_EN: Dict[str, str] = {
    "一般": "GENERAL",
    "较大": "LARGE",
    "重大": "MAJOR",
    "特别重大": "EXTREME",
}
LEVEL_TO_PRIORITY: Dict[str, int] = {
    "特别重大": 0,  # 0 = 最高优先级
    "重大": 1,
    "较大": 2,
    "一般": 3,
}

# 内部特征名 → 输出展示名（更短，对接外部更友好）
FEATURE_DISPLAY_NAME: Dict[str, str] = {
    "is_earthquake": "is_earthquake",
    "is_flood": "is_flood",
    "is_urban_fire": "is_urban_fire",
    "is_forest_fire": "is_forest_fire",
    "is_landslide": "is_landslide",
    "magnitude": "magnitude",
    "building_collapse_rate": "collapse_rate",
    "estimated_trapped": "trapped_count",
    "temperature_c": "temp_c",
    "hours_since_disaster": "hours_since_event",
    "rescue_eta_hours": "rescue_eta",
    "road_accessibility": "road_access",
}

# Teacher alignment baseline = 验证集 Pearson r（来自 evaluate.py 输出）
# 当数据集或模型改动后，需手工同步该常数
TEACHER_ALIGNMENT_BASELINE: float = 0.9021

DEFAULT_DEVICE_ID = os.environ.get("DEVICE_ID", "RPI5-EM-01")


# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _utc_iso_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _compute_confidence(score: float, mock_score: float) -> float:
    """启发式置信度，bounded [0.50, 0.99]：

    - 距离最近等级边界（25/50/75）越远 → 越自信
    - 与 Mock baseline 分歧越大 → 越不自信
    """
    base = 0.92
    boundaries = [25.0, 50.0, 75.0]
    nearest = min(abs(score - b) for b in boundaries)
    if nearest < 5:
        base -= 0.10
    elif nearest < 10:
        base -= 0.05

    diff = abs(score - mock_score)
    if diff > 20:
        base -= 0.15
    elif diff > 10:
        base -= 0.05

    return round(max(0.50, min(0.99, base)), 4)


def _top_shap_features(contrib_row: np.ndarray, top_n: int = 4,
                       eps: float = 0.01) -> List[Dict]:
    """LightGBM TreeSHAP 贡献（按绝对值归一化），返回 top N。

    每条特征同时输出：
      - value:     绝对值归一化后的相对重要度（与旧版兼容）
      - raw_shap:  保留正负号的原始 SHAP 贡献（推高 score 为正，拉低为负）
      - direction: "↑" 表示推高 score / 加剧风险；"↓" 表示拉低 score / 缓解风险
    """
    contributions = contrib_row[:-1]  # 最后一列是 base value
    abs_contribs = np.abs(contributions)
    total = float(abs_contribs.sum())
    if total <= 0:
        return []
    normalized = abs_contribs / total
    ranked = sorted(enumerate(normalized), key=lambda x: -x[1])
    out: List[Dict] = []
    for idx, val in ranked[:top_n]:
        if val < eps:
            break
        feat_internal = FEATURE_ORDER[idx]
        raw = float(contributions[idx])
        out.append({
            "feature": FEATURE_DISPLAY_NAME.get(feat_internal, feat_internal),
            "value": round(float(val), 4),
            "raw_shap": round(raw, 2),
            "direction": "↑" if raw >= 0 else "↓",
        })
    return out


def _build_trace_summary(sample: Dict, top_features: List[Dict],
                         level_en: str, priority: int) -> str:
    """从 top SHAP 特征生成一行带方向的 trace summary。

    格式示例：
        Top drivers: road_access↓ (-7.2) + collapse_rate↑ (+6.1) + magnitude↑ (+5.9)
                     → EXTREME (priority 0)
    箭头 + 符号让调用方一眼区分风险驱动 (↑) 与风险缓解 (↓)。
    """
    if not top_features:
        return f"Low-signal scenario; level={level_en}, priority={priority}"

    parts: List[str] = []
    for f in top_features[:3]:
        display = f["feature"]
        direction = f.get("direction", "")
        raw = f.get("raw_shap")
        if raw is None:
            parts.append(f"{display}{direction}")
        else:
            parts.append(f"{display}{direction} ({raw:+.1f})")

    return f"Top drivers: {' + '.join(parts)} → {level_en} (priority {priority})"


# ── 核心类 ────────────────────────────────────────────────────────────────


class RiskScorer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        import joblib
        self.booster = joblib.load(model_path)
        self.best_iter = getattr(self.booster, "best_iteration", None)

    def predict_one(self, sample: Dict) -> Dict:
        """简洁输出 — 用于 benchmark / batch / 内部调用。"""
        X = np.asarray([vectorize(sample)], dtype=np.float32)
        score = float(self.booster.predict(X, num_iteration=self.best_iter)[0])
        score = max(0.0, min(100.0, score))
        return {"score": round(score, 2), "risk_level": _level(score)}

    def predict_batch(self, samples: List[Dict]) -> List[Dict]:
        X = np.asarray([vectorize(s) for s in samples], dtype=np.float32)
        scores = self.booster.predict(X, num_iteration=self.best_iter)
        return [
            {"score": round(float(max(0, min(100, s))), 2), "risk_level": _level(float(s))}
            for s in scores
        ]

    def predict_full(self, sample: Dict, *,
                     device_id: Optional[str] = None,
                     location: Optional[Dict] = None) -> Dict:
        """完整决策 JSON 包：header + inference_result + SHAP + CoT trace。"""
        X = np.asarray([vectorize(sample)], dtype=np.float32)
        score = float(self.booster.predict(X, num_iteration=self.best_iter)[0])
        score = max(0.0, min(100.0, score))
        level_zh = _level(score)
        level_en = LEVEL_ZH_TO_EN.get(level_zh, "UNKNOWN")
        priority = LEVEL_TO_PRIORITY.get(level_zh, 3)

        # Mock baseline 用于置信度计算（边端零网络可用）
        mock_label = mock_labeler.label(sample)
        mock_score = float(mock_label["score"])
        confidence = _compute_confidence(score, mock_score)

        # LightGBM TreeSHAP 贡献
        contrib = self.booster.predict(
            X, num_iteration=self.best_iter, pred_contrib=True,
        )
        top_features = _top_shap_features(contrib[0])
        trace_summary = _build_trace_summary(sample, top_features, level_en, priority)

        return {
            "header": {
                "device_id": device_id or DEFAULT_DEVICE_ID,
                "timestamp": _utc_iso_now(),
                "location": location,  # None 表示无 GPS 数据
            },
            "inference_result": {
                "score": round(score, 2),
                "level": level_en,
                "level_zh": level_zh,
                "confidence": confidence,
                "priority": priority,
            },
            "explainability_analysis": {
                "method": "LightGBM_TreeSHAP",
                "feature_importance": top_features,
            },
            "cot_reasoning_trace": {
                "trace_summary": trace_summary,
                "teacher_alignment": TEACHER_ALIGNMENT_BASELINE,
            },
        }


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--json", help="单条样本 JSON 字符串")
    parser.add_argument("--file", help="批量推理：每行一条 JSON 的文件路径")
    parser.add_argument("--full", action="store_true",
                        help="输出完整决策 JSON（含 SHAP / 优先级 / GPS 等）")
    parser.add_argument("--device-id", default=None,
                        help="设备 ID（默认环境变量 DEVICE_ID 或 RPI5-EM-01）")
    parser.add_argument("--lat", type=float, default=None, help="GPS 纬度")
    parser.add_argument("--lon", type=float, default=None, help="GPS 经度")
    args = parser.parse_args()

    scorer = RiskScorer(Path(args.model))

    location = None
    if args.lat is not None and args.lon is not None:
        location = {"lat": args.lat, "lon": args.lon}

    if args.json:
        sample = json.loads(args.json)
        if args.full:
            result = scorer.predict_full(sample, device_id=args.device_id, location=location)
        else:
            result = scorer.predict_one(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.file:
        samples = []
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if args.full:
            for s in samples:
                result = scorer.predict_full(s, device_id=args.device_id, location=location)
                print(json.dumps(result, ensure_ascii=False))
        else:
            results = scorer.predict_batch(samples)
            for s, r in zip(samples, results):
                sid = s.get("sample_id", "?")
                print(f"{sid:8s}  score={r['score']:6.2f}  level={r['risk_level']}")
        return

    parser.error("must provide --json or --file")


if __name__ == "__main__":
    main()
