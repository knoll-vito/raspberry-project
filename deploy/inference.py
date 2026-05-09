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

from student.calibration import load_calibrator  # noqa: E402
from student.features import FEATURE_ORDER, vectorize, score_to_level as _level  # noqa: E402
from student.reasoning_generator import generate_student_reasoning  # noqa: E402
from teacher import mock_labeler  # noqa: E402
from teacher.recommendations import build_recommendations  # noqa: E402

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

# 内部特征名 → 数据采集来源（喂给 trace_summary 的叙事生成）
# RS = Remote Sensing（遥感）；GIS = Geographic Information System；
# WX = Weather Station；InSAR = 干涉雷达；FieldReport = 现场上报
FEATURE_SOURCE: Dict[str, str] = {
    "is_earthquake": "Seismograph",
    "is_flood": "RS+Hydro",
    "is_urban_fire": "Sensor+RS",
    "is_forest_fire": "Satellite-Thermal",
    "is_landslide": "RS+InSAR",
    "magnitude": "Seismograph",
    "building_collapse_rate": "RS",
    "estimated_trapped": "FieldReport",
    "temperature_c": "WX",
    "hours_since_disaster": "GPS",
    "rescue_eta_hours": "LogisticsPlanner",
    "road_accessibility": "GIS",
}

# 风险等级 → 死亡风险叙事尾标
LEVEL_TO_RISK_PHRASE: Dict[str, str] = {
    "EXTREME": "Extreme mortality risk",
    "MAJOR":   "High mortality risk",
    "LARGE":   "Moderate mortality risk",
    "GENERAL": "Manageable mortality risk",
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


def _phrase_for_feature(feature_internal: str, sample: Dict, direction: str) -> str:
    """把单个 top-SHAP 特征翻译成一句"通过 X 数据源观测到 Y"的自然语言短语。"""
    src = FEATURE_SOURCE.get(feature_internal, "Sensor")

    if feature_internal == "building_collapse_rate":
        v = sample.get("building_collapse_rate")
        if v is None:
            return f"Collapse rate observation via {src}"
        if v >= 0.7:
            qual = "High"
        elif v >= 0.3:
            qual = "Moderate"
        else:
            qual = "Low"
        return f"{qual} collapse rate detected via {src}"

    if feature_internal == "hours_since_disaster":
        h = sample.get("hours_since_disaster")
        if h is None:
            return f"{src} timestamp unavailable"
        if h > 72:
            return f"{src} timestamp > 72h (golden window expired)"
        if h > 48:
            return f"{src} timestamp > 48h"
        if h > 24:
            return f"{src} timestamp > 24h"
        return f"{src} timestamp = {h:g}h"

    if feature_internal == "road_accessibility":
        ra = sample.get("road_accessibility")
        if ra is None:
            return f"Road access unknown via {src}"
        if ra < 0.2:
            return f"{src}-based road isolation"
        if ra < 0.5:
            return f"{src}-based partial road blockage"
        return f"{src}-based road accessible"

    if feature_internal == "temperature_c":
        t = sample.get("temperature_c")
        if t is None:
            return f"Ambient temperature unknown ({src})"
        if t < 0:
            return f"{src} reports extreme cold ({t:g}°C, hypothermia risk)"
        if t > 35:
            return f"{src} reports extreme heat ({t:g}°C, heatstroke risk)"
        return f"{src} reports ambient {t:g}°C"

    if feature_internal == "magnitude":
        m = sample.get("magnitude")
        if m is None:
            return f"Intensity unknown via {src}"
        return f"{src} measures intensity {m:g}"

    if feature_internal == "estimated_trapped":
        n = sample.get("estimated_trapped")
        if n is None:
            return f"Trapped count unknown via {src}"
        if n >= 1000:
            return f"{src} estimates {int(n)} trapped (mass casualty)"
        if n >= 100:
            return f"{src} estimates {int(n)} trapped"
        return f"{src} estimates {int(n)} trapped (small scale)"

    if feature_internal == "rescue_eta_hours":
        eta = sample.get("rescue_eta_hours")
        if eta is None:
            return f"Rescue ETA unknown via {src}"
        if eta > 12:
            return f"{src} reports prolonged rescue ETA ({eta:g}h)"
        if eta > 6:
            return f"{src} reports moderate rescue ETA ({eta:g}h)"
        return f"{src} reports rapid rescue ETA ({eta:g}h)"

    if feature_internal.startswith("is_"):
        disaster = feature_internal[3:]
        if sample.get("disaster_type") == disaster:
            return f"{src} confirms {disaster} event"
        return f"{src} signal for {disaster}"

    return f"{feature_internal} via {src}"


def _build_trace_summary(sample: Dict, top_features: List[Dict],
                         level_en: str, priority: int) -> str:
    """数据源叙事版 trace summary：

    例：
        High collapse rate detected via RS + GPS timestamp > 24h
        + GIS-based road isolation = Extreme mortality risk.

    每条 top-SHAP 特征翻译成「数据源 + 观测值/阈值」的自然语言短语，
    末尾按风险等级附 mortality risk 标签。priority 不再出现在文本里
    （它已经在 inference_result.priority 字段，不必重复）。
    """
    if not top_features:
        return f"Low-signal scenario; level={level_en}."

    phrases: List[str] = []
    for f in top_features[:3]:
        display = f["feature"]
        # 反查内部名以从 sample 取值 / 拿 SOURCE
        internal = next(
            (k for k, v in FEATURE_DISPLAY_NAME.items() if v == display),
            display,
        )
        direction = f.get("direction", "↑")
        phrases.append(_phrase_for_feature(internal, sample, direction))

    risk_phrase = LEVEL_TO_RISK_PHRASE.get(level_en, "Elevated mortality risk")
    return " + ".join(phrases) + f" = {risk_phrase}."


# ── 核心类 ────────────────────────────────────────────────────────────────


class RiskScorer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        import joblib
        self.booster = joblib.load(model_path)
        self.best_iter = getattr(self.booster, "best_iteration", None)
        self._calibrator = load_calibrator()

    def _calibrate(self, score: float) -> float:
        if self._calibrator is None:
            return score
        return float(self._calibrator.predict([score])[0])

    def predict_one(self, sample: Dict) -> Dict:
        """简洁输出 — 用于 benchmark / batch / 内部调用。"""
        X = np.asarray([vectorize(sample)], dtype=np.float32)
        raw = float(self.booster.predict(X, num_iteration=self.best_iter)[0])
        score = max(0.0, min(100.0, self._calibrate(raw)))
        return {"score": round(score, 2), "risk_level": _level(score)}

    def predict_batch(self, samples: List[Dict]) -> List[Dict]:
        X = np.asarray([vectorize(s) for s in samples], dtype=np.float32)
        raw_scores = self.booster.predict(X, num_iteration=self.best_iter)
        return [
            {"score": round(float(max(0, min(100, self._calibrate(s)))), 2),
             "risk_level": _level(float(max(0, min(100, self._calibrate(s)))))}
            for s in raw_scores
        ]

    def predict_full(self, sample: Dict, *,
                     device_id: Optional[str] = None,
                     location: Optional[Dict] = None,
                     timestamp: Optional[str] = None) -> Dict:
        """完整决策 JSON 包：header + inference_result + SHAP + CoT trace。

        header 字段优先级（高 → 低）：
          CLI / 调用方显式传入 → sample 自带的元数据 → 默认值
        让 starter_real.csv 的 lat/lon/event_time_utc/device_id 等元数据
        在批量推理时自动落进 header，不必手动传 --lat / --lon。
        """
        X = np.asarray([vectorize(sample)], dtype=np.float32)
        raw = float(self.booster.predict(X, num_iteration=self.best_iter)[0])
        score = max(0.0, min(100.0, self._calibrate(raw)))
        level_zh = _level(score)
        level_en = LEVEL_ZH_TO_EN.get(level_zh, "UNKNOWN")
        priority = LEVEL_TO_PRIORITY.get(level_zh, 3)

        # Mock baseline 用于置信度计算（边端零网络可用）
        mock_label = mock_labeler.label(sample)
        mock_score = float(mock_label["score"])
        confidence = _compute_confidence(score, mock_score)

        # LightGBM TreeSHAP 贡献（保留 raw_shap / direction 扩展字段）
        contrib = self.booster.predict(
            X, num_iteration=self.best_iter, pred_contrib=True,
        )
        top_features = _top_shap_features(contrib[0])

        # Student 动态 reasoning（基于 SHAP top features，不再雷同）
        reasoning_text = generate_student_reasoning(
            sample, score, top_features, level_zh,
        )
        trace_summary = _build_trace_summary(sample, top_features, level_en, priority)

        # ── header 元数据回退链 ────────────────────────────────────────────
        eff_device_id = device_id or sample.get("device_id") or DEFAULT_DEVICE_ID

        eff_location = location
        if eff_location is None:
            lat = sample.get("lat")
            lon = sample.get("lon")
            if lat is not None and lon is not None:
                eff_location = {"lat": lat, "lon": lon}

        eff_timestamp = timestamp or sample.get("event_time_utc") or _utc_iso_now()

        # 针对性救援建议（与 mock_labeler 共享同一规则引擎，保证表述一致）
        recommendations = build_recommendations(sample, level_zh, max_items=4)

        return {
            "header": {
                "device_id": eff_device_id,
                "timestamp": eff_timestamp,
                "location": eff_location,  # None 表示无 GPS 数据
            },
            "inference_result": {
                "score": round(score, 2),
                "level": level_en,
                "level_zh": level_zh,  # 扩展字段：保留中文等级以兼容下游中文台账
                "confidence": confidence,
                "priority": priority,
            },
            "explainability_analysis": {
                "method": "SHAP_Values",  # 模板字段
                "feature_importance": top_features,  # 含扩展字段 raw_shap / direction
            },
            "cot_reasoning_trace": {
                "trace_summary": trace_summary,
                "reasoning": reasoning_text,
                "teacher_alignment": TEACHER_ALIGNMENT_BASELINE,
            },
            # 扩展字段：1~4 条针对性救援建议（按 priority + tag + text 结构化输出）
            "recommendations": recommendations,
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
