"""一致性评估：Student vs Teacher。

输出指标：
- MAE / RMSE
- Pearson r
- 风险等级分类准确率（low/medium/high/critical）
- Top10 预测分歧最大样本（导出到 evaluation_report.json 便于人工 review）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import load_clean, FEATURE_ORDER  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "medium"
    if score < 75:
        return "high"
    return "critical"


def main() -> None:
    import joblib

    cfg = load_config()
    in_path = ROOT / cfg["paths"]["clean_dataset"]
    model_path = ROOT / cfg["paths"]["model"]
    report_path = ROOT / cfg["paths"]["eval_report"]

    booster = joblib.load(model_path)
    X, y_soft, y_hard, w = load_clean(in_path)
    pred = booster.predict(X, num_iteration=booster.best_iteration)

    err = pred - y_soft
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if np.std(pred) > 1e-9 and np.std(y_soft) > 1e-9:
        pearson = float(np.corrcoef(pred, y_soft)[0, 1])
    else:
        pearson = 0.0

    teacher_levels = [_level(s) for s in y_soft]
    pred_levels = [_level(p) for p in pred]
    level_acc = float(np.mean([a == b for a, b in zip(teacher_levels, pred_levels)]))

    # Top10 分歧
    top_idx = np.argsort(-np.abs(err))[:10]
    disagreements = []
    raw_records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    for i in top_idx:
        rec = raw_records[i]
        disagreements.append({
            "sample_id": rec.get("sample_id"),
            "input": rec.get("input"),
            "teacher_score": float(y_soft[i]),
            "student_score": float(pred[i]),
            "abs_err": float(abs(err[i])),
        })

    report = {
        "n": int(len(X)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": round(pearson, 4),
        "level_accuracy": round(level_acc, 4),
        "top10_disagreements": disagreements,
    }
    print("[evaluate]")
    for k, v in report.items():
        if k != "top10_disagreements":
            print(f"  {k:18s} {v}")
    print(f"  top10_disagreements -> see {report_path.name}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
