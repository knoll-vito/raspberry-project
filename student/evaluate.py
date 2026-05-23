"""一致性评估：Student vs Teacher（仅 hold-out 集）。

修复版：复用 train.py 同样的 (seed, test_size) 重切，只在 80/20 划分里
那 20% 验证集上计算指标，避免训练数据污染评估。

输出：
- valid 集 MAE / RMSE / Pearson / level_accuracy（**主指标**）
- train 集同款指标（仅作对比，看 overfit 程度）
- top10 disagreement 也只从 valid 集挑

设计原则：评估必须建立在模型从未见过的样本上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import load_clean, FEATURE_ORDER, score_to_level as _level  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if np.std(pred) > 1e-9 and np.std(y) > 1e-9:
        pearson = float(np.corrcoef(pred, y)[0, 1])
    else:
        pearson = 0.0
    pred_levels = [_level(p) for p in pred]
    teacher_levels = [_level(s) for s in y]
    level_acc = float(np.mean([a == b for a, b in zip(teacher_levels, pred_levels)]))
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": round(pearson, 4),
        "level_accuracy": round(level_acc, 4),
    }


def main() -> None:
    import joblib
    from sklearn.model_selection import train_test_split

    cfg = load_config()
    scfg = cfg["student"]
    in_path = ROOT / cfg["paths"]["clean_dataset"]
    model_path = ROOT / cfg["paths"]["model"]
    report_path = ROOT / cfg["paths"]["eval_report"]

    booster = joblib.load(model_path)
    X, y_soft, y_hard, w = load_clean(in_path)
    n = len(X)

    # 复用 train.py 的同一 (seed, test_size) 切分；按下标切回原始位置
    indices = np.arange(n)
    idx_tr, idx_va = train_test_split(
        indices,
        test_size=float(scfg["test_size"]),
        random_state=int(scfg["seed"]),
    )

    # 加载原始记录（带 input/event_name 等元数据）
    raw_records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    pred_all = booster.predict(X, num_iteration=booster.best_iteration)

    # 主指标：仅在 valid 集
    pred_va = pred_all[idx_va]
    y_va = y_soft[idx_va]
    valid_metrics = _metrics(pred_va, y_va)

    # 对照：train 集（看 overfit gap）
    pred_tr = pred_all[idx_tr]
    y_tr = y_soft[idx_tr]
    train_metrics = _metrics(pred_tr, y_tr)

    # top10 disagreement 只从 valid 集挑
    err_va = pred_va - y_va
    top_local = np.argsort(-np.abs(err_va))[:10]
    disagreements = []
    real_in_top = 0
    for li in top_local:
        global_i = idx_va[li]
        rec = raw_records[global_i]
        is_real = bool(rec.get("input", {}).get("event_name"))
        if is_real:
            real_in_top += 1
        disagreements.append({
            "sample_id": rec.get("sample_id"),
            "is_real": is_real,
            "input": rec.get("input"),
            "teacher_score": float(y_va[li]),
            "student_score": float(pred_va[li]),
            "abs_err": float(abs(err_va[li])),
        })

    # 数据集组成
    n_real = sum(1 for r in raw_records if r.get("input", {}).get("event_name"))
    composition = {
        "n_total": n,
        "n_real": n_real,
        "n_synthetic": n - n_real,
        "n_train": int(len(idx_tr)),
        "n_valid": int(len(idx_va)),
    }

    print("[evaluate] dataset")
    for k, v in composition.items():
        print(f"  {k:13s} {v}")

    print("\n[evaluate] === VALIDATION (主指标，模型未见过) ===")
    for k, v in valid_metrics.items():
        print(f"  {k:18s} {v}")

    print("\n[evaluate] === TRAIN（对比，模型已见过） ===")
    for k, v in train_metrics.items():
        print(f"  {k:18s} {v}")

    gap = round(valid_metrics["rmse"] - train_metrics["rmse"], 3)
    print(f"\n[evaluate] overfit gap (valid-train RMSE) = {gap}")
    print(f"[evaluate] real samples in top10 valid disagreement: {real_in_top}/10")

    report = {
        "composition": composition,
        "valid_metrics": valid_metrics,
        "train_metrics": train_metrics,
        "overfit_gap_rmse": gap,
        "top10_valid_disagreements": disagreements,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[evaluate] full report → {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
