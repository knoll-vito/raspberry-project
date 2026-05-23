"""阶段 5：Student 训练。

- 读取 clean_dataset.jsonl
- 80/20 划分
- LightGBM 回归（confidence 加权 + early stopping）
- 输出特征重要性
- 保存 model.pkl 与 feature_meta.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import FEATURE_ORDER, load_clean  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    import argparse
    import lightgbm as lgb
    import joblib
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="覆盖 config 中的 clean_dataset 路径（可传合并后的文件）")
    args = parser.parse_args()

    cfg = load_config()
    scfg = cfg["student"]
    in_path = ROOT / (args.dataset if args.dataset else cfg["paths"]["clean_dataset"])
    model_path = ROOT / cfg["paths"]["model"]
    meta_path = ROOT / cfg["paths"]["feature_meta"]

    X, y_soft, y_hard, w = load_clean(in_path)
    if len(X) < 10:
        raise RuntimeError(f"too few samples after filter: {len(X)}; check filter rules / dataset")
    print(f"[train] dataset: n={len(X)} features={X.shape[1]}")

    sample_w = w if scfg.get("use_confidence_weight") else None

    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        X, y_soft, sample_w if sample_w is not None else np.ones(len(X)),
        test_size=float(scfg["test_size"]),
        random_state=int(scfg["seed"]),
    )

    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=FEATURE_ORDER)
    valid_set = lgb.Dataset(X_va, label=y_va, weight=w_va, feature_name=FEATURE_ORDER, reference=train_set)

    params = dict(scfg["lightgbm"])
    n_estimators = int(params.pop("n_estimators", 300))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(int(scfg["early_stopping_rounds"])),
            lgb.log_evaluation(period=50),
        ],
    )

    # 特征重要性
    importance = sorted(
        zip(FEATURE_ORDER, booster.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )
    print("[train] feature importance (gain):")
    for name, val in importance:
        print(f"  {name:30s} {val:.1f}")

    # 落盘
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, model_path)
    meta = {
        "feature_order": FEATURE_ORDER,
        "n_train": len(X_tr),
        "n_valid": len(X_va),
        "best_iteration": booster.best_iteration,
        "feature_importance": [{"name": n, "gain": float(v)} for n, v in importance],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    size_kb = model_path.stat().st_size / 1024
    print(f"[train] saved → {model_path.relative_to(ROOT)} ({size_kb:.1f} KB)")
    print(f"[train] meta  → {meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
