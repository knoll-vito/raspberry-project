"""沙箱 smoke test：用 numpy-only 的最小实现端到端验证数据链路。

由于沙箱无外网装不上 lightgbm，本脚本：
- 跑通 generate → mock label → filter → 特征化
- 用 numpy 闭式解的加权岭回归充当"轻量 Student"
- 报告 MAE / Pearson / 推理延迟

⚠️ 这只是数据流 / 接口 / 配置正确性的 smoke 验证，
   真正训练请在装好 lightgbm 的本机跑 `python student/train.py`。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import FEATURE_ORDER, load_clean, vectorize  # noqa: E402


def ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """加权岭回归闭式解：β = (XᵀWX + αI)⁻¹ XᵀWy。"""
    X1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    W = np.diag(w)
    A = X1.T @ W @ X1 + alpha * np.eye(X1.shape[1])
    b = X1.T @ W @ y
    return np.linalg.solve(A, b)


def ridge_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    return X1 @ beta


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clean_path = ROOT / cfg["paths"]["clean_dataset"]

    if not clean_path.exists():
        print(f"[smoke] missing {clean_path}; run generate→label→filter first")
        sys.exit(1)

    X, y_soft, y_hard, w = load_clean(clean_path)
    print(f"[smoke] loaded n={len(X)} features={X.shape[1]} ({FEATURE_ORDER[:3]} ...)")
    if len(X) < 10:
        print("[smoke] too few rows; widen filter for PoC")
        sys.exit(1)

    # 80/20 划分（无 sklearn，手写）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_va = max(1, len(X) // 5)
    va = idx[:n_va]
    tr = idx[n_va:]

    beta = ridge_fit(X[tr], y_soft[tr], w[tr], alpha=1.0)
    pred = ridge_predict(beta, X[va])
    pred = np.clip(pred, 0, 100)

    err = pred - y_soft[va]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pearson = float(np.corrcoef(pred, y_soft[va])[0, 1]) if np.std(pred) > 1e-9 else 0.0
    print(f"[smoke] valid: MAE={mae:.2f}  RMSE={rmse:.2f}  pearson={pearson:.3f}")

    # 推理延迟
    sample = {
        "disaster_type": "earthquake", "magnitude": 6.5, "building_collapse_rate": 0.4,
        "estimated_trapped": 50, "temperature_c": 5, "hours_since_disaster": 12,
        "rescue_eta_hours": 6, "road_accessibility": 0.3,
    }
    xs = np.asarray([vectorize(sample)], dtype=np.float32)
    times = []
    for _ in range(2000):
        t0 = time.perf_counter()
        ridge_predict(beta, xs)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    print(f"[smoke] latency mean={times.mean():.4f}ms p99={np.percentile(times,99):.4f}ms")

    print("[smoke] OK — pipeline data flow verified end-to-end")
    print("        next: on your Mac → `pip install -r requirements.txt && python student/train.py`")


if __name__ == "__main__":
    main()
