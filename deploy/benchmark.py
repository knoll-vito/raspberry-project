"""推理 Benchmark：1000 次推理统计 P50/P95/P99 延迟。

也输出模型大小、内存占用（如果 psutil 可用）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.inference import RiskScorer, DEFAULT_MODEL_PATH  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    n_iter = int(cfg["deploy"]["benchmark_iterations"])
    n_warmup = int(cfg["deploy"]["warmup_iterations"])

    model_path = DEFAULT_MODEL_PATH
    scorer = RiskScorer(model_path)

    sample = {
        "disaster_type": "earthquake",
        "magnitude": 6.5,
        "building_collapse_rate": 0.4,
        "estimated_trapped": 50,
        "temperature_c": 5,
        "hours_since_disaster": 12,
        "rescue_eta_hours": 6,
        "road_accessibility": 0.3,
    }

    # warmup
    for _ in range(n_warmup):
        scorer.predict_one(sample)

    # benchmark
    times_ms = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        scorer.predict_one(sample)
        times_ms.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times_ms)

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    mean = float(np.mean(times))
    size_kb = model_path.stat().st_size / 1024

    print(f"[benchmark] iterations={n_iter} warmup={n_warmup}")
    print(f"  mean   {mean:.3f} ms")
    print(f"  P50    {p50:.3f} ms")
    print(f"  P95    {p95:.3f} ms")
    print(f"  P99    {p99:.3f} ms")
    print(f"  model  {size_kb:.1f} KB")

    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"  rss    {rss:.1f} MB")
    except Exception:
        pass

    # 验收检查
    targets = {"p99_ms": 5.0, "model_kb": 2048}
    print("[benchmark] acceptance:")
    print(f"  P99 < 5ms      : {'PASS' if p99 < targets['p99_ms'] else 'FAIL'} ({p99:.3f})")
    print(f"  model < 2MB    : {'PASS' if size_kb < targets['model_kb'] else 'FAIL'} ({size_kb:.1f}KB)")


if __name__ == "__main__":
    main()
