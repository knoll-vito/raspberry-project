"""阶段 3：Teacher 标注编排。

支持两种 provider：
  --provider mock      使用 teacher.mock_labeler（无需 API key）
  --provider deepseek  使用 teacher.deepseek_client（需 DEEPSEEK_API_KEY）

输出：data/teacher_labeled.jsonl 与 data/failed_samples.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teacher import mock_labeler  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_mock(cfg: dict) -> None:
    in_path = ROOT / cfg["paths"]["raw_samples"]
    out_path = ROOT / cfg["paths"]["teacher_labeled"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(read_jsonl(in_path))
    print(f"[run_labeling/mock] labeling {len(samples)} samples …")
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            # 给 v2/v3 类型的样本加更大 jitter，模拟真实 Teacher 在边界样本上的不确定性
            jitter = 4.0 if s.get("magnitude", 0) >= 6.5 or s.get("building_collapse_rate", 0) > 0.7 else 1.5
            res = mock_labeler.label(s, jitter=jitter, seed=42 + i)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[run_labeling/mock] wrote {n_ok} → {out_path.relative_to(ROOT)}")


def run_deepseek(cfg: dict) -> None:
    # 延迟导入，避免没装 openai 时也能跑 mock
    from teacher.deepseek_client import DeepSeekLabeler

    in_path = ROOT / cfg["paths"]["raw_samples"]
    out_path = ROOT / cfg["paths"]["teacher_labeled"]
    fail_path = ROOT / cfg["paths"]["failed_samples"]

    samples = list(read_jsonl(in_path))
    print(f"[run_labeling/deepseek] labeling {len(samples)} samples …")
    labeler = DeepSeekLabeler(cfg)
    stats = asyncio.run(
        labeler.label_dataset(samples, out_path, fail_path, prompt_version=cfg["teacher"]["prompt_version"])
    )
    print("[run_labeling/deepseek] done:", json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "deepseek"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    provider = args.provider or cfg["teacher"]["provider"]
    if provider == "mock":
        run_mock(cfg)
    elif provider == "deepseek":
        run_deepseek(cfg)
    else:
        raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":
    main()
