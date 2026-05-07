"""阶段 6：树莓派离线推理脚本。

设计原则：
- 进程启动时一次性加载 model.pkl（避免每次推理重新加载）
- 单条 / 批量推理 API 一致
- 不依赖 openai / pandas，只需 lightgbm + joblib + numpy

CLI 用法（命令行 demo）:
    python deploy/inference.py --json '{"disaster_type":"earthquake", ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import vectorize  # noqa: E402

DEFAULT_MODEL_PATH = ROOT / "student" / "model.pkl"


def _level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "medium"
    if score < 75:
        return "high"
    return "critical"


class RiskScorer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        import joblib
        self.booster = joblib.load(model_path)
        self.best_iter = getattr(self.booster, "best_iteration", None)

    def predict_one(self, sample: Dict) -> Dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--json", help="单条样本 JSON 字符串")
    parser.add_argument("--file", help="批量推理：每行一条 JSON 的文件路径")
    args = parser.parse_args()

    scorer = RiskScorer(Path(args.model))

    if args.json:
        sample = json.loads(args.json)
        print(json.dumps(scorer.predict_one(sample), ensure_ascii=False, indent=2))
        return

    if args.file:
        samples = []
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        results = scorer.predict_batch(samples)
        for s, r in zip(samples, results):
            sid = s.get("sample_id", "?")
            print(f"{sid:8s}  score={r['score']:6.2f}  level={r['risk_level']}")
        return

    parser.error("must provide --json or --file")


if __name__ == "__main__":
    main()
