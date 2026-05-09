"""Phase 4：Student 答题训练 + 双向蒸馏循环。

流程：
  data/test_set.jsonl (100道题 + ground_truth from Teacher)
      ↓
  Student 对每道题推理 → score
      ↓
  与 Teacher ground_truth_score 对比 → 偏差作为 soft label
      ↓
  data/bidirectional_train.jsonl
      ↓
  student/train.py → 重新训练 model.pkl

使用方法：
    python3 -m student.bidirectional_train
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.features import vectorize


def load_test_set(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                continue
            records.append(obj)
    return records


def main() -> None:
    import joblib
    import yaml

    cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
    model_path = ROOT / cfg["paths"]["model"]
    test_set_path = ROOT / "data" / "test_set.jsonl"
    output_path = ROOT / "data" / "bidirectional_train.jsonl"

    # 加载 Student 模型
    booster = joblib.load(model_path)
    best_iter = getattr(booster, "best_iteration", None)

    # 加载测试题
    test_records = load_test_set(test_set_path)
    if not test_records:
        print("[bidirectional] test_set.jsonl 为空或全部失败，跳过")
        return
    print(f"[bidirectional] loaded {len(test_records)} test items")

    # Student 推理 → 与 Teacher ground_truth 对比 → 生成训练样本
    new_samples = []
    for rec in test_records:
        sample = rec.get("input", {})
        gt_score = float(rec["score"])
        gt_level = rec.get("risk_level", "较大")
        gt_conf = float(rec.get("confidence", 0.8))

        # Student 推理
        X = np.asarray([vectorize(sample)], dtype=np.float32)
        pred_score = float(booster.predict(X, num_iteration=best_iter)[0])
        pred_score = max(0.0, min(100.0, pred_score))

        # 偏差 = Student 预测 - Teacher 标准答案（作为 soft label 的微调信号）
        diff = pred_score - gt_score

        # 生成新的训练样本：input + student_pred + teacher_gt + diff_weight
        # confidence 折扣：偏差越大，confidence 越低
        error_ratio = min(abs(diff) / 30.0, 1.0)
        adjusted_conf = max(0.3, gt_conf * (1.0 - error_ratio * 0.5))

        new_samples.append({
            "sample_id": f"BID{rec.get('sample_id', 'UNK')}",
            "input": sample,
            "score": round(pred_score, 2),
            "risk_level": _level(pred_score),
            "confidence": round(adjusted_conf, 3),
            "teacher_gt_score": round(gt_score, 2),
            "teacher_gt_level": gt_level,
            "diff": round(diff, 2),
            "source": "bidirectional",
        })

    # 写入训练集
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in new_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[bidirectional] wrote {len(new_samples)} samples → {output_path}")

    # 统计偏差
    diffs = [s["diff"] for s in new_samples]
    print(f"  diff mean={np.mean(diffs):.2f} std={np.std(diffs):.2f}")
    print(f"  |diff| > 10: {sum(1 for d in diffs if abs(d) > 10)}")


def _level(score: float) -> str:
    if score < 25:
        return "一般"
    if score < 50:
        return "较大"
    if score < 75:
        return "重大"
    return "特别重大"


if __name__ == "__main__":
    main()
