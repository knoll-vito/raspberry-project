"""Post-hoc score calibration via Anchored Isotonic Regression.

LightGBM 回归天然有均值回归倾向：高分被压低、中分被抬高。
用 Isotonic Regression 拟合 Student→Teacher 的单调映射来修正偏差，
并在低分段加入 identity anchor 防止外推崩坏。

用法：
    # 1) 拟合并保存校准器
    python student/calibration.py --fit

    # 2) 查看校准效果
    python student/calibration.py --eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CALIBRATOR_PATH = ROOT / "student" / "calibrator.pkl"

IDENTITY_ANCHORS = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def fit_calibrator(
    student_scores: np.ndarray,
    teacher_scores: np.ndarray,
    out_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> IsotonicRegression:
    combined_s = np.concatenate([IDENTITY_ANCHORS, student_scores])
    combined_t = np.concatenate([IDENTITY_ANCHORS, teacher_scores])

    ir = IsotonicRegression(y_min=0.0, y_max=100.0, out_of_bounds="clip")
    ir.fit(combined_s, combined_t)
    joblib.dump(ir, out_path)
    print(f"[calibration] saved calibrator → {out_path.relative_to(ROOT)}")
    return ir


def load_calibrator(path: Path = DEFAULT_CALIBRATOR_PATH) -> IsotonicRegression | None:
    if path.exists():
        return joblib.load(path)
    return None


def calibrate(ir: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return np.clip(ir.predict(scores), 0.0, 100.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="拟合校准器")
    parser.add_argument("--eval", action="store_true", help="评估校准效果")
    parser.add_argument("--eval-data", default="data/eval_test_set.jsonl")
    args = parser.parse_args()

    eval_path = ROOT / args.eval_data

    from deploy.inference import RiskScorer
    scorer = RiskScorer()

    teacher_scores = []
    student_scores = []

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            teacher_scores.append(rec["score"])
            pred = scorer.predict_one(rec["input"])
            student_scores.append(pred["score"])

    t = np.array(teacher_scores)
    s = np.array(student_scores)

    if args.fit:
        ir = fit_calibrator(s, t)
        cal = calibrate(ir, s)

        print(f"\n[calibration] before → RMSE={np.sqrt(((s-t)**2).mean()):.2f}, "
              f"MAE={np.abs(s-t).mean():.2f}, bias={np.mean(s-t):+.2f}")
        print(f"[calibration] after  → RMSE={np.sqrt(((cal-t)**2).mean()):.2f}, "
              f"MAE={np.abs(cal-t).mean():.2f}, bias={np.mean(cal-t):+.2f}")

        print(f"\n[calibration] score range: {s.min():.1f}~{s.max():.1f} → {cal.min():.1f}~{cal.max():.1f}")

        from student.features import score_to_level
        before_acc = sum(1 for si, ti in zip(s, t) if score_to_level(si) == score_to_level(ti)) / len(t)
        after_acc = sum(1 for ci, ti in zip(cal, t) if score_to_level(ci) == score_to_level(ti)) / len(t)
        print(f"[calibration] level accuracy: {before_acc*100:.1f}% → {after_acc*100:.1f}%")

    if args.eval:
        ir = load_calibrator()
        if ir is None:
            print("[calibration] no calibrator found; run --fit first")
            return

        cal = calibrate(ir, s)
        diff_before = s - t
        diff_after = cal - t

        print("=== Calibration Effect by Score Band ===")
        for lo, hi in [(0, 50), (50, 75), (75, 85), (85, 95), (95, 100)]:
            mask = (t >= lo) & (t < (hi + 1 if hi == 100 else hi))
            n = mask.sum()
            if n == 0:
                continue
            print(f"  Teacher [{lo:3d},{hi:3d}): n={n:3d}  "
                  f"before: bias={diff_before[mask].mean():+5.1f} rmse={np.sqrt((diff_before[mask]**2).mean()):5.1f}  "
                  f"after:  bias={diff_after[mask].mean():+5.1f} rmse={np.sqrt((diff_after[mask]**2).mean()):5.1f}")

        print(f"\n  Overall RMSE: {np.sqrt((diff_before**2).mean()):.2f} → {np.sqrt((diff_after**2).mean()):.2f}")
        print(f"  Overall MAE:  {np.abs(diff_before).mean():.2f} → {np.abs(diff_after).mean():.2f}")
        print(f"  Pearson r:    {np.corrcoef(t,s)[0,1]:.4f} → {np.corrcoef(t,cal)[0,1]:.4f}")

        print("\n=== Sample Calibration Mapping ===")
        test_pts = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        mapped = calibrate(ir, test_pts)
        for raw, cal_val in zip(test_pts, mapped):
            print(f"  {raw:5.0f} → {cal_val:5.1f}")


if __name__ == "__main__":
    main()
