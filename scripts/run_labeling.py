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

# 自动加载 .env（仅当 python-dotenv 可用；mock 模式不需要 key）
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

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


def _peek_existing_teacher(out_path: Path) -> str | None:
    """返回 out_path 已有记录里的 teacher 字段（用于跨 provider 检测）；不存在返回 None。"""
    if not out_path.exists():
        return None
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec.get("teacher")
                if t:
                    return t
    except Exception:
        pass
    return None


def _maybe_reset(out_path: Path, fail_path: Path, expected_teacher_prefix: str, reset: bool) -> None:
    """处理跨 provider 冲突 + --reset 的语义：

    - --reset 永远清空（用户明确表达"重跑"意图，比如 prompt 改了）
    - 没传 --reset 但发现是另一个 provider 的输出：报错退出，避免污染数据集
    - 没传 --reset 且是同一 provider：断点续标，沿用旧记录
    """
    if reset:
        # 用 truncate 而非 unlink：兼容只读限制目录 + 保留已打开句柄
        for p in (out_path, fail_path):
            if p.exists():
                p.write_text("", encoding="utf-8")
        print(f"[run_labeling] --reset：已清空 {out_path.name} 与 {fail_path.name}")
        return

    existing = _peek_existing_teacher(out_path)
    if existing is None:
        return  # 干净状态
    if existing.startswith(expected_teacher_prefix):
        return  # 同 provider 断点续标
    print(
        f"⚠️  {out_path} 已存在且是另一个 teacher（'{existing}'）的输出。\n"
        f"    断点续标会因 sample_id 冲突跳过所有样本。\n"
        f"    解决：加 --reset 清空旧文件后重跑。"
    )
    sys.exit(2)


def run_mock(cfg: dict, limit: int | None = None, reset: bool = False) -> None:
    in_path = ROOT / cfg["paths"]["raw_samples"]
    out_path = ROOT / cfg["paths"]["teacher_labeled"]
    fail_path = ROOT / cfg["paths"]["failed_samples"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_reset(out_path, fail_path, expected_teacher_prefix="mock", reset=reset)

    samples = list(read_jsonl(in_path))
    if limit is not None:
        samples = samples[:limit]
    print(f"[run_labeling/mock] labeling {len(samples)} samples …")
    n_ok = 0
    # mock 永远 overwrite，因为它确定性快
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            jitter = 4.0 if s.get("magnitude", 0) >= 6.5 or s.get("building_collapse_rate", 0) > 0.7 else 1.5
            res = mock_labeler.label(s, jitter=jitter, seed=42 + i)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[run_labeling/mock] wrote {n_ok} → {out_path.relative_to(ROOT)}")


def run_deepseek(cfg: dict, limit: int | None = None, reset: bool = False) -> None:
    from teacher.deepseek_client import DeepSeekLabeler

    in_path = ROOT / cfg["paths"]["raw_samples"]
    out_path = ROOT / cfg["paths"]["teacher_labeled"]
    fail_path = ROOT / cfg["paths"]["failed_samples"]
    _maybe_reset(out_path, fail_path, expected_teacher_prefix="deepseek", reset=reset)

    samples = list(read_jsonl(in_path))
    if limit is not None:
        samples = samples[:limit]
        print(f"[run_labeling/deepseek] LIMIT 模式：仅前 {limit} 条用于试跑 / 验证")
    print(f"[run_labeling/deepseek] labeling {len(samples)} samples …")
    labeler = DeepSeekLabeler(cfg)
    stats = asyncio.run(
        labeler.label_dataset(samples, out_path, fail_path, prompt_version=cfg["teacher"]["prompt_version"])
    )
    print("[run_labeling/deepseek] done:", json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "deepseek"], default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="只标前 N 条（用于小批量验证）")
    parser.add_argument("--reset", action="store_true",
                        help="切换 provider 时清空旧的 teacher_labeled.jsonl / failed_samples.jsonl")
    args = parser.parse_args()

    cfg = load_config()
    provider = args.provider or cfg["teacher"]["provider"]
    if provider == "mock":
        run_mock(cfg, limit=args.limit, reset=args.reset)
    elif provider == "deepseek":
        run_deepseek(cfg, limit=args.limit, reset=args.reset)
    else:
        raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":
    main()
