"""把真实灾情数据 CSV 转成 raw_samples.jsonl。

用法：
  python3 tools/csv_to_jsonl.py data/your_real_data.csv

输出：
  data/raw_samples.jsonl  ← 直接覆盖（先备份合成数据）
  打印过滤报告：保留 / 丢弃 / 标记缺失字段的行号

Schema 见 docs/data_schema.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "disaster_type", "magnitude", "building_collapse_rate", "estimated_trapped",
    "temperature_c", "hours_since_disaster", "rescue_eta_hours", "road_accessibility",
]
OPTIONAL = ["event_name", "source"]
ALL_FIELDS = REQUIRED + OPTIONAL

VALID_DISASTER_TYPES = {"earthquake", "flood", "urban_fire", "forest_fire", "landslide"}

RANGES = {
    "magnitude": (3.0, 8.5),
    "building_collapse_rate": (0.0, 1.0),
    "estimated_trapped": (0, 10000),
    "temperature_c": (-40.0, 50.0),
    "hours_since_disaster": (0.0, 720.0),
    "rescue_eta_hours": (0.0, 168.0),
    "road_accessibility": (0.0, 1.0),
}

INT_FIELDS = {"estimated_trapped"}
FLOAT_FIELDS = set(RANGES) - INT_FIELDS


def _is_missing(v: str) -> bool:
    return v is None or v.strip() in {"", "?", "NA", "N/A", "null", "none"}


def _convert_value(field: str, raw: str, row_idx: int, errors: list, missing: list) -> object:
    if _is_missing(raw):
        if field in REQUIRED:
            missing.append((row_idx, field))
        return None
    try:
        if field == "disaster_type":
            v = raw.strip().lower()
            if v not in VALID_DISASTER_TYPES:
                errors.append((row_idx, field, f"'{raw}' 不在 {VALID_DISASTER_TYPES}"))
                return None
            return v
        if field in INT_FIELDS:
            v = int(float(raw))
        elif field in FLOAT_FIELDS:
            v = float(raw)
        else:
            return raw.strip()
        lo, hi = RANGES[field]
        if v < lo or v > hi:
            errors.append((row_idx, field, f"{v} 超出范围 [{lo}, {hi}]"))
            return None
        return v
    except ValueError:
        errors.append((row_idx, field, f"无法解析 '{raw}'"))
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="输入 CSV 路径")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw_samples.jsonl"),
                        help="输出 JSONL（默认 data/raw_samples.jsonl）")
    parser.add_argument("--max-missing", type=int, default=2,
                        help="一行最多允许多少个必填字段缺失（默认 2，对应 prompt v4）")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"❌ 找不到 {csv_path}", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing_cols = [c for c in REQUIRED if c not in header]
        if missing_cols:
            print(f"❌ CSV 缺列：{missing_cols}", file=sys.stderr)
            print(f"   header={sorted(header)}", file=sys.stderr)
            return 2

        kept = 0
        skipped = 0
        partial = 0
        errors: list = []
        out_lines: list = []

        for i, row in enumerate(reader, start=2):  # 第 2 行起（第 1 行是 header）
            row_missing: list = []
            row_errors: list = []
            sample = {}
            for field in ALL_FIELDS:
                v = _convert_value(field, row.get(field, ""), i, row_errors, row_missing)
                if v is not None:
                    sample[field] = v

            if row_errors:
                errors.extend(row_errors)
                skipped += 1
                continue
            if len(row_missing) > args.max_missing:
                skipped += 1
                errors.append((i, "*", f"必填缺失 {len(row_missing)} 个 > 阈值 {args.max_missing}"))
                continue
            if row_missing:
                partial += 1

            sample["sample_id"] = sample.get("event_name") or f"R{i:05d}"
            out_lines.append(json.dumps(sample, ensure_ascii=False))
            kept += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"[csv_to_jsonl] kept    = {kept}")
    print(f"[csv_to_jsonl] partial = {partial}（有缺失但仍保留，会进入 prompt v4）")
    print(f"[csv_to_jsonl] skipped = {skipped}")
    try:
        rel = out_path.resolve().relative_to(ROOT)
        print(f"[csv_to_jsonl] output  → {rel}")
    except ValueError:
        print(f"[csv_to_jsonl] output  → {out_path}")
    if errors:
        print("\n[csv_to_jsonl] 问题清单（前 20 条）:")
        for row_i, field, msg in errors[:20]:
            print(f"  行 {row_i} · {field}: {msg}")
    return 0 if errors == [] else 0  # 软失败：有问题但保留 kept 部分


if __name__ == "__main__":
    sys.exit(main())
