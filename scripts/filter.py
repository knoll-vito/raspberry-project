"""阶段 4：质量过滤。

5 类规则（与方案 §阶段4 对齐）：
1. 格式合规性：必须有 reasoning + score + risk_level + confidence
2. 置信度门槛：confidence >= cfg.filter.confidence_min
3. 领域常识冲突：例如 magnitude>=7 但 score<50（地震） → 丢弃
4. 推理链完整性：reasoning 字符数 >= cfg.filter.reasoning_min_chars
5. 重复样本：按 input 特征哈希去重

输出：data/clean_dataset.jsonl + 过滤报告打印
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hash_input(inp: Dict) -> str:
    s = json.dumps(inp, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _violates_common_sense(rec: Dict) -> bool:
    inp = rec.get("input", {})
    score = float(rec.get("score", 0))
    # 7 级以上地震 + score < 40：高度可疑
    if inp.get("disaster_type") == "earthquake" and float(inp.get("magnitude", 0)) >= 7.0 and score < 40:
        return True
    # 倒塌率 > 0.8 + 被困 > 50 + score < 50
    if (
        float(inp.get("building_collapse_rate", 0)) > 0.8
        and int(inp.get("estimated_trapped", 0)) > 50
        and score < 50
    ):
        return True
    return False


def main() -> None:
    cfg = load_config()
    fcfg = cfg["filter"]
    in_path = ROOT / cfg["paths"]["teacher_labeled"]
    out_path = ROOT / cfg["paths"]["clean_dataset"]

    if not in_path.exists():
        print(f"[filter] input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    counters = {
        "total": 0, "format": 0, "confidence": 0, "common_sense": 0,
        "reasoning_len": 0, "duplicate": 0, "kept": 0,
    }
    seen_hashes = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            counters["total"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counters["format"] += 1
                continue

            # 规则 1：格式
            if not all(k in rec for k in ("reasoning", "score", "risk_level", "confidence", "input")):
                counters["format"] += 1
                continue
            # 规则 2：置信度
            if float(rec["confidence"]) < float(fcfg["confidence_min"]):
                counters["confidence"] += 1
                continue
            # 规则 3：常识
            if _violates_common_sense(rec):
                counters["common_sense"] += 1
                continue
            # 规则 4：推理长度
            if len(rec["reasoning"]) < int(fcfg["reasoning_min_chars"]):
                counters["reasoning_len"] += 1
                continue
            # 规则 5：去重
            if fcfg.get("enable_dedup", True):
                h = _hash_input(rec["input"])
                if h in seen_hashes:
                    counters["duplicate"] += 1
                    continue
                seen_hashes.add(h)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["kept"] += 1

    total = counters["total"]
    drop = total - counters["kept"]
    rate = (drop / total * 100) if total else 0
    print("[filter] report")
    for k, v in counters.items():
        print(f"  {k:15s} {v}")
    print(f"  drop_rate       {rate:.1f}%")
    print(f"  → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
