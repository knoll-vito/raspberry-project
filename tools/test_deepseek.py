"""DeepSeek API 连通性测试（单条调用，约 0.001 USD）。

验证 4 件事：
  1. .env 里的 DEEPSEEK_API_KEY 有效
  2. config.yaml 里 teacher.model 在该 endpoint 上确实存在
  3. prompt 模板 + parser 能完整跑通
  4. 输出 reasoning / score 符合救援领域常识

跑这个脚本不会动 data/ 下的任何文件。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 加载 .env 到环境变量
load_dotenv(ROOT / ".env")

from teacher.parser import parse  # noqa: E402
from teacher.prompt_templates import SYSTEM_PROMPT, build  # noqa: E402

# 一个中等危险度的地震样本——预期 score 落在 60~85 之间
TEST_SAMPLE = {
    "sample_id": "TEST001",
    "disaster_type": "earthquake",
    "magnitude": 6.5,
    "building_collapse_rate": 0.4,
    "estimated_trapped": 50,
    "temperature_c": 5,
    "hours_since_disaster": 12,
    "rescue_eta_hours": 6,
    "road_accessibility": 0.3,
}


def _load_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main() -> int:
    from openai import AsyncOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "sk-replace-me":
        print("❌ DEEPSEEK_API_KEY 未设置（或仍是模板值）。请编辑 .env 后重试。")
        return 1

    cfg = _load_cfg()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", cfg["teacher"]["base_url"])
    configured_model = cfg["teacher"]["model"]
    print(f"[test] base_url        = {base_url}")
    print(f"[test] configured model= {configured_model}")
    print(f"[test] api key tail    = ...{api_key[-4:]}")
    print()

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # 1) 列出 endpoint 可用模型，方便核验 model 名是否对
    print("[test] step 1: list available models")
    try:
        models = await client.models.list()
        names = sorted({m.id for m in models.data})
        for n in names:
            mark = "  ← configured" if n == configured_model else ""
            print(f"  - {n}{mark}")
        if configured_model not in names:
            print()
            print(f"⚠️  当前 config.yaml 的 teacher.model='{configured_model}' 不在以上列表中。")
            print( "   将 config.yaml 的 teacher.model 改成上面任一个可用名再重试。")
    except Exception as e:  # noqa: BLE001
        print(f"  (跳过：models.list 不可用 — {e})")
    print()

    # 2) 单条样本调用
    print("[test] step 2: single chat completion")
    print("  sample =", json.dumps(TEST_SAMPLE, ensure_ascii=False))
    prompt = build(TEST_SAMPLE, "v1")
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=configured_model,
            temperature=float(cfg["teacher"]["temperature"]),
            max_tokens=int(cfg["teacher"]["max_tokens"]),
            timeout=float(cfg["teacher"]["timeout_s"]),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:  # noqa: BLE001
        print(f"❌ API 调用失败：{e}")
        print("   常见原因：模型名错（见上方列表）/ 余额不足 / key 无权限 / 网络代理 / timeout 太短")
        return 2
    elapsed = time.time() - t0

    # 防御：openai SDK 在某些边界（content_filter / 无 choices / 空响应）
    # 会返回 resp=None 或 resp.choices=[]，并不抛异常
    if resp is None:
        print(f"❌ API 返回 None（延迟 {elapsed:.2f}s）—— 可能触发 content filter 或上游异常")
        return 2
    if not getattr(resp, "choices", None):
        print(f"❌ API 返回无 choices（延迟 {elapsed:.2f}s）—— 检查 finish_reason / safety policy")
        print(f"     raw resp = {resp!r}")
        return 2

    print(f"  ✅ 调用成功，延迟 {elapsed:.2f}s")
    if getattr(resp, "usage", None):
        print(f"     tokens: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}")
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    print()

    # 3) 打印原始输出
    print("[test] step 3: raw response")
    print("─" * 60)
    print(text)
    print("─" * 60)
    print()

    # 4) parser 能否解析
    print("[test] step 4: parse XML tags")
    parsed = parse(text)
    if parsed is None:
        print("❌ parse 失败：模型返回未严格匹配 <reasoning>/<score>/<risk_level>/<confidence> 标签。")
        print("   后续放量前可能需要：")
        print("   - 调高 prompt 强度（如 v2 强制推理链版）")
        print("   - 放宽 parser（在 teacher/parser.py 增加容错）")
        print("   - 使用 response_format=json 类参数（如果 V4 支持）")
        return 3
    print("  parsed:")
    for k, v in parsed.items():
        if k == "reasoning":
            head = v.replace("\n", " ")
            print(f"    reasoning  ({len(v)} chars): {head[:160]}{'…' if len(v)>160 else ''}")
        else:
            print(f"    {k:14s}: {v}")
    print()

    # 5) 领域常识 sanity check
    print("[test] step 5: domain sanity check")
    score = parsed["score"]
    if 55 <= score <= 90:
        print(f"  ✅ score={score:.1f} ∈ [55, 90]：与「6.5级地震+倒塌率0.4+50人被困+严寒」预期一致")
    else:
        print(f"  ⚠️  score={score:.1f} 偏离预期区间 [55, 90]：可能 prompt 需要进一步约束")
    if len(parsed["reasoning"]) >= 100:
        print(f"  ✅ reasoning {len(parsed['reasoning'])} 字符（≥ 100）")
    else:
        print(f"  ⚠️  reasoning 仅 {len(parsed['reasoning'])} 字符（< 100），考虑用 prompt v2")
    # 黄金 72h 段必答检查
    keywords = ["72", "黄金", "生还率", "暴露", "衰减"]
    if any(k in parsed["reasoning"] for k in keywords):
        print(f"  ✅ reasoning 包含黄金 72h 衰减分析关键词")
    else:
        print(f"  ⚠️  reasoning 未明显包含 72h 衰减分析（关键词检查），考虑加强 prompt")
    # 等级 enum 检查
    if parsed["risk_level"] in {"一般", "较大", "重大", "特别重大"}:
        print(f"  ✅ risk_level={parsed['risk_level']} 在四级官方 enum 内")
    else:
        print(f"  ⚠️  risk_level={parsed['risk_level']} 不在四级 enum，parser 已兜底")
    # event_category 检查
    if parsed.get("event_category") in {"自然灾害", "事故灾难", "公共卫生事件", "社会安全事件"}:
        print(f"  ✅ event_category={parsed['event_category']}")
    else:
        print(f"  ⚠️  event_category={parsed.get('event_category')} 缺失或不规范")
    if 0.5 <= parsed["confidence"] <= 1.0:
        print(f"  ✅ confidence={parsed['confidence']}")
    else:
        print(f"  ⚠️  confidence={parsed['confidence']} 异常")
    print()
    print("[test] done. 如果以上检查都 ✅ 或仅有可接受的 ⚠️，下一步：")
    print("       python scripts/run_labeling.py --provider deepseek --limit 5")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
