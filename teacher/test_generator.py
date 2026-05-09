"""Phase 2：Teacher 出题测试生成器。

使用 DeepSeek-V4 API，基于合成样本生成 100 道测试题（区别于 starter_real.csv），
每道题包含 question + ground_truth_score + ground_truth_reasoning。

用途：
- 作为 Student 答题训练的题目来源（bidirectional distillation）
- 验证 Student 模型在未见过的数据上的泛化能力

使用方法：
    from teacher.test_generator import TestGenerator
    gen = TestGenerator(cfg)
    results = gen.generate_from_file(augmented_path, output_path, n=100)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from teacher.parser import parse
from teacher.prompt_templates import SYSTEM_PROMPT

log = logging.getLogger("test_generator")

# 专门用于生成测试题的 system prompt
TEST_SYSTEM_PROMPT = """你是一名灾后人员风险评估专家。你需要扮演"出题人"角色，根据给定的灾情特征生成一道评估题目，并给出标准答案。

【出题格式要求】
题目为开放问答格式：
"根据以下灾情特征，评估被困人员伤亡风险，并给出风险评分(0-100)和详细推理过程。"

【标准答案格式】
请严格按以下 XML 标签格式输出：

<question>
（复述题目，以上述开放问答格式为准）
</question>
<reasoning>
（分步推理，至少包含三段：
 1) 关键风险因子识别；
 2) 黄金 72 小时衰减分析（必答）；
 3) 多因子叠加效应与最终评分依据。不少于 100 字。）
</reasoning>
<score>0~100 的浮点数，越高代表伤亡风险越大</score>
<risk_level>一般 | 较大 | 重大 | 特别重大 之一</risk_level>
<confidence>0~1 的浮点数</confidence>

请只输出上述 XML 标签内容，不要输出任何额外解释。"""


class TestGenerator:
    def __init__(self, cfg: dict):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", cfg["teacher"]["base_url"])
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.cfg = cfg["teacher"]
        self.semaphore: Optional[asyncio.Semaphore] = None
        self._token_usage = {"prompt": 0, "completion": 0, "calls": 0}

    @staticmethod
    def _load_done_ids(path: Path) -> set:
        if not path.exists():
            return set()
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id")
                    if sid:
                        ids.add(sid)
                except json.JSONDecodeError:
                    continue
        return ids

    def _format_sample(self, sample: dict) -> str:
        """把 sample 格式化成灾情特征描述字符串"""
        dt = sample.get("disaster_type", "未知")
        mag = sample.get("magnitude", "未知")
        cr = sample.get("building_collapse_rate", "未知")
        trapped = sample.get("estimated_trapped", "未知")
        temp = sample.get("temperature_c", "未知")
        hours = sample.get("hours_since_disaster", "未知")
        eta = sample.get("rescue_eta_hours", "未知")
        road = sample.get("road_accessibility", "未知")
        med = sample.get("medical_accessibility", "未知")
        skill = sample.get("rescue_skill_level", "未知")
        night = sample.get("night_time", "未知")
        holiday = sample.get("holiday_event", "未知")

        lines = [
            f"- 灾种 (disaster_type): {dt}",
            f"- 强度/震级 (magnitude): {mag}",
            f"- 建筑倒塌率 (building_collapse_rate): {cr}",
            f"- 估计被困人数 (estimated_trapped): {trapped}",
            f"- 环境温度 (temperature_c): {temp}℃",
            f"- 灾后已过小时数 (hours_since_disaster): {hours}h",
            f"- 救援预计 ETA (rescue_eta_hours): {eta}h",
            f"- 道路可通行性 (road_accessibility): {road}",
            f"- 医疗资源可及性 (medical_accessibility): {med}",
            f"- 救援队伍技能水平 (rescue_skill_level): {skill}",
            f"- 夜间事件 (night_time): {'是' if night else '否'}",
            f"- 节假日/大型活动期间 (holiday_event): {'是' if holiday else '否'}",
        ]
        return "\n".join(lines)

    async def _call_once(self, sample: dict) -> Optional[dict]:
        features_str = self._format_sample(sample)
        user_prompt = (
            "请根据以下灾情特征生成一道评估题目及标准答案：\n\n"
            f"{features_str}\n\n"
            "请严格按 XML 格式输出标准答案。"
        )
        resp = await self.client.chat.completions.create(
            model=self.cfg["model"],
            temperature=0.3,
            max_tokens=1500,
            timeout=float(self.cfg["timeout_s"]),
            messages=[
                {"role": "system", "content": TEST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if resp is None:
            raise ValueError("api returned None response")
        if not getattr(resp, "choices", None):
            raise ValueError("api returned no choices")

        if getattr(resp, "usage", None):
            self._token_usage["prompt"] += resp.usage.prompt_tokens or 0
            self._token_usage["completion"] += resp.usage.completion_tokens or 0
        self._token_usage["calls"] += 1

        text = resp.choices[0].message.content or ""
        # 解析 XML
        parsed = self._parse_xml(text)
        if parsed is None:
            raise ValueError(f"parse failed: {text[:200]}")
        return parsed

    def _parse_xml(self, text: str) -> Optional[dict]:
        """解析 XML 标签格式的测试题输出"""
        import re
        def _extract(tag):
            m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
            return m.group(1).strip() if m else None

        q = _extract("question")
        reasoning = _extract("reasoning")
        score_str = _extract("score")
        level = _extract("risk_level")
        conf_str = _extract("confidence")

        if not all([q, reasoning, score_str, level, conf_str]):
            return None

        try:
            score = float(score_str)
            confidence = float(conf_str)
        except ValueError:
            return None

        return {
            "question": q,
            "reasoning": reasoning,
            "score": round(score, 2),
            "risk_level": level,
            "confidence": round(confidence, 3),
        }

    async def generate_one(self, sample: dict) -> dict:
        assert self.semaphore is not None
        async with self.semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(int(self.cfg["max_retries"])),
                    wait=wait_random_exponential(multiplier=1, max=20),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        result = await self._call_once(sample)
                return {
                    "sample_id": sample.get("sample_id"),
                    "input": sample,
                    **result,
                    "teacher": self.cfg["model"],
                }
            except Exception as e:  # noqa: BLE001
                log.warning("test generation failed sid=%s err=%s", sample.get("sample_id"), e)
                return {"sample_id": sample.get("sample_id"), "input": sample, "error": str(e)}

    async def generate_from_file(
        self,
        input_path: Path,
        output_path: Path,
        n: int = 100,
    ) -> dict:
        """从输入文件读取样本，生成 n 道测试题"""
        import asyncio
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(int(self.cfg["max_concurrent"]))

        done = self._load_done_ids(output_path)
        # 读取样本，限制数量
        all_samples = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("sample_id") not in done:
                    all_samples.append(obj)
                if len(all_samples) >= n:
                    break

        if not all_samples:
            return {"generated": 0, "note": "所有样本已生成或文件为空"}

        out_f = open(output_path, "a", encoding="utf-8")
        t0 = time.time()

        try:
            async def _go(sample):
                res = await self.generate_one(sample)
                line = json.dumps(res, ensure_ascii=False) + "\n"
                out_f.write(line)
                out_f.flush()
                return res

            tasks = [asyncio.create_task(_go(s)) for s in all_samples]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - t0
        finally:
            out_f.close()

        failed = sum(1 for r in results if "error" in r)
        return {
            "generated": len(results) - failed,
            "failed": failed,
            "elapsed_s": round(elapsed, 2),
            "tokens": self._token_usage,
        }
