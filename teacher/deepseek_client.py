"""真实 DeepSeek-V4 异步标注客户端。

特性（与方案 §3.3 工程特性对齐）：
- 异步并发（asyncio.Semaphore 控制 max_concurrent）
- 指数退避重试（tenacity）
- 断点续标（基于 sample_id 去重，输出文件已有的不再请求）
- 失败队列（独立 jsonl 落盘，便于二次重试）
- 立即落盘（每条标注完成立即追加，进程崩溃可恢复）

使用方法：
    from teacher.deepseek_client import DeepSeekLabeler
    labeler = DeepSeekLabeler(cfg)
    await labeler.label_dataset(samples, output_path, failed_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from teacher.parser import parse
from teacher.prompt_templates import SYSTEM_PROMPT, build

log = logging.getLogger("deepseek_client")


class DeepSeekLabeler:
    def __init__(self, cfg: dict):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set; copy .env.example → .env and fill it in")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", cfg["teacher"]["base_url"])
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.cfg = cfg["teacher"]
        self.semaphore = asyncio.Semaphore(int(self.cfg["max_concurrent"]))
        self._token_usage = {"prompt": 0, "completion": 0, "calls": 0}

    @staticmethod
    def _load_done_ids(path: Path) -> Set[str]:
        if not path.exists():
            return set()
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id") or obj.get("input", {}).get("sample_id")
                    if sid:
                        ids.add(sid)
                except json.JSONDecodeError:
                    continue
        return ids

    async def _call_once(self, sample: dict, prompt_version: str) -> Optional[dict]:
        prompt = build(sample, prompt_version)
        resp = await self.client.chat.completions.create(
            model=self.cfg["model"],
            temperature=float(self.cfg["temperature"]),
            max_tokens=int(self.cfg["max_tokens"]),
            timeout=float(self.cfg["timeout_s"]),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        # token 统计
        if getattr(resp, "usage", None):
            self._token_usage["prompt"] += resp.usage.prompt_tokens or 0
            self._token_usage["completion"] += resp.usage.completion_tokens or 0
        self._token_usage["calls"] += 1

        text = resp.choices[0].message.content or ""
        parsed = parse(text)
        if parsed is None:
            raise ValueError(f"parse failed: {text[:200]}")
        return parsed

    async def label_one(self, sample: dict, prompt_version: str) -> Optional[dict]:
        async with self.semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(int(self.cfg["max_retries"])),
                    wait=wait_random_exponential(multiplier=1, max=20),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        parsed = await self._call_once(sample, prompt_version)
                return {
                    "sample_id": sample.get("sample_id"),
                    "input": sample,
                    **parsed,
                    "teacher": self.cfg["model"],
                    "prompt_version": prompt_version,
                }
            except Exception as e:  # noqa: BLE001
                log.warning("label failed sid=%s err=%s", sample.get("sample_id"), e)
                return {"sample_id": sample.get("sample_id"), "input": sample, "error": str(e)}

    async def label_dataset(
        self,
        samples: Iterable[dict],
        output_path: Path,
        failed_path: Path,
        prompt_version: str = "v1",
    ) -> dict:
        """标注整个数据集，断点续标 + 实时落盘 + 失败分流。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.parent.mkdir(parents=True, exist_ok=True)

        done = self._load_done_ids(output_path)
        pending: List[dict] = [s for s in samples if s.get("sample_id") not in done]
        log.info("done=%d pending=%d", len(done), len(pending))

        if not pending:
            return {"labeled": len(done), "failed": 0, "skipped": 0, "tokens": self._token_usage}

        out_f = open(output_path, "a", encoding="utf-8")
        fail_f = open(failed_path, "a", encoding="utf-8")

        async def _go(sample):
            res = await self.label_one(sample, prompt_version)
            line = json.dumps(res, ensure_ascii=False) + "\n"
            if "error" in res:
                fail_f.write(line)
                fail_f.flush()
            else:
                out_f.write(line)
                out_f.flush()
            return res

        try:
            t0 = time.time()
            tasks = [asyncio.create_task(_go(s)) for s in pending]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - t0
        finally:
            out_f.close()
            fail_f.close()

        failed = sum(1 for r in results if "error" in r)
        return {
            "labeled": len(results) - failed,
            "failed": failed,
            "skipped": len(done),
            "elapsed_s": round(elapsed, 2),
            "tokens": self._token_usage,
        }
