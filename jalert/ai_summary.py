"""Optional LLM step: Chinese TL;DR for the top entries.

Pure standard library. Credentials come from the environment only (never from
config.json, which is committed to git):

    JALERT_LLM_API_KEY   required - DeepSeek key, or any OpenAI-compatible key
    JALERT_LLM_BASE_URL  default https://api.deepseek.com
    JALERT_LLM_MODEL     default deepseek-chat

Any failure degrades silently: the entry simply gets no Chinese summary and
the pipeline continues as if this module did not exist.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger("jalert")

SYSTEM_PROMPT = (
    "你是消化内科研究生的文献助手。用简体中文总结医学论文，"
    "只基于用户给出的标题和摘要，严禁编造原文没有的内容。"
)


def _prompt(title: str, abstract: str) -> str:
    return (
        "请用中文总结下面这篇论文：\n"
        f"标题：{title}\n"
        f"摘要：{abstract}\n\n"
        "要求：\n"
        "1. 第一行给出中文标题翻译（不超过40字）；\n"
        "2. 接着用2-3句话概括：研究做了什么、关键结果是什么；\n"
        "3. 总字数控制在150字以内。"
    )


def _call_llm(title: str, abstract: str, api_key: str, base_url: str, model: str,
              timeout: int = 40) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(title, abstract)},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def summarize_top_entries(entries: list[dict], max_items: int = 10) -> int:
    """Attach ``entry["ai_zh"]`` for the first ``max_items`` entries.

    Returns how many entries got a summary. Never raises: without a key it
    returns 0 immediately, and per-entry failures are logged and skipped.
    """
    api_key = os.environ.get("JALERT_LLM_API_KEY", "").strip()
    if not api_key:
        return 0
    base_url = os.environ.get("JALERT_LLM_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("JALERT_LLM_MODEL", "deepseek-chat").strip()
    done = 0
    for entry in entries[:max_items]:
        item = entry["item"]
        abstract = (item.abstract or "").strip()
        if not abstract:
            continue
        try:
            entry["ai_zh"] = _call_llm(item.title, abstract, api_key, base_url, model)
            done += 1
            log.info("ai 摘要 %d/%d: %s", done, max_items, item.title[:40])
        except Exception as exc:  # noqa: BLE001 - degrade, never break the run
            log.warning("ai 摘要失败（已跳过）: %s: %s", item.title[:40], str(exc)[:100])
    return done
