#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible deterministic transport smoke probe."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, Any, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    models_status, models, models_s = request_json(f"{args.base_url}/v1/models", None, args.timeout)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: SGLang smoke passed"}],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    chat_status, chat, chat_s = request_json(
        f"{args.base_url}/v1/chat/completions", payload, args.timeout
    )
    choices = chat.get("choices", []) if isinstance(chat, dict) else []
    content = ""
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content") or ""
    passed = models_status == 200 and chat_status == 200 and bool(content.strip())
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "models_status": models_status,
        "chat_status": chat_status,
        "models_elapsed_s": models_s,
        "chat_elapsed_s": chat_s,
        "nonempty_content": bool(content.strip()),
        "passed": passed,
        "models_response": models,
        "chat_response": chat,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
