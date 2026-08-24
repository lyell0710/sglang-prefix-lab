#!/usr/bin/env python3
"""EXP-P02 · token 契约矩阵:什么样的"同一段文本"才真正共享前缀。

每格协议:flush_cache(确认 success)→ 发 A → 发 B → 记 B 的 cached_tokens。
格子(A=第一发,B=第二发):
  base_messages   : 相同 messages 双发                    预期 hit ≈ n-1
  thinking_flip   : A 默认(thinking on),B enable_thinking=False  预期自 system 段分叉,hit ≪ base
  input_ids_direct: 相同 input_ids 双发(绕过模板)        预期 hit = n-1
  salt_same       : 相同 messages + 同 cache_salt          预期 hit ≈ n-1
  salt_diff       : 相同 messages,A salt=X B salt=Y       预期 hit = 0(命名空间隔离)
"""
import argparse, json, time, urllib.request

BASE_TEXT = "请逐条复述以下清单,不要添加内容:" + "".join(
    f"第{i}条,内容编号{i * 7};" for i in range(1, 120))

def call(base, payload, timeout=120):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    det = (d.get("usage", {}) or {}).get("prompt_tokens_details") or {}
    return {"prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens"),
            "content": d["choices"][0]["message"]["content"]}

def flush(base):
    req = urllib.request.Request(base + "/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
    time.sleep(0.5)
    return body

def payload(model, text=BASE_TEXT, ctk=None, salt=None, input_ids=None):
    p = {"model": model, "temperature": 0.0, "max_tokens": 8, "stream": False}
    if input_ids is not None:
        p["messages"] = [{"role": "user", "content": text}]
        p["input_ids"] = input_ids
    else:
        p["messages"] = [{"role": "user", "content": text}]
    if ctk is not None:
        p["chat_template_kwargs"] = ctk
    if salt is not None:
        p["cache_salt"] = salt
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokenizer-dir", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    ids = tok.apply_chat_template([{"role": "user", "content": BASE_TEXT}],
                                  tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):                      # transformers 5.x BatchEncoding
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    cells = {
        "base_messages":    (payload(a.model), payload(a.model)),
        "thinking_flip":    (payload(a.model),
                             payload(a.model, ctk={"enable_thinking": False})),
        "input_ids_direct": (payload(a.model, input_ids=ids),
                             payload(a.model, input_ids=ids)),
        "salt_same":        (payload(a.model, salt="saltX"),
                             payload(a.model, salt="saltX")),
        "salt_diff":        (payload(a.model, salt="saltX"),
                             payload(a.model, salt="saltY")),
    }
    out = {"template_rendered_len": len(ids), "cells": {}}
    for name, (pa, pb) in cells.items():
        fl = flush(a.base_url)
        ra, rb = call(a.base_url, pa), call(a.base_url, pb)
        out["cells"][name] = {
            "flush": json.loads(fl).get("success") if fl.startswith("{") else fl[:60],
            "A_prompt_tokens": ra["prompt_tokens"], "A_cached": ra["cached_tokens"],
            "B_prompt_tokens": rb["prompt_tokens"], "B_cached": rb["cached_tokens"],
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
