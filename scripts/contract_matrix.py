#!/usr/bin/env python3
"""EXP-P02 · token 契约矩阵:什么样的"同一段文本"才真正共享前缀。

解决什么问题:radix tree 的 key 是 token id 序列(+cache_salt/extra_key),
不是文本——本矩阵逐格验证哪些"看起来相同"的请求真的逐 token 相同、能命中。
每格协议:flush_cache(确认 success)→ 发 A → 发 B → 记 B 的 cached_tokens。
格子(A=第一发,B=第二发;预期为跑前锁定的预注册值):
  base_messages   : 相同 messages 双发                    预期 hit ≈ n-1
  thinking_flip   : A 默认(thinking on),B enable_thinking=False  预期自 system 段分叉,hit ≪ base
  input_ids_direct: 相同 input_ids 双发(绕过模板)        预期 hit = n-1
  salt_same       : 相同 messages + 同 cache_salt          预期 hit ≈ n-1
  salt_diff       : 相同 messages,A salt=X B salt=Y       预期 hit = 0(命名空间隔离)

实测(EXP-P02):五格中四格符合预注册;thinking_flip 被证伪——Qwen3 的
enable_thinking 开关是纯尾扩展(只增删模板尾部,不动前缀),B_cached=1326/1329
≈ 全命中,而非预期的自 system 段分叉。预注册文本按史料保留,不改写。

机制依据(docs/theory/01 §2.1):RadixKey = token ids + extra_key + cache_salt;
salt 不同 → child_key 不同 → 硬 miss,树节点完全不共享。

面试点:①cache_salt 说明前缀缓存有安全语义——命中时延差可作侧信道探测他人
prompt,salt 是官方隔离开关,salt_diff 格即验证该隔离确实为硬 miss;②"同文本
不等于同 token":模板渲染参数(thinking 开关等)也编进 token 流,契约必须在
token 层验证,这正是收益实验全部改用 input_ids 直传的依据。
"""
import argparse, json, time, urllib.request

# 119 条编号清单,渲染后 ~1329 token:够长使模板头噪声占比可忽略;
# "逐条复述"给 temperature=0 的确定性比对一个强约束输出。
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
    time.sleep(0.5)   # 给 server 一拍完成清树,防 A 请求与 flush 竞态污染格子
    return body

def payload(model, text=BASE_TEXT, ctk=None, salt=None, input_ids=None):
    p = {"model": model, "temperature": 0.0, "max_tokens": 8, "stream": False}
    if input_ids is not None:
        # input_ids 直传时 messages 仍需带上:schema 合法性占位,server 端 input_ids 优先。
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
    # 客户端本地渲染同一模板:input_ids_direct 格的 ids 与 server 端渲染结果同源。
    ids = tok.apply_chat_template([{"role": "user", "content": BASE_TEXT}],
                                  tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):                      # transformers 5.x 返回 BatchEncoding
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):          # batch 形态时取首行
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
        fl = flush(a.base_url)   # 每格独立 flush:格与格之间零残留,B 的命中只可能来自 A
        ra, rb = call(a.base_url, pa), call(a.base_url, pb)
        out["cells"][name] = {
            # flush 响应为 JSON 时记 success 布尔,异常形态截断存原文供事后核对。
            "flush": json.loads(fl).get("success") if fl.startswith("{") else fl[:60],
            "A_prompt_tokens": ra["prompt_tokens"], "A_cached": ra["cached_tokens"],
            "B_prompt_tokens": rb["prompt_tokens"], "B_cached": rb["cached_tokens"],
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
