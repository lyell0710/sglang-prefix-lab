---
topic: SGLang RadixAttention / 前缀缓存的机制
status: 源码级完成(file:line 锚 /root/repos/sglang-v0.5.18/python/sglang/srt)
verified_against: 本仓 EXP-S02(cached_tokens 实测)——待跑
---

# 01 · RadixAttention:前缀缓存怎么把"重复的前缀只算一次"

## 1. 一句话结论

服务端把每个请求的 **token id 序列**(不是字符串)插进一棵 **radix tree**,节点的
`value` 是这段 token 对应的 **KV cache 物理块索引**。新请求进来先在树上做最长前缀
匹配(page 对齐),匹配到的部分**直接复用已有 KV,跳过 prefill**——省下的正是
"共享前缀"那段的 attention 计算。这就是 cache-aware 路由值得做的前提:把同前缀的
请求送到**已经持有那段 KV 的那张卡**,命中率最大化。

## 2. 机制(自己的话 + 关键 file:line)

### 2.1 树与节点
- `RadixKey`(radix_cache.py:59)= token ids **+ `extra_key`(LoRA id 等)+ `cache_salt`**。
  salt/extra_key 不同 → `child_key`(:217-229)落到不同子键 → **硬 miss,不共享节点**。
  → 实验含义:要让两个请求共享前缀,不只是文本相同,连 LoRA/salt/chat template
  渲染出的 token 都必须逐 token 相同(见 §5 与 theory/03)。
- `TreeNode`(:238):`children`、`value`=KV 索引张量、**`lock_ref`**(在用计数)、
  `last_access_time`(LRU 依据)。`value is None` 即"已被逐出"(:268-270)。
- 根节点建成时 `lock_ref=1`(:359),永不可逐出。

### 2.2 匹配(为什么 page 对齐、为什么至少留 1 token)
- `match_prefix`(:376-434):先把 query **page 对齐**(:419,`page_aligned` :150-154),
  再 `_match_prefix_helper`(:678-702)沿树下行;匹配落在节点中段时**分裂节点**
  (`_split_node` :704-727)。匹配到的 token 数 = 拼接各节点 value 的长度(:426)。
- 调度侧入口 `Req.init_next_round_input`(schedule_batch.py:1292,匹配 :1353-1364)把
  匹配长度**上限压到 input_len-1**(`_compute_max_prefix_len` :1411-1416)——**永远至少
  重算一个 token**,否则没有 logits 可采样。→ 这解释了实验里 cached_tokens 最多
  = prompt_len-1,不会等于 prompt_len。

### 2.3 写回与在途保护(lock_ref 的意义)
- 请求结束 `cache_finished_req`(:458-513):把 `input+output` 的 token(截到 KV 长度)
  插树,释放重复区间,再 `dec_lock_ref`(:512)。
- **chunked prefill 中途**也插:`cache_unfinished_req`(:515-583)插完**重新匹配复用**
  (:550)、改写 `req_to_token`、`dec_lock_ref(old)+inc_lock_ref(new)`。
- `inc/dec_lock_ref`(:622-656):0→1 时把这段 token 从 `evictable` 移到 `protected`,
  反之亦然。**在途请求正在用的前缀被锁住,逐出扫描跳过它**(`_update_leaf_status`
  :821-824)。→ 这是"边生成边缓存"不出错的关键:别人不能把我正在读的 KV 逐出。

### 2.4 逐出(LRU 默认,可换)
- `evict`(:592-620):对 `evictable_leaves` 按 `eviction_strategy.get_priority` 建**最小堆**,
  弹叶子、放它的 KV、删节点、父变叶再入堆。
- 策略由 `--radix-eviction-policy`(server_args.py:919-931)选,默认 **lru**
  (evict_policy.py:16-18,按 `last_access_time`);另有 lfu/fifo/mru/priority。
- **需求驱动**:只逐出缺口那么多(common.py:114-138),不是一次清空。

### 2.5 什么时候根本不建 radix
- `--disable-radix-cache` **且** chunked prefill 开 → 用 `ChunkCache`(registry.py:91-95);
  否则建 `RadixCache`(:165-167)。→ 我们的反例臂(证明收益确来自前缀复用)可以用
  `--disable-radix-cache` 做 A/B 对照。

## 3. 本项目实证(待跑,EXP-S02/S04)

- EXP-S02 正确性/契约:构造 byte 级相同前缀的 manifest,单 worker 开 `--enable-cache-report`,
  验 `usage.prompt_tokens_details.cached_tokens`(见 §4)= 预期共享前缀长度(±page)。
- EXP-S04:`sglang:cache_hit_rate` 与 per-worker `cached_tokens_total` 随路由策略变化。
- 反例臂:`--disable-radix-cache` 时命中率应 ≈0,cache-aware 相对 round-robin 的 TTFT 收益消失。
  *(数字待本仓 raw;此处只写机制,不写未测数字——CORE 铁律 6。)*

## 4. 怎么"看见"命中(观测点,写实验必须知道)

- **响应里**:`meta_info.cached_tokens` 恒设(tokenizer_manager.py:2287-2292);
  OpenAI `usage.prompt_tokens_details.cached_tokens` **仅 `--enable-cache-report`**
  (usage_processor.py:39-45)。
- **Prometheus**(仅 `--enable-metrics`,/metrics):`sglang:cache_hit_rate`(Gauge,
  metrics_collector.py:292-297,算式 metrics_reporter.py:646-656)、
  `sglang:cached_tokens_total{cache_source=device|host|...}`、
  `sglang:prefill_effective_tokens_total{mode=input|device_hit|...}`、
  `sglang:kv_evictable_tokens` / `kv_used_tokens` / `token_usage`、
  `sglang:evicted_tokens_total`。
- **每请求时序**(仅 `--enable-metrics`):`e2e_latency`(恒设)、`queue_time`、
  `forward_entry_time`、`prefill_finished_time`(req_time_stats.py:1167-1182)——
  这批字段让我们把 TTFT 拆成"排队 vs prefill",做 S05 归因。
- `/flush_cache`(http_server.py:966-981):**仅在 idle 时**真的清(scheduler.py:4251-4279),
  有 pending 请求会 warn 且 success=false。→ 每个 A/B arm 冷启动 = flush 且确认成功后开始。

## 5. 面试追问 Q&A

- **Q:cache-aware 路由到底缓存的是文本还是 token?** 服务端 radix 缓存的是 **token id**
  (RadixKey);但 sgl-model-gateway 的 `cache_aware` 路由器**近似树用的是原始字符**
  (为省 tokenize,router tree.rs 注释自述),两者是**两棵不同的树**——engine 侧决定
  "算不算得省",router 侧决定"往哪张卡送"。见 theory/02。
- **Q:为什么共享前缀却没命中?** 逐 token 不一致:chat template 的 BOS/role token、
  `enable_thinking` 开关改了 system 段、`cache_salt`/LoRA id 不同,任一处第一个 diff
  token 起就分叉。用 `request.input_ids` 直传可绕过模板(serving_chat.py:1123-1134)。
- **Q:在途请求的 KV 会被别人逐出吗?** 不会,lock_ref>0 的前缀在 protected 段,
  逐出扫描跳过(:821-824)。

## 6. 延伸(源码锚)

radix_cache.py(树/匹配/逐出/锁)、schedule_policy.py(lpm 调度与 prefix 排序)、
metrics_collector.py(命中率指标)、usage_processor.py(cached_tokens 出口)。
对照:vllm/experiments 的 automatic prefix caching(同思想不同实现),
Kernel_Optimazation flash-attn(KV 布局的算子侧)。
