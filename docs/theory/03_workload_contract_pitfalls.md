---
topic: 共享前缀实验的工作负载契约 —— "看起来相同"到"token 级相同"之间的坑
status: 源码级完成;实测锚待 EXP-P02
---

# 03 · 为什么"同一段文本"不等于"同一段前缀"

## 1. 一句话结论

radix 命中判定发生在 **token id 序列**上(theory/01 §2.1),而 OpenAI 接口收到的是
**消息列表**——中间隔着 chat template 渲染、thinking 开关、salt/LoRA 命名空间三层变换,
任何一层不一致,第一处 diff token 起全部 miss。**实验的第一个 gate 不是测速度,是证明
两臂请求在 token 级逐位相同**(manifest 固化 + tokenize 校验)。

## 2. 三层变换(file:line)

1. **chat template**:`/v1/chat/completions` 走 `tokenizer.apply_chat_template(...,
   add_generation_prompt=True)`(serving_chat.py:1335-1342)再 encode(:1343-1345)。
   role 标记、system 头、`<|im_start|>` 等都进 token 流——**同样的 user 文本,包一层
   messages 后前缀就变了**。
- **绕过法**:请求直接给 `input_ids`(serving_chat.py:1123-1134 完全跳过模板)——
   本仓 manifest 的正式形态(byte 级可固化、可 sha256)。
2. **thinking 开关**(Qwen3):模板含 `enable_thinking` 切换(protocol.py:958-971,
   qwen3 默认开,template_detection.py:188-208);`--default-chat-template-kwargs` 与
   每请求 `chat_template_kwargs` 合并(serving_chat.py:1053-1060,请求侧优先)。
   **实测修正(EXP-P02,预注册假设被证伪)**:Qwen3 模板的开关落在 generation
   prompt 尾部——thinking-off = thinking-on 的完整渲染 **原样 + 追加**
   `<think>\n\n</think>\n\n`(1325→1329 token,首分叉位=1325 即无分叉,raw=
   EXP-P02/20260824T163438_template_divergence.json)。因此 **thinking 配置不一致
   并不破坏前缀共享**(命中 1326/1329);"改 system 段导致全 miss"是错误推断。
   顺带实测:radix 树缓存的是 input+output **全序列**(cache_finished_req 插
   input+output,theory/01 §2.3),thinking-off 请求的 `<think>` 尾 token 恰与
   前一请求的首个输出 token 相同,命中因此多 1(1326=1325+1)。
3. **命名空间**:`cache_salt`(protocol.py:867)与 LoRA `extra_key` 进 RadixKey
   (schedule_batch.py:922-929;radix_cache.py:227-229)→ 不同 salt 天然隔离,
   同 salt 才可能共享。(可反过来用:用 salt 做"人为 miss"的对照臂。)

## 3. 计时口径(报告里每个时延数字的定义)

- client 侧(bench_serving):TTFT=首个含文本 chunk − 发出;E2E=末 chunk − 发出;
  TPOT=(E2E−TTFT)/(out−1);ITL=chunk 间隔均摊(serving.py:747-767,:1116-1127)。
- server 侧(`--enable-metrics` 才有):`queue_time`、`forward_entry_time`、
  `prefill_finished_time`、`e2e_latency`(req_time_stats.py:1167-1182,:479-480)。
  → TTFT ≈ 排队 + prefill + 流式开销;归因实验(EXP-P05)靠 server 侧字段拆分,
  **不要**拿 client TTFT 直接说"prefill 变快了"。
- 预热与冷启动:JIT/CUDA graph 首跑污染第一批请求;`/flush_cache` 只在 idle 真清
  (scheduler.py:4251-4279)→ 每臂开始前:确认 flush success=true + 固定 warmup。

## 4. page 对齐与"最多 n-1"两个必然折扣

- 匹配 page 对齐截断(radix_cache.py:150-154,:210):`page_size=1` 时无损;>1 时命中
  长度是 page 的整数倍。
- 至少重算 1 token(schedule_policy 上限 input_len-1):cached_tokens ≤ prompt-1。
  → EXP-P02 的判定阈值要按这两条折扣建模,不设"命中=prompt 长度"这种会假 FAIL 的 gate。

## 5. 面试追问 Q&A

- **Q:怎么证明两臂负载完全相同?** manifest JSONL 固化(内容+顺序+seed),两臂读同一
  文件,记录 sha256 进 provenance;tokenize 校验列(prefix_len/total_len)在生成时写死。
- **Q:为什么用 input_ids 直传而不是 messages?** 消除模板/思考开关两层变数,让"共享
  前缀长度"成为实验的自变量而不是被渲染出来的因变量。代价:不覆盖"真实 chat 流量"
  ——所以留一个 messages 形态的 probe 臂验证结论迁移。
- **Q:0 输出长度能测 TTFT 吗?** max_tokens≥1,且流式下首 chunk 才停表;输出固定 32
  保证 TPOT 也有定义(与 sibling 仓协议一致,便于交叉核对)。
