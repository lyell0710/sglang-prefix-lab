# 讲义 01 · RadixAttention 前缀缓存全景:从 token 契约到重用距离

> 读者:准备校招面试的作者本人,以及第一次接触前缀缓存的工程师。
> 读法:不跳步。每个论断后面跟着它的证据锚(EXP 编号 / 文件:行号 / raw 路径),
> 所有数字与仓内现行口径逐字一致,来源见 records/ 与 data/derived/。

## 1. 这一篇回答什么问题

前缀缓存(SGLang 的 RadixAttention)到底缓存了什么、命中判定发生在哪一层、
省下的毫秒从哪里来、什么时候失效。读完你应当能:①在白板上手推"命中上限为什么
是 input_len−1"与"LRU 命中 ⇔ 池 ≥ 重用距离"两条结论;②解释为什么"同一段文本"
不等于"同一段前缀",并完整复盘一次预注册假设被证伪的全过程(EXP-P02);③答上
"你的 −77% 怎么测的、凭什么归因给前缀复用"这类追问——包括 466,944 逐 token
三方相等是怎么做到的(EXP-P03/P07)。

## 2. 直觉与第一性原理

**没有前缀缓存的世界**:Transformer 解码是自回归的,生成第 $t$ 个 token 需要前面
所有 token 的 Key/Value 张量(KV)。一条请求进来,引擎先对全部输入 token 做一次
前向计算(prefill)把 KV 算出来,然后逐 token 解码。现在考虑真实 serving 流量:
几百个请求共享同一个 2000 token 的系统提示词。没有缓存时,这段完全相同的前缀
会被逐请求重算几百遍——每一遍的计算结果(KV)逐位相同。计算是幂等的,重复
计算纯属浪费,而且浪费的正是 TTFT(首 token 延迟)里最大的一块:prefill。

**为什么"相同前缀"可以复用而"相同后缀"不行**:causal attention 下,位置 $i$ 的
KV 只依赖 token $0..i$。所以两条请求只要前 $k$ 个 token 逐位相同,它们前 $k$ 个
位置的 KV 就逐位相同——这是复用的数学前提,也是它的硬边界:第一个不同的 token
之后,后面所有位置的 KV 全部不同(哪怕后缀文本一样),一位都不能复用。

**日常类比与失效点**:像连锁咖啡店把"每天都要做的糖浆底"提前熬好一大锅,
每杯只现做上面的部分。类比在两处失效:①糖浆差不多就行,KV 复用要求 token
序列**逐位相同**,"差一个字"就整段作废(见 §3.2 的三层变换陷阱);②糖浆锅
容量不够时可以少熬点,KV 池不够时 LRU 逐出会让命中率**阶跃归零**而不是按比例
下降(见 §3.5 的重用距离模型,EXP-P05 实测)。

**RadixAttention 的选型直觉**:要复用就要能查"新请求和历史请求的最长公共前缀
是谁、它的 KV 在哪"。把历史 token 序列组织成 radix tree(压缩前缀树),节点的
value 存 KV 物理块索引,最长前缀匹配即一次树下行——这正是 radix tree 的看家
操作。哈希整段 prompt 做 key 做不到"部分命中",逐 token 的 trie 空间开销大,
radix tree(边上压缩一段 token)是二者的折中。

## 3. 完整推导与机制

### 3.1 树结构与 token 级契约,以及 n−1 上限的推导

服务端为每个请求构造 `RadixKey`(radix_cache.py:59):**token id 序列 +
extra_key(LoRA id 等)+ cache_salt**。树下行用 `child_key`(radix_cache.py:217)
做字典键,salt/extra_key 直接编进键——所以不同 salt 的两个请求即使 token 逐位
相同也是**硬 miss**(命名空间隔离,EXP-P02 的 salt_diff 格实测 B 发 cached 缺失
= 0)。节点 `TreeNode`(radix_cache.py:238)持有 `value`(该段 token 对应的 KV
块索引张量)、`lock_ref`(在用计数,>0 时禁止逐出)、`last_access_time`(LRU
依据)。匹配入口 `match_prefix`(radix_cache.py:376)先做 page 对齐
(page_size=1 时无损),再沿树下行,命中落在节点中段时把节点分裂
(`_split_node`,radix_cache.py:704)以暴露精确边界。

**n−1 上限,逐步推**(每步一行"为什么"):

1. 解码第 1 个输出 token,需要"输入最后一个 token 位置"的 logits。
   ——采样定义如此:logits 是下一 token 的分布。
2. logits 来自当次 forward 对该位置的 hidden state 经 lm_head 投影。
   ——缓存里存的是 KV,不存 logits,也不存 hidden state。
3. 若输入全部 $n$ 个 token 的 KV 都来自缓存、本次 forward 一个位置都不算,
   则最后位置的 hidden state 不存在,没有 logits 可采样。
   ——复用跳过的是计算,被跳过的位置不产生任何本次前向的中间量。
4. 所以调度器必须强制至少重算 1 个 token:匹配上限 = input_len − 1
   (`_compute_max_prefix_len`,schedule_batch.py:1411-1416,§4 走读第 6 段)。

**实测锚**:EXP-P01 同一请求第二发 `cached_tokens=1324, prompt_tokens=1325`
(raw=data/raw/EXP-P01/20260824T162947_probe_cached.json),恰为 $n-1$ 的
**精确值**而非近似;engine 侧 `cache_hit_rate=0.9992`(= 1324/1325)。这是
"从源码读出上限 → 测量精确落在上限上"的最小闭环样本。

### 3.2 chat template 三层变换陷阱:thinking 证伪案例全程复盘

radix 命中判定发生在 token id 层,而 OpenAI 接口收到的是消息列表,中间隔着
三层变换(theory/03),任何一层不一致,第一处 diff token 起全部 miss:

1. **chat template 渲染**:`/v1/chat/completions` 把 messages 走
   `apply_chat_template(..., add_generation_prompt=True)` 再 encode
   (serving_chat.py:1335-1345)。role 标记、system 头都进 token 流。
   绕过法:请求直接给 `input_ids`,模板整段跳过(serving_chat.py:1125-1134)
   ——本仓收益实验的正式形态。
2. **thinking 开关**(Qwen3):`enable_thinking` 经 `chat_template_kwargs`
   进入模板(protocol.py:958-971 把 thinking 布尔写进 ctk;请求侧与服务端默认
   合并见 serving_chat.py:1053-1060)。
3. **命名空间**:`cache_salt`(protocol.py:867)与 LoRA extra_key
   (schedule_batch.py:928-929)进 RadixKey。

第 2 层是本仓最完整的一次**预注册证伪**,按时间序复盘(EXP-P02):

- **预注册**(跑前锁定,写在 docs/PLAN.md#exp-p02 与脚本 docstring):
  "thinking 开关不一致 → 模板从 system 段分叉 → hit ≪ base"。依据是想当然的
  推断:开关是模板参数,模板参数应该改模板头部。
- **实测**:thinking_flip 格 B 发 `cached=1326 / prompt=1329`
  (raw=data/raw/EXP-P02/20260824T163438_contract_matrix.json)——接近全命中,
  与"hit ≪ base"直接矛盾。**假设证伪。**
- **CPU 复核**(不碰 GPU,纯 tokenizer 渲染对比):thinking-on 渲染 1325 token,
  thinking-off 渲染 1329 token,**首分叉位 = 1325**,off 尾部多出
  `<think>\n\n</think>\n\n` 恰 4 个 token
  (raw=data/raw/EXP-P02/20260824T163438_template_divergence.json)。
  即:off = on 的完整渲染**原样 + 尾部追加**,前缀共享完好。
- **修正**:theory/01 §5 与 theory/03 §2 当场改写;结论限定 Qwen3 模板族。
- **意外收获**:1326 = 1325 + 1,多出的 1 个 token 是 A 请求的**首个输出 token**
  (`<think>`)——因为树在请求结束时插入的是 input+output **全序列**
  (`cache_finished_req`,radix_cache.py:458-513,`token_ids =
  (req.origin_input_ids + req.output_ids)`),B 的第 1326 个 token 恰好咬上。
  一个 off-by-one 现象反向证实了一条独立机制。

方法论提炼:错误的预注册假设 + 忠实测量 + 一次廉价的分层复核(CPU 渲染),
比"猜对了"教得更多。这也是为什么收益实验(P03 起)全部改用 input_ids 直传:
让"共享前缀长度"成为实验的自变量,而不是被模板渲染出来的因变量。

### 3.3 收益账:省掉的 prefill 值多少毫秒

**算式**(把 TTFT 拆开):

$$\mathrm{TTFT} \approx C + S_{\mathrm{prefill}}(n_{\mathrm{recompute}}) + W_{\mathrm{queue}}$$

其中 $C$ 是与 prefill 无关的固定地板(请求解析、调度、首 token 解码、流式
开销),$S_{\mathrm{prefill}}$ 随需重算 token 数近似线性(每 token 的 FLOP
≈ $2 P$,$P$ 为参数量,attention 项另计),$W_{\mathrm{queue}}$ 是排队项,
并发 1 时约为 0。命中 $k$ 个 token 后 $n_{\mathrm{recompute}} = n - k$,
理想收益即 $S_{\mathrm{prefill}}$ 按 $k/n$ 比例缩短。

**实测曲线对账**(总长 2048,扫 prefix ∈ {0,512,1024,1536,1792},3 seeds):

- Qwen3-8B(EXP-P07,data/derived/exp_p07_8b_ttft_vs_prefix.csv):并发 1 的
  TTFT p50 从 228.4±3.9 ms(prefix=0)降到 52.9±0.5 ms(prefix=1792),
  **−77%**;并发 8 从 1068.3±13.3 降到 234.5±4.6 ms,**−78%**。
  斜率账:$(228.4-52.9)/1792 \approx 98\ \mu s/\mathrm{token}$(并发 1)。
  前缀占比 1792/2048 = 87.5%,实际只省 77%——差额就是 $C$:按线性外推,
  纯 prefill 剩 $228.4 \times 256/2048 \approx 28.6$ ms,实测 52.9 ms,
  余量 ~24 ms 即固定开销与首 token 解码(EXP-P07 §6 的"余量"口径)。
- Qwen3-0.6B(EXP-P03,data/derived/exp_p03_ttft_vs_prefix.csv):同协议仅
  −36%(并发 1,26.84→17.27 ms)/ −63%(并发 8,115.14→42.73 ms)。
  0.6B 的 miss TTFT 只有 ~27 ms,里面 ~17 ms 是地板 $C$——prefill 在 TTFT
  中占比小,可被跳过的部分就小,**收益天花板正比于 prefill 占比**。
  模型量级账(推断,标注):参数量 8B/0.6B ≈ 14×,实测 miss TTFT 之比
  228.4/26.84 ≈ 8.5×,同量级;没到 14× 是因为 0.6B 的 TTFT 被 $C$ 垫高。
- **并发放大**(实测→机理推断,EXP-P03 §6):0.6B 的每命中 token 收益斜率从
  并发 1 的 5.3 µs/token 升到并发 8 的 40 µs/token(×7.6)。机理:并发下
  prefill 算力是队列瓶颈,省掉一条请求的 prefill 同时缩短了**其余请求的排队**
  ($W_{\mathrm{queue}}$ 项),收益复利。"低并发测不出缓存价值"的根源在此。

**反例臂**:`--disable-radix-cache` 起服的 off 臂在 8B 下全线 229.7-231.8 ms
持平(exp_p07 csv 的 off 行),0.6B 下 26.59-26.87 ms 持平——收益只在
radix 开启且确有共享前缀时出现,归因成立。

### 3.4 逐 token 归因闭环:466,944 三方相等怎么做到

"TTFT 降了"与"命中了前缀"之间还差一步归因:降幅是否**恰好**由省掉的 prefill
token 兑现?本仓用三个**互相独立**的计数源对账:

1. **客户端逐请求**:流式响应的 `usage.prompt_tokens_details.cached_tokens`
   (随流返回,与停表同一响应内闭合,§4 走读第 1 段)。逐请求硬校验:
   on 臂必须恰 = prefix_len,off 臂与 prefix=0 必须恰 = 0(§4 第 4 段)。
2. **引擎计数器**:`/metrics` 的
   `sglang:prefill_effective_tokens_total{mode="device_hit"}` counter,
   实测终值 **466,944.0**(raw=data/raw/EXP-P07/20260824T171520_8b_radix_on_metrics.txt,
   0.6B 的 EXP-P03 同值复现)。
3. **协议期望**(纯算术):每点 16 请求,Σ over prefix∈{512,1024,1536,1792}:
   $16 \times (512+1024+1536+1792) = 16 \times 4864 = 77{,}824$;
   × 3 seeds = 233,472;× 2 个并发臂(c1+c8)= **466,944**。prefix=0 点贡献 0。

三方逐 token 相等(0.6B/8B 双复现,EXP-P03/P07 §6)意味着:没有一个 token 的
命中是虚报的,也没有协议外的意外命中(预热残留、跨点污染都会破坏等式)。
这就是"降幅确实且仅由省掉的 prefill 兑现"这句话的证据形态。

### 3.5 重用距离模型:完整推导与三池验证

前缀缓存何时失效?KV 池是有限的,逐出策略默认 LRU
(`--radix-eviction-policy` 默认 lru,server_args.py:919-931;策略实现
evict_policy.py:16-18 按 `last_access_time` 建最小堆,radix_cache.py:592 的
`evict` 只逐出缺口那么多,common.py:114-138)。

**推导**(白板可复现,docs/talk/whiteboard_card_reuse_distance.md):

1. **定义重用距离 $D$**:同一热前缀两次被访问之间,注入缓存池的 token 总量。
   ——LRU 只看"最近",所以决定一个条目存亡的正是"两次使用之间进来了多少"。
2. 轮转访问 $H$ 个热前缀、每条请求总长 $T$、每条热请求后跟 $c$ 条冷请求:
   两次访问同一热前缀之间恰好流过 $H$ 条热请求与 $Hc$ 条冷请求,
   $$D = H \cdot T \cdot (1+c)$$
   ——每条请求(热或冷)都把约 $T$ 个 token 写进池(树缓存 input+output)。
3. **LRU 命中条件**:池容量 $P \ge D$。
   ——$P \ge D$ 时,热前缀在被重访之前不可能被挤出(比它更旧的都先走);
   $P < D$ 时,它**必然**在重访之前被逐出(进来的量已超池容量)。
4. **预测阶跃而非斜坡**:轮转访问下所有热前缀的 $D$ 相同,条件对全体同时
   成立或同时破——命中率只有 1.0 与 ~0 两个稳态,没有中间态。
   这是"循环工作集 + LRU"的经典最坏搭配,与 CPU cache 的 LRU thrashing 同构。

**实验构造**(EXP-P05,bench_evict.py):$H=4, T=2048$,故
$D = 8192 \times (1+cr)$;特意让 $H \cdot T = 8192$ = 最小池位,使 cr=0 时
$D$ 恰压在池边界上,cr 每 +1 把 $D$ 线性外推一个池位。三池
(`--max-total-tokens` 8192 / 16384 / 默认 ≈16 万,161671,EXP-P01 启动日志)
× 冷流量 cr ∈ {0,1,2,4} × 3 seeds,串行并发 1(隔离排队与 lock_ref 扰动)。

**实测**(data/derived/exp_p05_eviction_cliff.csv,seed 间 std 全为 0):

| 池(token) | cr=0(D=8192) | cr=1(16384) | cr=2(24576) | cr=4(40960) |
|---|---|---|---|---|
| 8192 | **1.0000** | **0.0625** | 0.0625 | 0.0625 |
| 16384 | — | — | — | 0.125 |
| 默认 | 1.0000 | — | — | **1.0000** |

模型三池全符合:8192 池仅 cr=0($D=P$,恰好)保命中;16384 池到 cr=4
($D=40960>16384$)崩;默认池 $P \gg D$ 全保。残余 0.0625 = 1/16,即预热后
的首个热请求(raw 里 `hot_cached` 序列是 `[1536, 0, 0, ...]`,
data/raw/EXP-P05/20260824T164650_smallpool_cr1_s20260824.json)。仅边界格
16384@cr4 见 0.125 残余(首周期多存活一拍)。`evicted_tokens_total` 佐证:
小池随 cr 单调升(33.9 万 → 66.3 万),默认池全程无逐出(counter 不曝光,
按 0 记——EXP-P05 §7)。

**工程含义**:容量规划按**热前缀重用距离**配池,不是按热集大小;冷流量占比
把 $D$ 线性推过边界,是一阶变量。这条模型还是讲义 02 里路由实验(EXP-P06)
预注册预测的推导前提。

## 4. 代码逐段走读:scripts/bench_prefix.py 的停表与闭环

以下按执行顺序走读收益曲线实验的测量端(引用为逐字拷贝,标 文件:起-止行)。

**第 1 段 · 单请求:停表与命中数在同一响应内闭合**(scripts/bench_prefix.py:43-68)

```python
async def one_request(session, url, model, ids, out_tokens):
    # messages 仅为 OpenAI schema 占位;server 端 input_ids 扩展字段优先生效。
    payload = {"model": model, "input_ids": ids,
               "messages": [{"role": "user", "content": "x"}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None; cached = None
    async with session.post(url + "/v1/chat/completions", json=payload) as r:
        async for raw in r.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            body = line[5:].strip()
            if body == "[DONE]": break
            try: d = json.loads(body)
            except json.JSONDecodeError: continue
            ch = d.get("choices") or []
            if ttft is None and ch and (ch[0].get("delta") or {}).get("content"):
                # 停表口径:首个非空 content delta(自动跳过 role-only 首 chunk)。
                ttft = (time.perf_counter() - t0) * 1e3
            u = d.get("usage")
            if u and u.get("prompt_tokens_details"):
                # 命中数随流取:与停表同一响应内闭合,归因不依赖事后指标。
                cached = u["prompt_tokens_details"].get("cached_tokens")
    e2e = (time.perf_counter() - t0) * 1e3
    return {"ttft_ms": ttft, "e2e_ms": e2e, "cached_tokens": cached}
```

角色:全仓统一的 TTFT 停表口径就定义在这 26 行里。三个关键选择:
①`input_ids` 直传(EXP-P02 契约结论)——token 序列完全受控,`cached_tokens`
才能与 prefix_len 逐 token 对账;②停表停在**首个非空 content delta** 到达
客户端,不停在 HTTP 首字节(那只是 SSE 响应头,不含 token),也不用 server
侧直方图(要的是含排队+prefill+首 token 解码的用户可感知延迟);③
`stream_options.include_usage` 让 usage 随流返回,"快了多少"与"命中了多少"
同源。改错会怎样:若停表停在首个 chunk 而不判 `delta.content`,会被 role-only
首 chunk 提前触发,TTFT 系统性偏小;若事后查 /metrics 取命中,并发下无法
归属到单个请求,闭环校验(第 4 段)就做不成。

**第 2 段 · 测量点前置:flush 与预热**(scripts/bench_prefix.py:74-91)

```python
async def run_point(args, tok_seed, prefix_len, concurrency, warm):
    # 可复现负载:prefix 只由 tok_seed 决定(点内所有请求共享同一前缀);
    # 每条后缀用独立派生 seed——彼此不同(唯一性)且跨复跑逐 token 相同。
    random.seed(tok_seed)
    prefix = build_ids(None, prefix_len) if prefix_len else []
    reqs = []
    for i in range(args.num_requests):
        random.seed(tok_seed * 1000 + i + 1)
        suffix = build_ids(None, args.total_len - prefix_len)
        reqs.append(prefix + suffix)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        fl = await flush(s, args.base_url)
        if warm and prefix_len:
            # 预热:max_tokens=1 把成本压到最低;+8 随机尾 token 使预热请求
            # 区别于计时请求(尾巴唯一、不会被后续命中),prefix 整段随请求
            # 完成插进 radix tree。
            await one_request(s, args.base_url, args.model, prefix + build_ids(None, 8), 1)
```

角色:把"树里已有前缀"设成前置条件。flush 清掉上一测量点的树(点与点零残留,
否则 prefix_lens 的扫描顺序会污染结果);预热发 1 条 [prefix + 8 随机尾] 请求
把前缀灌进树(计时外)。改错会怎样:不 flush,则后一个点会命中前一个点的残树,
cached 校验(恰=prefix_len)当场 FAIL;不预热,计时臂的首请求变成"替大家种树"
的 miss,分布被污染;预热尾巴不加随机 token,预热请求自己就会被计时请求全长
命中,cached 会超出 prefix_len,同样被校验捕获。

**第 3 段 · 并发定义与分位数**(scripts/bench_prefix.py:92-113)

```python
        sem = asyncio.Semaphore(concurrency)
        async def lim(ids):
            # 客户端信号量限流即"并发"的定义:在途请求数上限,而非 server batch。
            async with sem:
                return await one_request(s, args.base_url, args.model, ids, args.output_len)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[lim(r) for r in reqs])
        dur = time.perf_counter() - t0
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    ttfts_sorted = sorted(ttfts)
    # 简单索引分位(int(p*n) 钳到 n-1):16 样本下的近似;结论一律再跨 3 seeds
    # 取 mean±std,分位定义的细微偏差不影响臂间相对比较。
    pct = lambda p: ttfts_sorted[min(len(ttfts_sorted)-1, int(p*len(ttfts_sorted)))] if ttfts_sorted else None
    return {"prefix_len": prefix_len, "total_len": args.total_len,
            "concurrency": concurrency, "warm": warm, "seed": tok_seed,
            "num_requests": args.num_requests, "flush": fl,
            "completed": len(ttfts), "duration_s": round(dur, 3),
            "ttft_ms_mean": round(sum(ttfts)/len(ttfts), 2) if ttfts else None,
            "ttft_ms_p50": round(pct(0.5), 2) if ttfts else None,
            "ttft_ms_p95": round(pct(0.95), 2) if ttfts else None,
            "cached_tokens": [r["cached_tokens"] for r in results],
            "ttft_ms": [round(t, 2) for t in ttfts]}
```

角色:并发的操作性定义(客户端在途上限,server 侧 batch 是引擎自己的事),
以及 raw 的输出契约——注意**逐请求数组**(`cached_tokens`、`ttft_ms`)整个
落盘,聚合器才可能做逐请求校验;只存分位数的 bench 无法事后审计。改错会怎样:
分位数若用插值定义,16 样本下与索引定义差一个次序统计量,但因两臂同法,相对
比较不受影响——这是把"分位定义"从实验变量里消掉的做法。

**第 4 段 · 聚合侧闭环硬校验**(scripts/aggregate_p03.py:27-38)

```python
for (arm, c, pl), ds in sorted(rows.items()):
    p50s = [d["ttft_ms_p50"] for d in ds]; means = [d["ttft_ms_mean"] for d in ds]
    # 闭环校验:计时臂全部请求逐个查(预热请求不落 raw)。on 臂预热已种树,
    # 每请求 cached 必须恰=prefix_len;off 臂或 pl=0 必须恰=0(usage 缺失归 0)。
    ok = True
    for d in ds:
        for cv in d["cached_tokens"]:
            v = cv or 0
            if arm == "on" and pl > 0 and v != pl: ok = False
            if (arm == "off" or pl == 0) and v != 0: ok = False
    out.write(f"{arm},{c},{pl},{len(ds)},{st.mean(p50s):.2f},{st.stdev(p50s):.2f},{st.mean(means):.2f},{pl if arm=='on' else 0},{ok}\n")
    print(f"{arm:3s} c{c} pl={pl:5d} p50={st.mean(p50s):6.2f}±{st.stdev(p50s):4.2f} ms mean={st.mean(means):6.2f} cached_ok={ok}")
```

角色:数据有效性 gate。`cached_ok_all` 是"每一个请求的命中数都精确等于协议
期望"的合取——任一请求违反即整格 False,该格数字不得进对外文档。`cv or 0`
处理的是 usage 语义:cached=0 时字段**缺失**而非 0(下一段)。改错会怎样:
校验若放宽为 `v >= pl*0.9` 之类的软阈值,预热失败、残树、中间层改写请求体
这三类协议破坏都会漏网(讲义 02 的 EXP-P06 首轮作废正是靠同思路的硬 gate
才被发现)。

**第 5 段 · 与引擎计数器对账**(scripts/aggregate_p03.py:41-47)

```python
m = glob.glob(f"{P}/*_radix_on_metrics.txt")
if m:
    txt = open(m[0]).read()
    for pat in ["device_hit", 'mode="input"', "cache_hit_rate"]:
        for line in txt.splitlines():
            if pat in line and not line.startswith("#"):
                print("METRICS:", line.strip())
```

角色:归因三方相等(§3.4)的第二方。把 /metrics 快照里的 device_hit counter
原样打印,与客户端 Σcached 对账。metrics 快照同时是一个现成的教学反例:同一
文件里 `cache_hit_rate` 是 0.0(窗口化 gauge,空闲后归零)而
`prefill_effective_tokens_total{mode="device_hit"}` 是 466944.0——累计口径
只能用 counter,gauge 不能当累计命中率读(EXP-P03 §6 的坑)。

**第 6 段 · 上限来自哪一行**(上游 sglang v0.5.18,
python/sglang/srt/managers/schedule_batch.py:1411-1416)

```python
    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)
```

角色:§3.1 推导的源码落点。注意第二个分支:请求要 logprob 时上限还会进一步
压低到 `logprob_start_len`——要"从头给 logprob"就得从头重算,缓存复用与
logprob 是此消彼长的。EXP-P01 的 1324/1325 精确命中了第一个分支。

**第 7 段 · cached 的报告语义**(上游
python/sglang/srt/entrypoints/openai/usage_processor.py:12-15)

```python
    @staticmethod
    def _details_if_cached(count: int) -> Optional[PromptTokensDetails]:
        """Return PromptTokensDetails only when count > 0 (keeps JSON slim)."""
        return PromptTokensDetails(cached_tokens=count) if count > 0 else None
```

角色:解释为什么冷启动请求的 raw 里 `cached_tokens` 是 `null` 而不是 0
(EXP-P01 首发、P03 off 臂全部如此)。details 仅在 >0 且
`--enable-cache-report` 开启时携带。改错会怎样:客户端不做 `or 0` 归一,
统计代码把 None 参与求和直接抛异常,或被当 0 静默——本仓所有脚本统一
`or 0`(EXP-P02 §6 的结论)。

## 5. 实验数据怎么读

以 figures/fig2_p07_ttft_vs_prefix_8b.png(8B 收益曲线)为主样本:

- **轴与口径**:x 轴是共享前缀长度(token,总长固定 2048——定总长扫前缀,
  变量只有 prefix_len);y 轴是 TTFT p50(ms),点值是 3 seeds 的 p50 均值,
  误差条是 3 seeds 间的 std(不是 16 请求内的分布宽度——组内分布见 raw 的
  ttft_ms 数组)。三条线:on·并发1、on·并发8、off·并发1(反例臂)。
- **这个设计防了哪些坑**:①**反例臂**(disable-radix 起服)排除"降幅来自
  别的什么"(如长前缀带来的分词/调度差异)——off 臂 229.7→231.8 ms 无趋势;
  ②**预热计时外**排除"首请求替大家种树"的混合态;③**每点 flush**排除跨点
  残树;④**3 独立 seeds**给出轮间波动的量尺,单轮的 0.28 ms 微趋势(EXP-P03
  off 臂)才能被按协议判为"无可区分";⑤**逐请求 cached 硬校验 + 计数器对账**
  (§3.4)把"命中了多少"钉死。
- **p50 与 mean 的分工**(一个仓内真实案例):exp_p07 csv 的 on/c1/512 行
  p50=180.51 而 mean=788.41——3 seeds 中一个 seed 的 1/16 请求出现 29,309 ms
  孤立离群(EXP-P07 §7,根因不可考,server 日志被启动截断,按终端级证据
  记录)。headline 用 p50(稳健),mean 列如实保留污染值不做剔除——两列并存
  本身就是诚实度的展示面。c8 的 mean < p50(如 1068.3 vs csv mean 列)则是
  并发批内先完成者拉低均值的左偏,属预期。
- **机理账怎么列**(读图时心算):并发 1 斜率 =(228.4−52.9)/1792 ≈ 98
  µs/token,乘回去可预测任意前缀长度的 TTFT;并发 8 斜率 ≈(1068.3−234.5)
  /1792 ≈ 465 µs/token,斜率比 ≈ 4.8 就是排队放大系数(0.6B 版本是 5.3 →
  40 µs/token,×7.6,EXP-P03 §6)。fig1(0.6B)与 fig2(8B)同图形语言,
  对读即见"收益天花板随模型规模"。
- **fig3(逐出悬崖)**:x 轴直接取重用距离 $D=8192\times(1+cr)$ 而非 cr——
  $D$ 才是机理变量,cr 只是构造手段;三色柱是三个池位,读图就是逐格核对
  "池 ≥ D ⇔ 命中 1.0"。std=0 的误差条不是"没画",是协议确定性的结果
  (temperature=0、串行、固定 seed 负载下逐出顺序完全确定)。

## 6. 误区与边界

至少踩过一次才写得出来的错误直觉(前两条是仓内被证伪/修正的真实案例):

1. **"thinking 开关会破坏前缀共享"——本仓预注册假设,已被证伪**(EXP-P02,
   §3.2 全程)。教训的一般形式:模板参数落在渲染结果的哪个位置,必须做一次
   CPU 渲染 diff 才知道,"参数改了所以头部变了"是想当然。边界:结论限定
   Qwen3 模板族;换 system-prompt 开头非 `<think>` 的模型,+1 现象不复现。
2. **"缓存吃紧时命中率按比例下降"——本仓预注册用词"退化曲线",被实测修正为
   阶跃**(EXP-P05,§3.5)。轮转工作集 + LRU 下不存在"缓存小一点、命中低
   一点"的软着陆;容量规划按重用距离,越线即崩。边界:非轮转的真实到达序
   (Zipf 偏斜、Poisson)会让各前缀的 $D$ 异质化,悬崖会被抹成分段的软化
   曲线——模型给的是每个前缀各自的越线条件,不是全局形状。
3. **"cached_tokens 应该等于 prompt 长度"**:上限是 $n-1$(§3.1),而且
   cached=0 时字段缺失而非 0(§4 第 7 段)。拿"命中=全长"做断言的 gate 会
   假 FAIL,拿 None 当 0 之外的语义会算错命中率。
4. **"cache_hit_rate 指标就是命中率"**:它是窗口化 gauge,空闲后归零——
   metrics 快照里它与 device_hit=466,944 同屏出现 0.0(§4 第 5 段)。累计
   口径必须用 `prefill_effective_tokens_total` 两条 counter 差分。
5. **"并发 1 测得的收益就是缓存的价值"**:0.6B 并发 1 只有 −36%,并发 8 有
   −63%(EXP-P03)——排队项把收益放大(§3.3)。反过来,拿高并发数字不带
   并发定语去讲"单请求快了 63%"同样是错的。

**适用边界(明确列出)**:本讲义全部数字来自单 worker、RTX 4090、SGLang
v0.5.18、Qwen3-0.6B/8B、input_ids 直传的合成负载(随机 token,总长 2048,
输出 32);−77%/−78% 带定语"共享前缀 1792/2048、TTFT p50、并发 1/8、3
seeds";真实 chat 流量(模板渲染、变长、多轮)只有 messages 形态的 probe 臂
(EXP-P01/P02)覆盖,收益曲线不直接外推。

## 7. 连环追问

1. **Q:RadixAttention 缓存的是文本还是 token?**
   token id 序列(RadixKey,radix_cache.py:59),外加 extra_key/cache_salt
   命名空间。文本相同但渲染后 token 不同即 miss——EXP-P02 的矩阵就是逐格
   验证这件事。
2. **Q:为什么命中上限是 input_len−1?**
   至少重算 1 个 token 才有最后位置的 logits 可采样(§3.1 四步推导;
   schedule_batch.py:1411-1416)。实测 1324/1325 精确落在上限(EXP-P01)。
3. **Q:radix tree 为什么优于"整段 prompt 哈希"?**
   哈希只有全等命中,radix tree 给出任意长度的最长公共前缀;匹配中段还能
   分裂节点暴露精确边界(radix_cache.py:704)。代价是树维护与锁。
4. **Q:正在被别的请求使用的 KV 会被逐出吗?**
   不会。lock_ref>0 的节点在 protected 段,逐出扫描跳过
   (inc/dec_lock_ref,radix_cache.py:622-656;叶状态维护 :820)。
5. **Q:−77% 的降幅怎么归因给"省掉的 prefill"而不是别的?**
   三件:off 反例臂持平;逐请求 cached 恰=prefix_len 的硬校验;engine
   device_hit 计数器与客户端 Σcached、协议期望三方 466,944 逐 token 相等
   (§3.4)。
6. **Q:为什么 0.6B 收益比 8B 小这么多?**
   TTFT = 地板 C + prefill;0.6B 的 C(~17 ms)占比大,可省部分小。收益
   天花板 ≈ prefill 占比 × 前缀占比(§3.3 的账)。
7. **Q:预热请求会污染测量吗?**
   预热在计时外,且尾部加 8 个随机 token 保证它不被计时请求全长命中;聚合
   校验(cached 恰=prefix_len)会捕获任何预热异常(§4 第 2/4 段)。
8. **Q:flush_cache 一定清干净吗?**
   只在引擎 idle 时真清,有 pending 请求会失败(theory/01 §4);所以每臂
   开始前要确认 flush 成功再动,冷验证(A 发 cached 缺失)双确认(EXP-P02 §7)。
9. **Q:重用距离模型对 lfu 逐出还成立吗?**
   不直接成立。模型第 3 步依赖"LRU 只看最近"的性质;lfu 按 hit_count
   (evict_policy.py:22),轮转热前缀的频次高于冷流量,理论上保热更好——
   本仓未测(EXP-P05 §7 列 backlog),不外推。
10. **Q:cache_salt 存在的意义?**
    安全语义:命中带来的时延差可作侧信道探测"别人是否问过同样的前缀";
    salt 提供硬隔离(EXP-P02 salt_diff 格实测全 miss)。反过来实验里可用
    salt 制造"人为 miss"对照。
11. **压力问 Q:466,944 三方相等,是否证明了 TTFT 降幅的因果?**
    诚实答:它证明的是**命中账目**分毫不差(没有虚报/漏报的 token),因果
    归因还需要 off 反例臂(排除其它机制)与曲线的线性形态(斜率稳定)共同
    支撑。严格的"降幅逐请求分解到 prefill 段"需要 server 侧 per-request
    时序(queue_time/prefill_finished_time),本仓用 counter 差分 + 反例臂
    的组合替代,这是口径上的取舍(theory/03 §3 明示不要拿 client TTFT 直接
    说"prefill 变快了")。
12. **压力问 Q:−77% 在真实业务流量上还成立吗?**
    不能承诺。该数字的定语是共享前缀 87.5%(1792/2048)的合成负载——这接近
    "长系统提示词 + 短用户后缀"的理想形态。真实流量前缀占比、多轮结构、
    到达序都不同;0.6B/8B 的对比说明收益还强依赖模型的 prefill 占比。本仓
    本项目对外措辞的硬约束是:数字必须连同负载定语一起说,"前缀缓存让 TTFT 降
    77%"单独成句是禁用措辞。

## 8. 工业对照与延伸

与生产实现的差距各在哪一层:

- **vLLM(automatic prefix caching)**:同思想不同数据结构——vLLM 以固定
  大小 block 的 hash 链实现前缀复用(block 粒度命中),SGLang radix tree 是
  token/page 粒度 + 节点分裂,长尾前缀的命中粒度更细;代价是树结构的维护与
  锁复杂度。本仓 page_size=1,未测 page>1 时的对齐折扣(theory/03 §4)。
- **SGLang 上游 HiCache / hiradix**(mem_cache/hiradix_cache.py):KV 分层到
  host 内存甚至存储,`cached_tokens_total{cache_source=device|host}` 的
  host 维度本仓从未触发——单机显存池内的结论,不含分层缓存的换入换出。
- **多副本 serving**:单 worker 结论到了多副本会叠加"前缀→副本映射"这一层
  (sgl-model-gateway 的字符级近似树与 engine 的 token 树是两棵树),见
  讲义 02 与 EXP-P06。
- **确定性与缓存的互斥面**:上游 `cache_finished_req` 里有
  `disable_finished_insert`(确定性模式不插树)——复用与逐 bit 可复现之间
  存在工程权衡,本仓 temperature=0 的确定性验证(EXP-P01)未开该模式。

延伸阅读(源码/文档锚):

1. mem_cache/radix_cache.py:376-434(match_prefix 全注释)与 :678-702
   (下行循环)——树操作的第一手实现。
2. managers/schedule_batch.py:1292 起(init_next_round_input)——请求到
   匹配的调度侧入口,limit/salt 如何传入。
3. mem_cache/evict_policy.py:16-38(五种逐出策略的 priority 函数)+
   mem_cache/common.py:114-138(需求驱动逐出)——重用距离模型的对手方。
4. docs/theory/01_radix_prefix_cache.md 与 03_workload_contract_pitfalls.md
   ——本讲义的机制笔记底稿(全部 file:line 锚)。
5. records/EXP-P02_token_contract_matrix.md ——预注册证伪案例的原始记录
   (§7 含 flush 语义的协议偏差说明)。
