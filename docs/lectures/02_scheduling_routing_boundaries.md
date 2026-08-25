# 讲义 02 · 调度与路由的边界:lpm/fcfs 权衡形态与"映射分散度"

> 读者:准备校招面试的作者本人,以及第一次接触 cache-aware 调度/路由的工程师。
> 读法:不跳步。上游源码锚以 /root/repos/sglang-v0.5.18 为准(worker 侧
> python/sglang/srt/,router 侧 sgl-model-gateway/);实验数字全部带 EXP 锚。

## 1. 这一篇回答什么问题

前缀缓存之上还有两层"cache-aware":引擎内的调度(lpm 排序等待队列)与多副本
前的路由(cache_aware 策略选卡)。这两层各自何时有收益、何时反噬。读完你应当
能:①沿上游源码走通 lpm 的完整决策链(>128 退化、in-batch 去重、排序键);
②解释为什么同一协议下 0.6B 的结论是"反劣"而 8B 是"分位数再分配"——模型重量
如何改变权衡形态,并给出排队论直觉;③复盘 EXP-P06 的双预测双证伪(奇偶巧合
与亲和集中),说清"映射分散度 > 策略标签"这句一般化结论的推导与限度。

## 2. 直觉与第一性原理

**没有调度策略的世界**:引擎每一步从等待队列头部取请求组 batch。到达序是
客户端与网络决定的,共享前缀的请求彼此打散——A 组首请求的 KV 还没插进树
(radix 树在请求**完成**时才插 input+output 全序列),同组第二条已经在跑,
本可命中的前缀 miss 掉了。调度器能做的唯一一件事:**重排等待队列**,把同前缀
请求聚到一起,让后来者稳定踩在前者刚种下的树上。

**代价从哪里来**:重排不创造算力,只是搬运等待时间。把 A 组聚簇提前,B 组就
整组延后——收益(命中率、被提前者的 TTFT)与代价(被延后者的尾延迟)是同一
枚硬币的两面。排队论里这对应经典结论:按"有效服务时间最短优先"(命中多 →
需重算的 prefill 短 → 服务时间短)排序能压低平均/中位等待,但牺牲最长作业的
公平性,尾部分位数变差。lpm 按最长前缀匹配降序排,正是"有效 prefill 最短
优先"的一个近似。

**日常类比与失效点**:超市收银把"只买一件"的顾客插到快速通道,平均结账时间
下降,但大采购车被越插越后。类比失效点:①lpm 的"件数"(前缀匹配长度)是
**动态**的——前一个请求结完账,后面所有同组请求的"件数"才变少(树插入),
排序建立的假设会随执行漂移;②收银员不会在队伍超过 128 人时突然放弃分流,
lpm 会(下一节第一段源码),这个工程保护把"重载"变成了它的设计边界之外。

**路由层的直觉**:多副本下每张卡一棵独立的 radix 树。路由决定"谁看见哪些
前缀",等价于决定**每张卡的重用距离**(讲义 01 §3.5):把热前缀分散钉住,
每卡只看热集的 1/N,重用距离缩短 N 倍——这就是"cache-aware 路由 = 扩大有效
池容量"的推导。EXP-P06 证明这条推导的前提(**分散**且**稳定**的映射)恰恰
不被现成策略保证——预测双双被证伪,见 §3.4。

## 3. 完整机制与两阶段认知

### 3.1 lpm 决策链(上游源码走读,文件均为
python/sglang/srt/managers/schedule_policy.py)

因果链按执行顺序拆五步:

1. **策略判定**:每轮调度先问队列多长。`_determine_active_policy`(:290-294)
   在 lpm 且等待队列 >128 时**整轮退化为 fcfs**——前缀匹配与排序是 O(队列长
   × 树深) 的开销,上游选择在重载时保调度器吞吐。含义:lpm 的设计窗口就是
   ≤128 等待;EXP-P04/P08 的 boundary 档(192 请求 @ 并发 64)故意跨过它。
2. **逐请求匹配**:`_compute_prefix_matches`(:314)对队列里每个请求做一次
   树匹配,把匹配长度写进 `req.num_matched_prefix_tokens`。
3. **in-batch 去重**:匹配很短(≤32,CHECK_THRESHOLD :81)的请求,再到一棵
   **模拟树**(等待队列自己的 radix)里查:如果队列里已有 ≥32 token 同前缀的
   请求(DEPRIORITIZE_THRESHOLD :88),当前请求被**临时降权**——让每个新前缀
   只有一个"开路者"先跑完种树,其余同伴等树种好再跑,避免同批并发把同一前缀
   重算 N 遍。
4. **排序**:`_sort_by_longest_prefix`(:373-384)按匹配长度降序,被降权者
   排到队尾(`float("inf")`)。
5. **执行漂移**:排序只发生在调度时刻;树随请求完成持续变化,fcfs 档下同组
   请求并发交错时,"首请求 KV 未入树、同组已在跑"的 miss 正是 8B 命中差
   17.7pp 的机理候选(EXP-P08 §6,推断标注)。

### 3.2 两阶段认知:0.6B 反劣 → 8B 分位数再分配

同一协议(bench_groups.py:组内共享 1536/2048 前缀、全列表 shuffle 对抗
到达序、std=G8×R8@c16 / boundary=G16×R12@c64、3 seeds)先后在两个模型上跑:

**第一阶段(EXP-P04,Qwen3-0.6B,data/derived/exp_p04_fcfs_vs_lpm.csv)**:

| 档 | 策略 | p50(ms) | p99(ms) | hit_frac |
|---|---|---|---|---|
| std | fcfs | 90.0±0.6 | 280±53 | 0.970±0.027 |
| std | lpm | 90.4±0.4 | 265±31 | 0.982±0.006 |
| boundary | fcfs | 249±8 | **661±36** | **0.992±0.011** |
| boundary | lpm | 233±6 | **747±20** | **0.968±0.005** |

std 档按预锁阈值判平(差值与轮间波动同量级);boundary 档 lpm p99 反劣
13%(>2σ)且 hit −2.4pp——超出设计窗口后排序反成负资产,且没有换来什么。

**第二阶段(EXP-P08,Qwen3-8B,data/derived/exp_p08_8b_fcfs_vs_lpm.csv)**:

| 档 | 策略 | p50(ms) | p99(ms) | hit_frac |
|---|---|---|---|---|
| boundary | fcfs | 6659±274 | 9538±535 | 0.757±0.034 |
| boundary | lpm | **2505±767** | **15656±1293** | **0.934±0.011** |

同一积压档,lpm p50 **−62%**、hit **+17.7pp**,但 p99 **+64%**——不再是
"反劣",而是延迟从中位数搬到尾部换命中率。

**为什么模型重量改变权衡形态**(排队论直觉,逐步):

1. 排序的收益 = 每次命中省掉的 prefill 时间。0.6B 一条 2048 token 的 prefill
   在 ~10 ms 量级(EXP-P03 miss TTFT 26.84 ms 内含 ~17 ms 地板),8B 在
   ~200 ms 量级(EXP-P07 miss TTFT 228.4 ms)——**命中价值放大约 20 倍**。
2. 排序的代价 = 被延后的组要多等的时间,同样以"别人的 prefill"计价——
   **代价也同比放大**。两者放大后,原本淹没在噪声里的重排效果(0.6B std 档
   判平)变得可测,且分别落在不同分位数上:被聚簇提前的多数请求压低 p50,
   被推后的少数组撑爆 p99。
3. 命中率差也被放大:0.6B 时 prefill 快,fcfs 下首请求很快完成入树,同组
   miss 窗口短(boundary 档 fcfs hit 0.992);8B prefill 慢,miss 窗口长
   (fcfs hit 掉到 0.757),lpm 的串行聚簇才有 17.7pp 的空间。
4. 0.6B boundary 档的"单纯反劣"是同一机制的退化形态:收益侧(1、3)没长大,
   代价侧(>128 退化让排序假设中途失效 + in-batch 降权把重复前缀请求延后)
   先出现——只剩账单没有收入。
   (①②两个候选机理未逐请求 trace 分离,EXP-P04 §7 如实列开放。)

**结论的正确形状**:"lpm 好不好"不是标量问题。答案必须带三个定语——模型
(prefill 占比)、负载(是否超 128 窗口)、SLO 分位数(p50 还是 p99)。本仓
本项目把无定语的"lpm 更好/更差"列为禁用措辞,原因即 P04 与 P08
的结论随定语翻转。

### 3.3 cache-aware 路由:决策伪码与失衡回退

router(sgl-model-gateway,PyPI sglang-router 0.3.2)对每个请求
(sgl-model-gateway/src/policies/cache_aware.rs:387 起):

```
healthy = 健康 worker                                   # :393
(min_load, max_load) = 各 worker 当前负载                # :405-409
is_imbalanced = (max-min) > balance_abs_threshold(64)
             && max > min × balance_rel_threshold(1.5)   # :412-413
if is_imbalanced -> 选最小负载(随机 tie-break),仍插树   # :415-424
match_rate = 匹配字符数 / 输入字符数                      # :436-441
if match_rate > cache_threshold(0.3)
    -> 送近似树记录的持有者 worker(若健康)               # :444-450
else -> 送最小负载(随机 tie-break)                       # :456-467
插树(text -> 被选 worker);处理数 +1                     # :471
```

三个必须知道的事实:①近似树存**原始字符**不是 token(tree.rs;文件头注释
自述为省 tokenize 开销)——router 的 match_rate 与 engine 的真实命中是两棵
不同的树上的两个量,通常同向但不相等;②失衡回退是安全阀:亲和性在偏斜负载
下会把请求堆到一张卡,回退用最短队列换回并行度;③**回退的触发条件依赖
负载计数**——串行注入(在途恒 ≤1)时 max−min 永远越不过 64 的绝对阈值,
回退**永不触发**,这正是 EXP-P06 击中的机理格。

### 3.4 EXP-P06 双证伪全程:奇偶巧合与亲和集中

**预注册预测**(由讲义 01 §3.5 的重用距离模型推出,跑前锁定):双 worker 各限
池 8192 token,6 个热前缀(每请求总长 ~2150 字符形态,工作集 ~12900 token)
> 单池、< 双池之和。预测:

- H-rr:round_robin 交替分发 → 每卡看到全部 6 个前缀 → 每卡 $D \approx
  6 \times 2150 > 8192$ → 双卡 thrash,hit → ~0。
- H-ca:cache_aware 把每前缀钉在一张卡 → 每卡 ~3 前缀 → $D \approx 6450
  < 8192$ → hit → ~1("路由=扩容")。

**实测(全部格 3 seeds、seed 间 std=0,data/derived/exp_p06_routing_pool.csv)
——两条预测双双反向**:

| 配置 | 策略 | hot_hit | 流量分布(worker 计数器差分) |
|---|---|---|---|
| hot6(偶) | round_robin | **1.0000**(24/24 全命中) | ~50/50(30878/30921) |
| hot6(偶) | cache_aware | **0.0020**(0/24) | **100/0**(61799/0) |
| hot5(奇,对照) | round_robin | 0.0020 | ~50/50 |
| hot5(奇,对照) | cache_aware | 0.0020 | 100/0(51507/0) |

**证伪一(rr 全命中是巧合)**:轮转周期 6 与 worker 数 2 **整除对齐**——严格
轮询下第 1/3/5 个前缀永远落卡 A,第 2/4/6 个永远落卡 B,rr 意外成了完美分片,
每卡 3 前缀、$D \approx 6450 < 8192$,全命中。这不是能力而是奇偶巧合——
**hot5 对照臂**(5 与 2 互素,每个前缀轮流落两张卡)打破整除后 rr 立即崩塌,
而流量仍然均分(26589/24918,raw=data/raw/EXP-P06/20260824T165910_hot5_round_robin_s20260824.json)
——"分得均匀"与"分得对"被这一格干净剥离。
**证伪二(cache_aware 亲和集中)**:worker 计数器差分直接显示全部流量落在
一张卡(61799/0)。机理:串行注入下负载恒 0,失衡回退永不触发(§3.3 第③);
冷启动时预热流量把 6 个前缀全部记到同一 worker 名下,此后 match_rate 恒高、
亲和恒指向该卡——~12900 token 的工作集塞进 8192 的单池,thrash,hit 0.002
(残余 3 个 token 是模板头级别的碎屑)。

**修正后的一般化**:cache-aware 亲和等效于扩容的前提是 tenant **分散**在多卡;
它的冷启动分配在低负载下会**集中**,此时亲和与容量约束相乘为负。rr 的命中
依赖热集数与副本数的整除关系,不可依赖。一句话:**容量受限的多副本里,前缀→
副本映射的质量(分散且稳定)比"cache-aware"这个策略标签重要;两种现成策略都
不保证这一点**。限度见 §6 条 5。

## 4. 代码逐段走读

按"引擎调度 → router 决策 → 测量端"的顺序,8 段。上游引用逐字拷贝。

**第 1 段 · lpm 的重载保护**(schedule_policy.py:290-294)

```python
    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy
```

角色:整个 P04/P08 边界档设计的靶点。128 是硬编码,不是 flag。改错会怎样:
若没有这条保护,重载下每轮调度对几百个请求做树匹配 + 排序,调度器自身成为
瓶颈;而有了它,"lpm 开着"不等于"lpm 在工作"——boundary 档里策略在 fcfs 与
lpm 之间随队列长度抖动,排序建立的聚簇假设中途失效(EXP-P04 §6 候选机理①)。

**第 2 段 · 每轮调度的分派**(schedule_policy.py:254-268)

```python
        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
```

角色:fcfs 的"实现"是一个提前 return(什么都不做,队列保持到达序)——两臂
对比的本质是"排序 vs 不排序",而非两套算法。注意 `policy`(本轮生效,可能
已被第 1 段退化)与 `self.policy`(配置)是两个变量,fcfs 分支查的是配置——
配置为 fcfs 时连匹配都不做,per-request 的 num_matched_prefix_tokens 走
另一条快照路径(:246-252)。

**第 3 段 · in-batch 前缀去重**(schedule_policy.py:339-358)

```python
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(
                            token_ids=prefix_ids,
                            extra_key=extra_key,
                            cache_salt=cache_salt,
                        )
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result, extra_key=extra_key
                    )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
```

角色:等待队列自己也有一棵(模拟)radix 树。真树 miss(匹配 ≤32)但队列里
已有同前缀请求(模拟树命中 ≥32)的请求被降权——"让开路者先种树"。两个阈值
都默认 32(:81-89,环境变量可调)。改错会怎样:去掉降权,同批 N 条同前缀
请求并发跑,同一前缀被重算 N 遍;但在 G16×R12 的大批量同组负载下,这条路径
也把大量请求推到队尾——EXP-P04 §6 的候选机理②,收益机制在超窗负载下的
另一面。

**第 4 段 · 排序键**(schedule_policy.py:373-384)

```python
    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )
```

角色:lpm 的全部"算法"就是这一个 sort:匹配长的在前(负号),被降权的最后
(inf)。它是"有效 prefill 最短优先"的近似(§2 排队论直觉)。改错会怎样:
去掉负号即变成"最短匹配优先",命中优势反转;Python sort 稳定,同匹配长度的
请求保持到达序——lpm 内部嵌着 fcfs 作 tie-break,这也是它对到达序方差的
削平作用(§5 的方差证据)可解释的原因之一。

**第 5 段 · router 失衡回退**(sgl-model-gateway/src/policies/cache_aware.rs:412-424)

```rust
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                &request_text,
                &healthy_indices,
                &tree_key,
                max_load,
                min_load,
            );
        }
```

角色:亲和与均衡的仲裁点,**先问负载再问缓存**。两个阈值(默认 64 / 1.5,
python 侧 router_args)是"与"关系:绝对差要大**且**相对比要大。改错会怎样:
换成"或",小流量下轻微不均就放弃亲和,命中率被随机打散;EXP-P06 的教学点
相反——串行注入下 max−min ≤ 1 远小于 64,这个分支从未执行,亲和集中无人
纠偏(§3.4 证伪二)。

**第 6 段 · router 亲和路径**(cache_aware.rs:436-450, 469-471)

```rust
            let result = tree.prefix_match_with_counts(text);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };

            // Select worker without String allocation
            let selected_idx = if match_rate > self.config.cache_threshold {
                // Cache hit path: find worker by URL (compare &str directly, no allocation)
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
```

```rust
            if let Some(idx) = selected_idx {
                // Update the tree with this request (use worker URL directly, no allocation)
                tree.insert(text, workers[idx].url());
```

角色:匹配对象是 `text` 的**字符**;`result.tenant` 是近似树记录的"这段前缀
上次被送去的 worker"——树的 tenant 归属就是路由记忆,`tree.insert` 在**每次
决策后**写回,于是首个请求落在哪张卡(冷启动、低负载时由 min-load 分支决定)
会被后续同前缀请求持续强化——这就是亲和集中的自增强回路。改错会怎样:阈值
0.3 调成 0,任何一丁点字符重合都触发亲和,不同前缀会被模板头的公共字符
错误地钉到同一张卡。

**第 7 段 · 测量端的对抗设计与命中口径**(scripts/bench_groups.py:60-67, 79-82)

```python
    reqs = []
    for g in range(a.groups):
        prefix = ids_of(a.prefix_len, a.seed * 100 + g)
        for r in range(a.per_group):
            suffix = ids_of(a.total_len - a.prefix_len, a.seed * 100000 + g * 1000 + r)
            reqs.append((g, prefix + suffix))
    random.seed(a.seed)
    random.shuffle(reqs)                      # 对抗到达序:聚簇能力归属调度器而非注入序
```

```python
    ttfts = sorted(r["ttft_ms"] for r in res if r["ttft_ms"] is not None)
    n = len(ttfts); pct = lambda p: ttfts[min(n - 1, int(p * n))]   # 索引分位近似,跨 seed 再取 mean±std
    hit_tokens = sum(r["cached"] for r in res)
    possible = a.prefix_len * (len(reqs) - a.groups)   # 每组首请求必 miss:分母扣 G 才是理论上限
```

角色:shuffle 是本实验的灵魂——按组顺序注入时 fcfs 也天然聚簇,两策略没有
差异空间;打散后"把同前缀请求重新聚到一起"的能力才归属调度器。hit_fraction
的分母扣掉 G(每组首请求必 miss,树里尚无该前缀),命中率才能跨配置比较。
改错会怎样:分母不扣 G,理论上限变成 (N·prefix_len),满命中也只能到
(N−G)/N ≈ 0.92(G16×R12 时),两臂差异被系统性压缩,0.992 vs 0.968 这种
2σ 分离可能就判不出来了。

**第 8 段 · 中间层改写的防线**(scripts/bench_route_pool.py:47-59)

```python
async def one(session, base, model, text, out_tokens=8, min_prompt=1000):
    # 文本形态负载:router 会按 OpenAI schema 重序列化,只有 messages 能存活。
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": False}
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        d = await r.json()
    u = d.get("usage", {}) or {}
    pt = u.get("prompt_tokens") or 0
    if pt < min_prompt:                       # 硬 gate:防请求体被中间层静默改写(首轮教训)
        raise RuntimeError(f"prompt_tokens={pt} < {min_prompt}: payload degraded, resp={str(d)[:200]}")
    det = u.get("prompt_tokens_details") or {}
    return det.get("cached_tokens") or 0, pt
```

角色:EXP-P06 首轮整批作废换来的防线。router 严格按 OpenAI schema 重序列化,
**丢弃 `input_ids` 扩展字段**——首轮全部请求静默退化成 ~10 token(靠
cached=8 与 worker 计数增量反推才发现)。修正三件套:负载改文本形态(恰好也
是 cache_aware 近似树的匹配对象,对齐了被测机制)、响应 `prompt_tokens` 硬
gate(低于下限直接抛异常,实验 fail-fast 而非静默继续)、每臂前置校验 router
指标的 policy 标签(首轮另一个事故:router 用 setproctitle 改名致旧身份校验
落空,第一支 router 存活跨臂,cache_aware 臂实际在跑 rr)。改错会怎样:去掉
这个 gate,拿到的仍是一套"看起来正常"的 json——静默退化的实验比失败的实验
危险得多。

## 5. 实验数据怎么读

- **fig4(figures/fig4_p08_sched_tradeoff.png)**:横条图,p50 与 p99 两组、
  fcfs/lpm 两色并排,只画 boundary 档(std 档判平,混入会稀释结论)。读法:
  lpm 的 p50 条比 fcfs 短(2505 vs 6659,−62%)而 p99 条比 fcfs 长(15656
  vs 9538,+64%)——"换来什么/付出什么"同图可见,这就是"分位数再分配"的
  图形形态。误差条是 3 seeds 间 std;lpm p50 的 ±767 之大不是测量差,是机制
  的一部分:哪些组先被聚簇决定中位请求落点,对 seed 敏感(EXP-P08 §7)。
- **方差本身是机制证据**(EXP-P04 §7):std 档 fcfs 的 hit std=0.027 ≫ lpm
  的 0.006——fcfs 命中依赖 shuffle 出的到达序,seed 间起伏;lpm 排序削平了
  这种随机性。读表时不要只看均值列:两臂方差的量级差直接指认"排序在起作用",
  即使均值差判平。
- **P06 的表怎么读**(data/derived/exp_p06_routing_pool.csv):关键列是
  worker0/1_traffic_mean(两 worker `prompt_tokens_total` 计数器的 before/
  after 差分,3 seeds 均值)——hot6_even 的 cache_aware 行是 0 / 61658,
  流量 100/0 不是推断而是计数器直读。hot_cached 逐请求序列在 raw 里:rr@hot6
  是 [1597, 1625, 1609, ...](全命中;数值大于名义 1536 是文本重编码漂移,
  命中率按 min(1.0, c/prefix_len) 钳制,bench_route_pool.py:109),崩塌臂
  是清一色 [3, 3, 3, ...](只剩模板头碎屑)。
- **std=0 怎么理解**:P05/P06 全部格 seed 间 std=0,不是"没测出波动",而是
  协议确定性的体现——temperature=0、串行注入、seed 只改变 token 内容不改变
  结构(热集大小/轮转序),逐出与路由的行为逐次完全相同。看到 std=0 应当去
  查协议是否确定性,而不是怀疑数据造假;反之,把这种格子的结论外推到并发/
  随机到达时必须重新测。
- **防坑清单(本组实验特有)**:①每策略**各起一次 worker**(而非热切换),
  杜绝上一策略的树残留;②臂前 flush + 预热;③路由臂的 policy 标签前置 gate
  与 prompt_tokens 硬 gate(§4 第 8 段);④对照臂设计——hot5 奇数臂专为
  打破整除关系而设,是"用对照组杀死巧合解释"的教科书局;⑤3 seeds 换 seed
  重跑而非重复同 seed(重复同 seed 的 std=0 无信息量)。
- **机理账**:rr@hot6 每卡看 3 个前缀,重用距离 $D \approx 3 \times 2150 =
  6450 < 8192$ → 命中;rr@hot5 每卡要装全部 5 个前缀($D \approx 5\times2150
  = 10750 > 8192$)→ 崩;cache_aware 单卡装 6 个($\approx 12900 > 8192$)
  → 崩。三个格子共用讲义 01 §3.5 的同一条不等式,这正是"预注册预测可以被
  推导出来"(哪怕预测错了,错的是前提不是推导)的示范。

## 6. 误区与边界

1. **"开 lpm 总没错"——被 EXP-P04 证伪**。轻载(std 档)到达序已足够友好,
   判平;超 128 窗口(boundary 档,0.6B)p99 反劣 13%、hit −2.4pp,单纯
   负资产。cache-aware 调度有收益窗口,窗口外反噬。
2. **"调度策略有标量优劣"——被 EXP-P08 修正**。8B 积压档 lpm p50 −62% 与
   p99 +64% 同时成立;"哪个好"取决于 SLO 定义在哪个分位数。任何不带模型/
   负载/分位数定语的比较结论都应当被追问口径。
3. **"cache-aware 路由 = 扩大有效缓存容量"——本仓预注册假设,被 EXP-P06
   证伪**。亲和等效扩容的前提是映射分散,而冷启动 + 低负载下 cache_aware
   恰恰把全部热前缀集中到一张卡(计数器 100/0)。同实验的镜像误区:"rr 全
   命中说明 rr 对缓存友好"——那是热集数与副本数整除的巧合,hot5 对照立即
   戳破。
4. **"router 的 match_rate 就是命中率"**。router 树是字符级近似(决定去哪张
   卡),engine 树是 token 级真值(决定省不省算);router 命中只是 engine
   命中的必要前置。报告命中率永远以 engine 侧计数为准(theory/02 §3)。
5. **适用边界**:①调度结论限 0.6B(EXP-P04)与 8B(EXP-P08)各自的负载档,
   两记录互为限定;boundary 档机理(退化开关 vs in-batch 降权;fcfs 并发
   交错 miss)为与数据一致的推断,未逐请求 trace,记录里如实标注开放。
   ②P06 是**串行、容量受限、冷启动**的机理格:并发到达会激活失衡回退、打破
   严格轮转的整除结构,高并发行为未测,不外推;cache_aware 冷启动集中于一卡
   的内部路径(确定性 tie-break 还是文档所称随机)未定——两次日志级尝试均
   无决策级输出(EXP-P06 §7,raw=data/raw/EXP-P06/20260824T175130_cache_aware_debug_rationale.txt)。
   ③双副本性能矩阵(吞吐/延迟 sweep)未执行,本讲义不含任何路由性能结论。

## 7. 连环追问

1. **Q:lpm 排序的 key 是什么?**
   每请求对全局 radix 树的匹配长度取负值升序(即匹配长的在前);in-batch
   降权者置 inf 沉底(schedule_policy.py:373-384)。
2. **Q:fcfs 在代码里长什么样?**
   一个提前 return——不排序,队列保持到达序(:254-259)。所以两臂对比干净:
   变量只有"是否重排"。
3. **Q:为什么 boundary 档取 192 请求、并发 64?**
   为了确定性地跨过 lpm 的 128 等待队列退化窗口(:291),把"策略在窗口内/外"
   变成实验变量(EXP-P04 §1 预注册)。
4. **Q:为什么 shuffle 后再注入?**
   按组连续注入时 fcfs 也天然聚簇,策略差异没有表达空间;shuffle 把"聚簇"
   的功劳完全归属调度器(bench_groups.py:67 注释,§4 第 7 段)。
5. **Q:8B 下 fcfs 命中率为什么只有 0.757?**
   并发交错:同组首请求的 KV 要等它**完成**才插树(讲义 01 §3.2 的
   cache_finished_req),8B prefill 慢、miss 窗口长,后续同组请求赶在树种好
   之前进了 batch(EXP-P08 §6,机理推断与 hit/流量数据一致,未逐请求 trace)。
6. **Q:lpm 换命中率的代价为什么落在 p99 而不是均匀摊开?**
   排序是全序重排:收益摊给被提前的多数(p50),代价集中给排最后的少数组
   (整组延后),分布两端被同时拉开——SPT 类调度的教科书性质。
7. **Q:失衡回退的两个阈值为什么是"与"关系?**
   绝对差(64)防小流量误触发,相对比(1.5)防大流量下绝对差虚高;单用任何
   一个都会在某个流量段错误触发/漏触发(cache_aware.rs:412-413)。
8. **Q:怎么证明 cache_aware 臂真的在跑 cache_aware?**
   每臂前置校验 router prometheus 的 `smg_worker_selection_total{policy=...}`
   标签(首轮事故后加的 gate:进程存活 ≠ 策略正确)。进程身份还要容忍
   setproctitle 改名形态(svc.sh 放行三形态)。
9. **Q:"映射分散度>策略标签"能推广到几副本?**
   不等式形式可推广:每卡命中 ⇔ 该卡分到的热前缀重用距离 ≤ 单卡池。N 副本
   下 rr 的"巧合分片"条件变成热集数与 N 的整除结构,更脆;分散且稳定的映射
   (如一致性哈希按前缀分片)是通解方向——但本仓只测了 2 副本,推广是推导
   不是实测。
10. **Q:engine 的 lpm 与 router 的 cache_aware 会互相干扰吗?**
    两层 cache-aware 叠加行为本仓未测(EXP-P04 §8 明确列为扩展问题)。可以
    说的只有机制事实:router 决定"谁看见请求",engine 决定"看见后怎么排"。
11. **压力问 Q:lpm p50=2505±767,std 这么大,−62% 可信吗?**
    诚实答:3 seeds 的区间(约 1738-3272)与 fcfs 的 6659±274 无重叠,方向
    与量级站得住;但 ±767 说明中位数本身对聚簇顺序敏感,−62% 是三轮均值的
    点估计,换负载结构(组数/组大小/并发)不应期望复现这个具体数值。方差大
    是机制(哪组先被聚簇)而非噪声,这一点记录里如实标注(EXP-P08 §7)。
12. **压力问 Q:P06 的结论会不会只是 sglang-router 0.3.2 一个版本的 bug?**
    可能性无法排除——冷启动集中的内部路径未定(日志级两次尝试无决策输出),
    上游新一代实现(experimental/sgl-router 的 cache_aware_zmq,block-hash
    of token ids + ZMQ 事件)已是不同设计。但本仓结论的一般化部分("亲和
    等效扩容以映射分散为前提;整除巧合不可依赖")是从重用距离不等式推出的,
    不依赖该版本的具体实现;版本特定的部分(冷启动全钉一卡)已限定版本号
    并保留为潜在上游 issue 素材(EXP-P06 §8)。

## 8. 工业对照与延伸

- **上游 SGLang 调度**:除 lpm/fcfs 外还有 dfs-weight(按树的 DFS 权重)、
  lof、random、routing-key(schedule_policy.py:200-215 两个枚举)。本仓只测
  了 lpm/fcfs 一对;dfs-weight 对深树负载的行为是自然的下一格。
- **router 新老两代**:被测的 sgl-model-gateway(字符近似树,HTTP 侧)与
  experimental/sgl-router 的 `cache_aware_zmq`(token block-hash + worker
  ZMQ KV 事件上报)是两代设计——后者的树与 engine 真值同源,理论上消除了
  "两棵树不一致"这层误差,但引入事件流的时延与可靠性问题;本仓未测。
- **vLLM 侧**:vLLM 的调度以 continuous batching + 优先级/FCFS 为主线,
  前缀感知调度不是默认路径;多副本亲和通常由外部网关(如各家 gateway 的
  session/prefix 亲和)承担——"engine 内调度"与"gateway 路由"的职责切分
  与 SGLang 同构,对比面试可讲两家 gateway 对"树"的不同近似。
- **与排队论文献的接口**:lpm≈SPT(最短处理时间优先)的近似这一映射,把
  P08 的"分位数再分配"接到调度理论的标准结论上(SPT 最小化平均等待、牺牲
  尾部);面试可从这里把实验数字上升到理论框架。

延伸阅读(源码/文档锚):

1. python/sglang/srt/managers/schedule_policy.py:237-300(calc_priority 全流程
   + 退化开关)与 :314-384(in-batch 去重与排序)——本讲义 §4 前四段的原文。
2. sgl-model-gateway/src/policies/cache_aware.rs:1-61(文件头设计注释,失衡
   判定与参数语义的权威表述)与 tree.rs(字符级近似树实现)。
3. experimental/sgl-router/src/policies/cache_aware_zmq.rs——新一代 token
   级路由树,与被测实现对读。
4. docs/theory/02_router_cache_aware.md——router 机制笔记(观测端点、
   smg_* 指标、无 cache-hit-rate 指标的坑)。
5. records/EXP-P06_routing_pool_capacity.md §7——首轮作废的完整事故记录
   (input_ids 丢弃、setproctitle、日志级排查),工程防线的第一手材料。
