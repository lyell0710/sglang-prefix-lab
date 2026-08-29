# 讲义 01 · RadixAttention 前缀缓存全景:从 token 契约到重用距离

> 读者：准备校招面试的作者本人，以及第一次接触前缀缓存的工程师。读法：不跳步。每个论断后面跟着它的证据锚（EXP 编号 / 文件：行号 / raw 路径）， 所有数字与仓内现行口径逐字一致，来源见 records/ 与 data/derived/。凡属论文/官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节或公式编号， 文档给 URL 路径 + 小节名）；凡属本讲义自己补出的推导或折算，行内标注 "本讲义推导"；无法用检索确认的说法标注"未核实"。上游源码锚以 /root/repos/sglang-v0.5.18 为准（worker 侧 python/sglang/srt/）。

## 目录

- [1. 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的五条能力](#11-本篇要建立的五条能力)
  - [1.2 符号与口径约定](#12-符号与口径约定)
  - [1.3 本篇引用的一级文献(详细出处见 §8.3)](#13-本篇引用的一级文献详细出处见-83)
  - [1.4 三个坐标系:同一件事的三种记账方式](#14-三个坐标系同一件事的三种记账方式)
- [2. 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 前缀可复用而后缀不可:严格陈述与失效条件](#21-前缀可复用而后缀不可严格陈述与失效条件)
  - [2.2 三条贯穿全篇的公理](#22-三条贯穿全篇的公理)
  - [2.3 日常类比与它的两处失效点](#23-日常类比与它的两处失效点)
  - [2.4 RadixAttention 的选型:三种候选结构的代价表](#24-radixattention-的选型三种候选结构的代价表)
  - [2.5 三层"命中"必须分开](#25-三层命中必须分开)
- [3. 完整推导与机制](#3-完整推导与机制)
  - [3.1 树结构与 token 级契约,以及 n−1 上限的推导](#31-树结构与-token-级契约以及-n1-上限的推导)
  - [3.2 chat template 三层变换陷阱:thinking 证伪案例全程复盘](#32-chat-template-三层变换陷阱thinking-证伪案例全程复盘)
  - [3.3 收益账:省掉的 prefill 值多少毫秒](#33-收益账省掉的-prefill-值多少毫秒)
  - [3.4 逐 token 归因闭环:466,944 三方相等怎么做到](#34-逐-token-归因闭环466944-三方相等怎么做到)
  - [3.5 重用距离模型:完整推导与三池验证](#35-重用距离模型完整推导与三池验证)
  - [3.6 KV 池的硬件账:161671 这个数是怎么来的(硬件语义层)](#36-kv-池的硬件账161671-这个数是怎么来的硬件语义层)
  - [3.7 page_size = 1 凭什么可行:分页 KV 的 kernel 侧语义(硬件语义层)](#37-page_size--1-凭什么可行分页-kv-的-kernel-侧语义硬件语义层)
  - [3.8 魔法数总表:每个数字由谁决定](#38-魔法数总表每个数字由谁决定)
- [4. 代码逐段走读:scripts/bench_prefix.py 的停表与闭环](#4-代码逐段走读scriptsbench_prefixpy-的停表与闭环)
- [5. 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 fig2(8B 收益曲线)的读法](#51-fig28b-收益曲线的读法)
  - [5.2 p50 与 mean 的分工(一个仓内真实案例)](#52-p50-与-mean-的分工一个仓内真实案例)
  - [5.3 fig3(逐出悬崖)的读法](#53-fig3逐出悬崖的读法)
  - [5.4 std = 0 意味着什么,以及它不保证什么](#54-std--0-意味着什么以及它不保证什么)
  - [5.5 口径速查:哪些数字能外推,哪些不能](#55-口径速查哪些数字能外推哪些不能)
- [6. 误区与边界](#6-误区与边界)
- [7. 连环追问](#7-连环追问)
- [8. 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 论文/文档声称 vs 本机实测:逐条对照](#81-论文文档声称-vs-本机实测逐条对照)
  - [8.2 与生产实现的差距各在哪一层](#82-与生产实现的差距各在哪一层)
  - [8.3 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)](#83-延伸阅读带精确出处每条一句话说明它能解决什么疑问)

## 1. 这一篇回答什么问题

前缀缓存（SGLang 的 RadixAttention）到底缓存了什么、命中判定发生在哪一层、省下的毫秒从哪里来、什么时候失效。读完你应当能：①在白板上手推"命中上限为什么是 input_len−1"与"LRU 命中 ⇔ 池 ≥ 重用距离"两条结论；②解释为什么"同一段文本" 不等于"同一段前缀"，并完整复盘一次预注册假设被证伪的全过程（EXP-P02《token 契约矩阵》）；③答上 "你的 −77% 怎么测的、凭什么归因给前缀复用"这类追问——包括 466,944 逐 token 三方相等是怎么做到的（EXP-P03《命中收益曲线》/P07）。

### 1.1 本篇要建立的五条能力

1. **契约能力**：知道命中判定发生在 token id 层而不是文本层，能列出"文本相同而 token 不同"的全部成因，并设计出把它们逐格证伪的实验矩阵。
2. **推导能力**：能从 causal attention 的因果结构出发，推出"前缀可复用、后缀不可复用"与"匹配上限 = n−1"两条硬边界，并说清每一步凭什么合法、在什么模型结构下会失效。
3. **缓存理论能力**：能把 KV 池当成一个变长条目的 LRU 缓存，用栈距离/重用距离把"池多大够用"变成一条可验算的不等式，并知道这条不等式哪一半是定理、哪一半靠实验构造兜底。
4. **归因能力**：能把"TTFT 降了 77%"拆成"命中了多少 token"与"每个 token 值多少毫秒"两问，并给出三方独立计数相等的闭环证据形态；同时说清这个闭环 **证明了什么、没有证明什么**。
5. **口径能力**：任何数字出口都带负载定语与硬件定语，知道 −77% 只属于 "共享前缀 1792/2048、TTFT p50、并发 1、Qwen3-8B、单 worker、RTX 4090"。

### 1.2 符号与口径约定

| 符号 | 含义 | 本仓典型取值 |
|---|---|---|
| $n$ | 一条请求的输入 token 数（prompt_tokens） | 2048（收益曲线），1325/1329（probe 臂） |
| $k$ | 该请求实际命中的前缀 token 数（cached_tokens） | 0 / 512 / 1024 / 1536 / 1792 |
| $L_p$ | 共享前缀长度 prefix_len（协议自变量） | 同上 |
| $T$ | 单请求总长 total_len | 2048 |
| $H$ | 热前缀个数 hot_count | 4(EXP-P05)/ 6、5(EXP-P06) |
| $c$ | 冷流量比 cold_ratio，每条热请求后跟几条冷请求 | 0 / 1 / 2 / 4 |
| $D$ | 重用距离：同一热前缀两次访问之间注入池的 token 量 | $8192\times(1+c)$ |
| $P$ | KV 池容量（token 数，`--max-total-tokens`） | 8192 / 16384 / 161671 |
| $p$ | page_size（每个 KV 页的 token 数） | 1（本仓全程） |
| $b$ | 每 token 的 KV 字节数 | 0.6B 112 KiB,8B 144 KiB(§3.6) |
| TTFT | 首 token 延迟（客户端停表，§4 第 1 段） | 见 §3.3 |
| $C$ | TTFT 中与 prefill 无关的固定地板 | 0.6B ~17 ms，8B ~24 ms |
| $S_{\mathrm{prefill}}$ | prefill 段耗时 | 随需重算 token 数近似线性 |
| $W_{\mathrm{queue}}$ | 排队等待项 | 并发 1 时约为 0 |

模型配置取自本机 checkpoint 的 `config.json`（固定 revision 见 ENV.md）： Qwen3-0.6B 28 层、16 Q 头 / 8 KV 头、head_dim 128、bf16；Qwen3-8B 36 层、 32 Q 头 / 8 KV 头、head_dim 128、bf16。两者 KV 头数相同、head_dim 相同， 差别只在层数——这一条决定了 §3.6 的每 token 字节数之比恰为 36/28。

### 1.3 本篇引用的一级文献(详细出处见 §8.3)

- RadixAttention 与 cache-aware 调度：Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104,§3 与 附录 A.2/A.3。
- KV 分页与前缀共享的另一条路线：Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", arXiv:2309.06180,§4、§6.4。
- 栈距离与命中率曲线：Mattson, Gecsei, Slutz, Traiger, "Evaluation techniques for storage hierarchies", IBM Systems Journal 9(2):78-117, 1970(DOI 10.1147/sj.92.0078)。
- 最优离线替换与 LRU 竞争比：Belady, "A study of replacement algorithms for a virtual-storage computer", IBM Systems Journal 5(2):78-101, 1966 (DOI 10.1147/sj.52.0078);Sleator & Tarjan, "Amortized efficiency of list update and paging rules", CACM 28(2):202-208, 1985(DOI 10.1145/2786.2793)。
- 真实流行度下的 LRU 命中率：Fricker, Robert, Roberts, "A versatile and accurate approximation for LRU cache performance", arXiv:1202.3974。
- 分页 KV 的 kernel 侧语义：Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", arXiv:2501.01005,§3.1.1、§3.1.2、 §3.2.1、附录 B。
- 硬件常数：NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书，Appendix A Table 2。

### 1.4 三个坐标系:同一件事的三种记账方式

本篇反复在三个坐标系之间换算，提前声明清楚可以省掉一半困惑：

| 坐标系 | 单位 | 谁决定 | 本篇对应小节 |
|---|---|---|---|
| 契约层 | token id | tokenizer + chat template + RadixKey 命名空间 | §3.1、§3.2 |
| 缓存层 | token（池位） | 重用距离 vs 池容量，逐出策略 | §3.5 |
| 硬件层 | 字节 / FLOP / 毫秒 | 每 token KV 字节数、显存带宽、Tensor 峰值 | §3.3.2、§3.6 |

最常见的错误就是跨坐标系直接搬数字：拿"命中率 0.99"当"省了 99% 的时间"（忽略了地板 $C$，§3.3），或者拿"热集只有 6 个前缀"当"池装得下"（忽略了冷流量把重用距离推远，§3.5）。

## 2. 直觉与第一性原理

**没有前缀缓存的世界**：Transformer 解码是自回归的，生成第 $t$ 个 token 需要前面所有 token 的 Key/Value 张量（KV）。一条请求进来，引擎先对全部输入 token 做一次前向计算（prefill）把 KV 算出来，然后逐 token 解码。现在考虑真实 serving 流量： 几百个请求共享同一个 2000 token 的系统提示词。没有缓存时，这段完全相同的前缀会被逐请求重算几百遍——每一遍的计算结果（KV）逐位相同。计算是幂等的，重复计算纯属浪费，而且浪费的正是 TTFT（首 token 延迟）里最大的一块：prefill。

SGLang 论文把这条观察写成一句话："KV cache computation depends only on prefix tokens. Therefore， requests with the same prompt prefix can reuse the KV cache， reducing redundant computation and memory usage."（arXiv:2312.07104，§3 开头）。注意论文这句话同时主张了两件事——省计算与省显存；本仓的实验只验证了前者（TTFT 曲线，EXP-P03/P07），显存侧的收益（更大 batch）本仓未测，不外推。

### 2.1 前缀可复用而后缀不可:严格陈述与失效条件

**陈述**：设两条请求的输入 token 序列为 $x_{0..n-1}$ 与 $y_{0..m-1}$，若 $x_i = y_i$ 对所有 $i < k$ 成立，则在同一模型、同一算子序列下，两条请求前 $k$ 个位置的 $K^{(l)}_i, V^{(l)}_i$（所有层 $l$）逐位相同。

**逐步推**（每步注明凭什么合法）：

1. 第 0 层输入 $h^{(0)}_i = \mathrm{Embed}(x_i)$，只依赖 $x_i$ 自己。——embedding 是逐 token 查表，不跨位置。
2. 归纳假设：$h^{(l)}_i$ 只依赖 $x_{0..i}$。——attention 被 causal mask 截断，位置 $i$ 只看 $\le i$；FFN 与 norm 是逐位置的。
3. 于是 $K^{(l)}_i = f(h^{(l)}_i, i)$、$V^{(l)}_i = g(h^{(l)}_i)$ 也只依赖 $x_{0..i}$ 与位置下标 $i$。——位置编码（RoPE）只用绝对下标 $i$，而 $i$ 在两条请求里相同（都是从 0 起的前缀）。
4. 对 $i < k$，$x_{0..i} = y_{0..i}$，故两边的 $h, K, V$ 全部相同。**得证。**
5. 反向：$i \ge k$ 时 $x_{0..i} \ne y_{0..i}$，第 2 步的依赖关系里含有已分叉的 token，所以**后缀文本即使一样，KV 也全部不同**——一位都不能复用。

**这条论证在哪一步会失效**（四个真实的边界）：

- **双向注意力**（encoder、BERT 式）：第 2 步的 causal mask 不存在，$h_i$ 依赖全序列，前缀不再是"历史的函数"。
- **滑动窗口注意力**(SWA)：第 2 步仍成立，但 KV 的**生存期**被窗口截断， 上游为此专门有一套混合分配器（`SWATokenToKVPoolAllocator`，见 mem_cache/common.py 的 hybrid 分支，§4 第 10 段）。本仓两个模型都是全注意力， 不触发该路径。
- **状态空间/线性注意力**：没有"每位置一份 KV"这种结构，复用要换成状态快照。
- **位置偏移**：如果实现把前缀放到非零起始位置（例如把缓存段拼到中间），第 3 步的"位置下标相同"就不成立，RoPE 之后的 K 会不同。前缀复用天然只在**前缀** 上成立，这也是"prefix caching"而不是"substring caching"的原因。

第四条同时解释了 PromptCache 一类"非前缀模块复用"方案为什么必须付出精度代价——SGLang 论文在相关工作里点名了这一点："PromptCache [12] proposes the modular reuse of the KV cache beyond the prefix but can impact accuracy by up to a 43% drop."(arXiv:2312.07104，§7)。本仓不涉及该路线。

### 2.2 三条贯穿全篇的公理

- **公理 A（命中判定在 token id 层，不在文本层）**：radix 树的键是 token id 序列 + extra_key + cache_salt(radix_cache.py：59)。任何"看起来一样的文本" 只要渲染出的 token 序列差一位，从第一处 diff 起全部 miss。EXP-P02 的五格矩阵就是把这条公理逐格钉死的。
- **公理 B（缓存容量的单位是 token，不是请求）**：池位按 token 计（`--max-total-tokens`），逐出按 token 补缺口（common.py：134-138）。"热集只有 6 个前缀"不是容量陈述，"热集 12900 token"才是。
- **公理 C（收益上限 = prefill 占比 × 前缀占比）**：命中省掉的是 prefill， 省不掉地板 $C$。0.6B 与 8B 在同一协议下差出两倍多的降幅，原因全在这条（§3.3）。任何不带模型的"前缀缓存能提速 X%"都缺一个定语。

### 2.3 日常类比与它的两处失效点

像连锁咖啡店把"每天都要做的糖浆底"提前熬好一大锅，每杯只现做上面的部分。类比在两处失效：①糖浆差不多就行，KV 复用要求 token 序列**逐位相同**，"差一个字"就整段作废（见 §3.2 的三层变换陷阱）；②糖浆锅容量不够时可以少熬点，KV 池不够时 LRU 逐出会让命中率**阶跃归零**而不是按比例下降（见 §3.5 的重用距离模型， EXP-P05《逐出压力》实测）。

第二处失效点值得多说一句：它不是 SGLang 的实现缺陷，而是"循环工作集 + LRU" 这一对组合的经典病态。Sleator 与 Tarjan 证明了确定性在线分页算法的竞争比下界就是缓存大小 $k$，而 LRU 恰好达到这个下界(CACM 28(2)：202-208, 1985)——达到下界意味着**存在**让 LRU 每次都 miss 的访问序，轮转访问 $k+1$ 个条目就是那个序。本仓的 EXP-P05 不是"碰巧测出了坏结果"，而是照着这个最坏例构造出来的。

### 2.4 RadixAttention 的选型:三种候选结构的代价表

要复用就要能查"新请求和历史请求的最长公共前缀是谁、它的 KV 在哪"。候选三种：

| 结构 | 查询能力 | 空间 | 插入/逐出 | 失效场景 |
|---|---|---|---|---|
| 整段 prompt 哈希 | 只有全等命中 | $O(1)$/条 | 简单 | 前缀差一个 token 即全 miss；无部分命中 |
| 逐 token trie | 任意前缀 | $O(\sum n_i)$ 节点 | 简单 | 节点数爆炸，指针开销远大于载荷 |
| radix tree（压缩前缀树） | 任意前缀 | 节点数 $O(\#\text{分叉})$ | 需分裂/合并 | 树维护与并发锁复杂度 |

SGLang 论文对第三项的说明是："Unlike typical trees， the edges of a radix tree can be labeled not just with single elements but also with sequences of elements of varying lengths， significantly enhancing efficiency."（arXiv:2312.07104，§3 "RadixAttention" 小节）。同一段还写明了两件本仓要反复用到的实现事实： KV 张量"are stored in a non-contiguous， paged layout， where the size of each page is equivalent to one token"；以及逐出策略是"a simple LRU eviction policy that evicts the least recently used **leaf** first. By evicting leaves first， we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted."（着重号为本讲义所加）。**"逐出叶子"这四个字是 §3.5.2 整节的出发点**——它看起来只是一个实现细节，实际上决定了共享前缀的存活语义。

### 2.5 三层"命中"必须分开

| 层 | 命中的含义 | 观测口径 | 本仓实例 |
|---|---|---|---|
| 路由层 | 请求被送到"可能有这段缓存"的副本 | router 近似树的 match_rate | 讲义 02 §3.3 |
| 引擎层 | radix 树真的匹配上了 $k$ 个 token | `cached_tokens` / device_hit counter | §3.4 |
| 效果层 | 这 $k$ 个 token 真的省下了毫秒 | TTFT 曲线 + 反例臂 | §3.3 |

三层同向但不等价：路由命中不保证引擎命中（两棵树、两种粒度），引擎命中不保证效果显著（地板 $C$ 占比大时，省下的 prefill 淹没在 $C$ 里）。本篇只处理后两层， 第一层留给讲义 02。

## 3. 完整推导与机制

### 3.1 树结构与 token 级契约,以及 n−1 上限的推导

服务端为每个请求构造 `RadixKey`(radix_cache.py：59)：**token id 序列 + extra_key（LoRA id 等）+ cache_salt**。树下行用 `child_key`(radix_cache.py：217) 做字典键，salt/extra_key 直接编进键——所以不同 salt 的两个请求即使 token 逐位相同也是**硬 miss**（命名空间隔离，EXP-P02 的 salt_diff 格实测 B 发 cached 缺失 = 0）。节点 `TreeNode`(radix_cache.py：238)持有 `value`（该段 token 对应的 KV 块索引张量）、`lock_ref`（在用计数，>0 时禁止逐出）、`last_access_time`（LRU 依据）。匹配入口 `match_prefix`(radix_cache.py：376)先做 page 对齐（page_size=1 时无损），再沿树下行，命中落在节点中段时把节点分裂（`_split_node`，radix_cache.py：704）以暴露精确边界。

#### 3.1.1 RadixKey 的三个分量与命名空间代数

把 RadixKey 写成三元组 $(\mathbf{t}, e, s)$：token 序列、extra_key、cache_salt。匹配是"在同一 $(e,s)$ 命名空间内对 $\mathbf{t}$ 求最长公共前缀"。上游把这条语义写进了 `match_prefix` 的 docstring(radix_cache.py：376-390)：

> The logical namespace for prefix matching is determined by both the token id sequence and the optional `extra_key` carried by `RadixKey`. Entries that share identical leading token ids but have *different* `extra_key` values are intentionally kept disjoint and never share prefix nodes.

三条推论（本讲义推导）：

1. **隔离是硬的，不是软的**：不同命名空间不是"匹配长度变短"，而是根本不进同一棵子树——`child_key` 把 $(e,s)$ 编进了字典键（radix_cache.py：217）。所以 EXP-P02 的 salt_diff 格测到的是 0 而不是"小一点的命中"。
2. **salt 是一个可用的实验旋钮**：想在同一负载上制造"人为 miss"的对照臂， 改 salt 比改 token 序列干净——token 序列不变意味着计算量不变，唯一变化的是命中与否。
3. **salt 的安全语义是时延侧信道防护**：命中带来的 TTFT 差可以被外部观察者用来探测"别人是否问过同样的前缀"，salt 提供硬隔离。这一条属于机制推断，本仓没有做侧信道实测，标注为未核实。

#### 3.1.2 n−1 上限,逐步推(每步一行"为什么")

1. 解码第 1 个输出 token，需要"输入最后一个 token 位置"的 logits。——采样定义如此：logits 是下一 token 的分布。
2. logits 来自当次 forward 对该位置的 hidden state 经 lm_head 投影。——缓存里存的是 KV，不存 logits，也不存 hidden state。
3. 若输入全部 $n$ 个 token 的 KV 都来自缓存、本次 forward 一个位置都不算， 则最后位置的 hidden state 不存在，没有 logits 可采样。——复用跳过的是计算，被跳过的位置不产生任何本次前向的中间量。
4. 所以调度器必须强制至少重算 1 个 token：匹配上限 = input_len − 1（`_compute_max_prefix_len`，schedule_batch.py：1411-1416，§4 走读第 6 段）。

**为什么不能"只算 lm_head"绕过第 3 步**（本讲义推导）：lm_head 的输入是最后一层的 hidden state，而 hidden state 不在缓存里——缓存只存每层的 K 与 V。要拿到 hidden state 就得跑完整的 28/36 层前向，而跑前向就必须给最后位置算一次 attention，也就必须为它准备 Q、算 QK、算 softmax·V。"重算 1 个 token"的成本因此不是零，而是"一个 token 的完整前向"——在 8B 上约 0.1 ms 量级（按 §3.3.2 的每 token 算术账折算，本讲义推导）。相对 2048 token 的 prefill 可以忽略，但它解释了为什么上限是 $n-1$ 而不是 $n$。

**第二个分支的代价**：`_compute_max_prefix_len` 的第二个分支把上限进一步压到 `logprob_start_len`。语义是：要"从第 $j$ 个位置起返回 logprob"，就必须从第 $j$ 个位置起真的算，缓存复用与 logprob 是此消彼长的。做评测（需要逐 token logprob） 时前缀缓存的收益会被这条压掉——这是一个容易在"为什么线上快、评测慢"里被忽略的机制。

**实测锚**：EXP-P01《env 与单 worker smoke》同一请求第二发 `cached_tokens=1324, prompt_tokens=1325` (raw=data/raw/EXP-P01/20260824T162947_probe_cached.json)，恰为 $n-1$ 的 **精确值**而非近似；engine 侧 `cache_hit_rate=0.9992`(= 1324/1325)。这是 "从源码读出上限 → 测量精确落在上限上"的最小闭环样本。

#### 3.1.3 page_size > 1 的对齐折扣(本讲义推导)

`match_prefix` 在下行前做 `key.page_aligned(self.page_size)` (radix_cache.py：419)，`page_aligned` 的实现是 `(matched_tokens // page_size) * page_size`(radix_cache.py：215)——**向下取整到页边界**。于是命中长度从 $k$ 变成 $\lfloor k/p \rfloor \cdot p$，损失 $k \bmod p$ 个 token。

若把 $k \bmod p$ 视作在 $\{0,\dots,p-1\}$ 上近似均匀，期望损失为 $(p-1)/2$ 个 token，相对损失 $\approx (p-1)/(2k)$。取 $p=16$、$k=1792$：期望损失 7.5 token， 相对 0.42%——可忽略。但取 $p=64$、$k=100$（短前缀、大页）：期望损失 31.5， 相对 31.5%——不可忽略。**结论：页粒度的对齐折扣只对短前缀致命**，这正是 "长系统提示词"场景下 vLLM 的 16-token block 与 SGLang 的 page_size=1 差别不大、而在长尾短前缀上差别明显的原因（本讲义推导；本仓 page_size=1 全程，未测 $p>1$， 该折算未经本机验证）。

### 3.2 chat template 三层变换陷阱:thinking 证伪案例全程复盘

radix 命中判定发生在 token id 层，而 OpenAI 接口收到的是消息列表，中间隔着三层变换（theory/03），任何一层不一致，第一处 diff token 起全部 miss：

1. **chat template 渲染**：`/v1/chat/completions` 把 messages 走 `apply_chat_template(..., add_generation_prompt=True)` 再 encode (serving_chat.py：1335-1345)。role 标记、system 头都进 token 流。绕过法：请求直接给 `input_ids`，模板整段跳过（serving_chat.py：1125-1134）——本仓收益实验的正式形态。
2. **thinking 开关**(Qwen3)：`enable_thinking` 经 `chat_template_kwargs` 进入模板（protocol.py：958-971 把 thinking 布尔写进 ctk；请求侧与服务端默认合并见 serving_chat.py：1053-1060）。
3. **命名空间**：`cache_salt`(protocol.py:867)与 LoRA extra_key (schedule_batch.py:928-929)进 RadixKey。

第 2 层是本仓最完整的一次**预注册证伪**，按时间序复盘（EXP-P02）：

- **预注册**（跑前锁定，写在 docs/PLAN.md#exp-p02 与脚本 docstring）： "thinking 开关不一致 → 模板从 system 段分叉 → hit ≪ base"。依据是想当然的推断：开关是模板参数，模板参数应该改模板头部。
- **实测**：thinking_flip 格 B 发 `cached=1326 / prompt=1329` (raw=data/raw/EXP-P02/20260824T163438_contract_matrix.json)——接近全命中， 与"hit ≪ base"直接矛盾。**假设证伪。**
- **CPU 复核**（不碰 GPU，纯 tokenizer 渲染对比）：thinking-on 渲染 1325 token， thinking-off 渲染 1329 token，**首分叉位 = 1325**，off 尾部多出 `<think>\n\n</think>\n\n` 恰 4 个 token (raw=data/raw/EXP-P02/20260824T163438_template_divergence.json)。即：off = on 的完整渲染**原样 + 尾部追加**，前缀共享完好。
- **修正**：theory/01 §5 与 theory/03 §2 当场改写；结论限定 Qwen3 模板族。
- **意外收获**：1326 = 1325 + 1，多出的 1 个 token 是 A 请求的**首个输出 token** (`<think>`)——因为树在请求结束时插入的是 input+output **全序列** (`cache_finished_req`，radix_cache.py：458-513，`token_ids = (req.origin_input_ids + req.output_ids)`)，B 的第 1326 个 token 恰好咬上。一个 off-by-one 现象反向证实了一条独立机制。

方法论提炼：错误的预注册假设 + 忠实测量 + 一次廉价的分层复核（CPU 渲染）， 比"猜对了"教得更多。这也是为什么收益实验（P03 起）全部改用 input_ids 直传： 让"共享前缀长度"成为实验的自变量，而不是被模板渲染出来的因变量。

#### 3.2.1 一般化:"文本相同而 token 不同"的成因清单

把 EXP-P02 的一次教训抽象成可复用的检查表（前三条本仓已逐格实测，后四条为机制推断，标注）。任何一条都会让命中从第一处 diff token 起归零：

| # | 成因 | 是否本仓实测 | 分层复核办法 |
|---|---|---|---|
| 1 | chat template 的 role 标记/system 头 | 是（EXP-P02 base 格） | CPU 侧 `apply_chat_template` 后 encode，逐 token diff |
| 2 | 模板参数（如 `enable_thinking`） | 是（thinking_flip 格，证伪） | 同上，重点看**首分叉位** |
| 3 | cache_salt / LoRA extra_key | 是（salt_diff 格） | 直接读请求体，不必跑模型 |
| 4 | tokenizer 版本或 `add_special_tokens` 差异 | 否（推断） | 固定 tokenizer revision，两侧 encode 对拍 |
| 5 | 客户端拼接的空白/换行差异 | 否（推断） | 对渲染字符串做 `repr()` diff 而不是肉眼看 |
| 6 | 中间层（网关/router）重序列化请求体 | 是（EXP-P06 首轮事故，见讲义 02） | 对响应内 `prompt_tokens` 设硬 gate |
| 7 | 多轮对话中历史被截断/摘要 | 否（推断） | 记录每轮实际发出的完整 token 序列 |

**这张表的读法**：第 1-5 条是"发出去的 token 就不同"，第 6 条是"发出去的和你以为的不同"，第 7 条是"你自己把前缀改了"。三类的排查成本差一个数量级——第 6 类最危险，因为请求和响应看起来都正常（讲义 02 §4 第 8 段的完整事故）。

### 3.3 收益账:省掉的 prefill 值多少毫秒

**算式**（把 TTFT 拆开）：

$$\mathrm{TTFT} \approx C + S_{\mathrm{prefill}}(n_{\mathrm{recompute}}) + W_{\mathrm{queue}}$$

其中 $C$ 是与 prefill 无关的固定地板（请求解析、调度、首 token 解码、流式开销），$S_{\mathrm{prefill}}$ 随需重算 token 数近似线性（每 token 的 FLOP ≈ $2 P$，$P$ 为参数量，attention 项另计），$W_{\mathrm{queue}}$ 是排队项， 并发 1 时约为 0。命中 $k$ 个 token 后 $n_{\mathrm{recompute}} = n - k$， 理想收益即 $S_{\mathrm{prefill}}$ 按 $k/n$ 比例缩短。

#### 3.3.1 这个分解在什么条件下合法(本讲义推导)

三条前提，缺一条这个算式就不能用来做归因：

1. **$C$ 与 $k$ 无关**。若命中长度会改变解析或调度开销（例如超长请求触发分块 prefill 的路径切换），$C$ 就成了 $k$ 的函数，曲线的截距失去意义。本仓固定 total_len=2048、只扫 prefix_len，请求体大小几乎不变，该前提近似成立。
2. **$S_{\mathrm{prefill}}$ 对 $n_{\mathrm{recompute}}$ 近似线性**。严格说 attention 项是二次的：$O(n^2)$。在 $n=2048$、$d_{\mathrm{model}}=4096$ 的 8B 上，attention 项占总 FLOP 的比例约为 8%（§3.3.2 的账），所以线性近似的相对误差在 10% 量级以内——**这正是实测斜率能稳定的原因，也是它的误差来源**。
3. **$W_{\mathrm{queue}} \approx 0$ 只在并发 1 成立**。并发 8 的曲线必须另配一条斜率，不能拿并发 1 的斜率外推（§3.3.3）。

#### 3.3.2 prefill 的算术账与硬件下界(硬件语义层,本讲义推导)

先立硬件端点。NVIDIA 的性能指南把"算术强度"定义为算法的操作数与访问字节数之比，并把处理器侧的对应量叫 ops：byte("GPU Performance Background User's Guide"，§4 Understanding Performance)。RTX 4090 的两个端点取自 NVIDIA Ada GPU Architecture 白皮书 Appendix A Table 2：显存带宽 1008 GB/s， BF16 Tensor（FP32 累加）峰值 165.2 TFLOPS（非稀疏）。故 ops：byte $\approx 165.2\times10^{12} / 1008\times10^{9} \approx 164$ FLOP/B。

**Qwen3-8B 每 token 的 prefill FLOP**（逐项，便于核对）：

| 项 | 形状 | 参数量 | FLOP/token（=2×参数） |
|---|---|---|---|
| q_proj | 4096×4096 | 16.78 M | 3.36e7 |
| k_proj | 4096×1024 | 4.19 M | 8.39e6 |
| v_proj | 4096×1024 | 4.19 M | 8.39e6 |
| o_proj | 4096×4096 | 16.78 M | 3.36e7 |
| gate/up/down | 4096×12288 ×3 | 150.99 M | 3.02e8 |
| 单层合计 |— | 192.94 M | 3.86e8 |
| ×36 层 |— | 6.95 G | **1.39e10** |

attention 本体另计：每层每 token 平均 $2\times 2\times (S/2)\times H\times D$ $= 2\times2\times1024\times32\times128 \approx 3.36\times10^7$ FLOP（取 $S=2048$ 的平均前缀长 $S/2$），×36 层 $\approx 1.21\times10^9$。合计每 token $\approx 1.51\times10^{10}$ FLOP，其中 attention 占 8.0%——**这就是 §3.3.1 第 2 条 "线性近似误差 10% 量级"的来源**。

**下界折算**：$1.51\times10^{10} / 1.652\times10^{14} \approx 91\ \mu s/\mathrm{token}$。实测斜率（§3.3 曲线）是 **98 µs/token**。

**这里必须诚实地记一条冲突**：98 / 91 意味着实测已达到白皮书峰值的约 93%， 而稠密 GEMM 在消费级卡上通常达不到这个比例。至少三种可能：①白皮书 Appendix A Table 2 的 165.2 TFLOPS 不是这一 kernel 组合的适用峰值（例如实际走的累加精度或时钟档与该数字的假设不同）；②本讲义的 FLOP 计数偏高（例如 attention 项的平均长度取法过于粗糙）；③"斜率"本身并不等于"纯 prefill 的每 token 时间"——斜率是 TTFT 的差分，里面可能吸收了随前缀长度变化的其它开销。**本仓没有 kernel 级测量，无法判定是哪一种，标注为未核实**。可以确定的只有方向性结论：8B 的 prefill 在本机上运行在算术界附近，kernel 侧几乎没有可压缩空间，**唯一的大杠杆是把计算整个跳过**——这正是前缀缓存的立足点。

**对照 decode 一步**：decode 读全部 KV，算术强度只有个位数 FLOP/B（与算术界 164 差两个数量级），是带宽/延迟受限。所以同一条请求的两个阶段落在 roofline 的两端：**prefill 靠算力，decode 靠带宽**；前缀缓存只对前者有效。这解释了 SGLang 论文里的那句观察："The speedup is more noticeable for short outputs because KV cache reuse mostly helps reduce the prefix time. For long outputs， because there is not much sharing between different chat sessions and the decoding time dominates， there is almost no speedup."(arXiv:2312.07104，§6.2)。本仓的 output_len 固定 32，正落在"短输出"这一侧，收益因此显著——**这是一个负载定语， 不是模型能力**。

#### 3.3.3 实测曲线对账(总长 2048,扫 prefix ∈ {0,512,1024,1536,1792},3 seeds)

- Qwen3-8B（EXP-P07《8B 收益曲线》，data/derived/exp_p07_8b_ttft_vs_prefix.csv）：并发 1 的 TTFT p50 从 228.4±3.9 ms(prefix=0)降到 52.9±0.5 ms(prefix=1792)， **−77%**；并发 8 从 1068.3±13.3 降到 234.5±4.6 ms，**−78%**。斜率账：$(228.4-52.9)/1792 \approx 98\ \mu s/\mathrm{token}$（并发 1）。前缀占比 1792/2048 = 87.5%，实际只省 77%——差额就是 $C$：按线性外推， 纯 prefill 剩 $228.4 \times 256/2048 \approx 28.6$ ms，实测 52.9 ms， 余量 ~24 ms 即固定开销与首 token 解码（EXP-P07 §6 的"余量"口径）。
- Qwen3-0.6B(EXP-P03，data/derived/exp_p03_ttft_vs_prefix.csv)：同协议仅 −36%（并发 1,26.84→17.27 ms）/ −63%（并发 8,115.14→42.73 ms）。 0.6B 的 miss TTFT 只有 ~27 ms，里面 ~17 ms 是地板 $C$——prefill 在 TTFT 中占比小，可被跳过的部分就小，**收益天花板正比于 prefill 占比**。模型量级账（推断，标注）：参数量 8B/0.6B ≈ 14×，实测 miss TTFT 之比 228.4/26.84 ≈ 8.5×，同量级；没到 14× 是因为 0.6B 的 TTFT 被 $C$ 垫高。
- **并发放大**（实测→机理推断，EXP-P03 §6）：0.6B 的每命中 token 收益斜率从并发 1 的 5.3 µs/token 升到并发 8 的 40 µs/token(×7.6)。机理：并发下 prefill 算力是队列瓶颈，省掉一条请求的 prefill 同时缩短了**其余请求的排队**（$W_{\mathrm{queue}}$ 项），收益复利。"低并发测不出缓存价值"的根源在此。

**并发放大的排队论解释**（本讲义推导，把机理接到标准结论上）：Kingman 对 G/G/1 队列的重流量近似给出等待时间的乘积形式(Kingman， "The single server queue in heavy traffic"， Proc. Cambridge Phil. Soc. 57(4)：902-904, 1961)：

$$E[W_q] \approx \left(\frac{\rho}{1-\rho}\right)\cdot\left(\frac{c_a^2+c_s^2}{2}\right)\cdot E[S]$$

其中 $\rho = \lambda E[S]$ 是利用率。前缀命中把服务时间 $E[S]$ 降到 $\alpha E[S]$($\alpha<1$)，于是 $\rho \to \alpha\rho$，等待项被 **两次**放大地压缩：一次来自 $E[S]$ 本身的线性下降，一次来自 $\rho/(1-\rho)$ 这个在高 $\rho$ 下急剧非线性的因子。这就是"5.3 → 40 µs/token" 的形状来源。

**但必须标注一处口径不匹配**：Kingman 的式子是**开环**模型（到达率 $\lambda$ 外生），而本仓的并发定义是客户端信号量限流（§4 第 3 段），在途请求数恒 ≤ 8， 属于**闭环**系统。闭环系统的正确框架是交互式响应时间律 $R = N/X - Z$（Lazowska et al.， *Quantitative System Performance*， Prentice-Hall， 1984，"Fundamental Laws" 一章），它是 Little 律 $L=\lambda W$ (Little， "A Proof for the Queuing Formula： L = λW"， Operations Research 9(3)：383-387, 1961)在交互式系统上的形式。闭环下吞吐会自然饱和，响应时间随 $N$ 线性上升而不是随 $\rho\to1$ 发散。所以 Kingman 只能用作**定性**解释（为什么放大），不能用来**定量**预测放大倍数——本仓 ×7.6 这个数字是实测， 不是理论预测。这一点在讲义 02 的 p50/p99 讨论里还会再用到。

**反例臂**：`--disable-radix-cache` 起服的 off 臂在 8B 下全线 229.7-231.8 ms 持平（exp_p07 csv 的 off 行），0.6B 下 26.59-26.87 ms 持平——收益只在 radix 开启且确有共享前缀时出现，归因成立。

### 3.4 逐 token 归因闭环:466,944 三方相等怎么做到

"TTFT 降了"与"命中了前缀"之间还差一步归因：降幅是否**恰好**由省掉的 prefill token 兑现？本仓用三个**互相独立**的计数源对账：

1. **客户端逐请求**：流式响应的 `usage.prompt_tokens_details.cached_tokens`（随流返回，与停表同一响应内闭合，§4 走读第 1 段）。逐请求硬校验： on 臂必须恰 = prefix_len，off 臂与 prefix=0 必须恰 = 0（§4 第 4 段）。
2. **引擎计数器**：`/metrics` 的 `sglang:prefill_effective_tokens_total{mode="device_hit"}` counter， 实测终值 **466,944.0**（raw=data/raw/EXP-P07/20260824T171520_8b_radix_on_metrics.txt， 0.6B 的 EXP-P03 同值复现）。
3. **协议期望**（纯算术）：每点 16 请求，Σ over prefix∈{512,1024,1536,1792}： $16 \times (512+1024+1536+1792) = 16 \times 4864 = 77{,}824$； × 3 seeds = 233,472；× 2 个并发臂（c1+c8）= **466,944**。prefix=0 点贡献 0。

三方逐 token 相等（0.6B/8B 双复现，EXP-P03/P07 §6）意味着：没有一个 token 的命中是虚报的，也没有协议外的意外命中（预热残留、跨点污染都会破坏等式）。这就是"降幅确实且仅由省掉的 prefill 兑现"这句话的证据形态。

#### 3.4.1 这个闭环证明了什么、没有证明什么

**证明了（账目层面）**：三条计数链路各自独立——客户端读的是响应体字段，引擎 counter 走的是调度器内部统计（metrics_reporter.py：646-657），协议期望是纯算术——三者相等排除了三类故障：虚报（引擎多算命中）、漏报（客户端少收）、污染（树里有协议外的残留）。任何一类发生，等式都会破。

**没有证明（因果层面）**：等式说的是"命中了 466,944 个 token"，不是"这 466,944 个 token 就是 TTFT 降幅的原因"。严格的因果分解需要 server 侧 per-request 时序（排队时刻、prefill 完成时刻），本仓没有采集。本仓用的是替代组合：①**反例臂**（disable-radix 起服）排除"降幅来自别的机制"；②**曲线的线性形态**（斜率在四个前缀点上稳定）说明降幅随命中量单调且比例稳定； ③**计数闭环**排除账目错误。三者合起来是一个强的观测性证据，但仍不是随机化干预实验。这一点 theory/03 §3 明示过：不要拿 client TTFT 直接说"prefill 变快了"。

**一个具体的可识别性漏洞**（本讲义推导）：假设存在某个与前缀长度相关但与 prefill 无关的机制（比如更长的公共前缀让某个哈希表更容易命中，进而加快请求解析），它会同时进入 $C$ 与斜率，而反例臂无法把它分离——因为反例臂关掉的是 radix，不是"前缀相同"这个负载性质。要堵住这个漏洞，需要一个"前缀相同但 radix 命中被强制置零"的第三臂（上游有 `SGLANG_RADIX_FORCE_MISS` 环境变量，见 schedule_policy.py：349 的用法），本仓未做，列为可执行的下一格。

### 3.5 重用距离模型:完整推导与三池验证

前缀缓存何时失效？KV 池是有限的，逐出策略默认 LRU（`--radix-eviction-policy` 默认 lru，server_args.py：919-931；策略实现 evict_policy.py：16-18 按 `last_access_time` 建最小堆，radix_cache.py：592 的 `evict` 只逐出缺口那么多，common.py：114-138）。

#### 3.5.1 经典栈距离:定义、定理与它要求的前提

Mattson 等人在 1970 年给出了一次遍历就能算出所有缓存大小命中率的方法 (Mattson， Gecsei， Slutz， Traiger， IBM Systems Journal 9(2)：78-117, 1970)。核心概念是**栈距离**（stack distance，后来常称 reuse distance）：对某次访问， 距离 = 自上次访问同一条目以来，被访问过的**不同条目**的个数。核心定理：

> 对满足**包含性质**(inclusion property)的替换算法（LRU、LFU、MIN 都满足）， 容量为 $C$ 时驻留的条目集合总是容量 $C+1$ 时驻留集合的子集；因此一次访问在容量 $C$ 下命中 $\iff$ 它的栈距离 $< C$。

两个直接推论，后面都要用：

- **命中率随容量单调不减**。所以"加大池反而更差"（Belady 异常，Belady， Nelson， Shedler 1969 对 FIFO 的构造）在 LRU 上**不可能发生**。EXP-P05 的三池数据（8192 崩、16384 部分崩、默认全保）与这条一致——这不是巧合，是定理。
- **命中率曲线可以一次遍历算全**。这意味着"该配多大池"原则上不需要逐档实测， 只要采到访问序列就能算出全曲线。本仓没有实现这套工具（需要 engine 侧逐访问 trace），列为扩展。

**这条定理要求的前提是条目等大**。KV 池里的条目是 radix 树节点，长度可变（一个节点可以是 1536 个 token 的前缀段，也可以是 8 个 token 的输出尾巴）。所以直接套用会出错，§3.5.3 给出变长版本。

#### 3.5.2 树形 LRU 的等价性:为什么"只逐出叶子"不是限制(本讲义推导)

SGLang 的逐出不是在一个平坦条目集合上做 LRU，而是**只在叶子上**做： `evict` 从 `self.evictable_leaves` 建堆（radix_cache.py：598-602），弹出最小 priority 的叶子，删掉之后如果父节点变成无子且 `lock_ref == 0`，再把父节点推进堆（radix_cache.py：613-615）。论文对此的表述是"evicts the least recently used **leaf** first"(arXiv:2312.07104，§3)。

一个自然的疑问：**这个"只看叶子"的限制，会不会让实际逐出顺序偏离真正的 LRU？**

**命题（本讲义推导）**：在 SGLang v0.5.18 的实现下，`last_access_time` 的全局最小值（几乎总是）落在某个叶子上，因此"叶子 LRU"与"全体节点 LRU"给出同一个逐出对象。

**证明要点**：关键在于时间戳的**路径刷新**语义。`_match_prefix_helper` (radix_cache.py：678-702)在下行时对**根节点与路径上每一个子节点**写同一个 `access_time`；`_insert_helper`(radix_cache.py：737-760)做同样的事。于是对任意节点 $v$ 与它的任意后代 $u$：每一次触达 $u$ 的操作都必然穿过 $v$，并把 $v$ 的时间戳刷新到该次操作的时刻。所以

$$\mathrm{last\_access}(v)\ \ge\ \mathrm{last\_access}(u)\quad \text{对所有 } u \in \mathrm{subtree}(v)\setminus\{v\}$$

——**除了一个例外**：在同一次 `_insert_helper` 调用里新建的子节点，其时间戳来自 `TreeNode.__init__` 里独立的一次 `time.monotonic()`(radix_cache.py：248)， 采样时刻**晚于**父节点被写入的 `access_time`。此时父节点的时间戳会比这个新子节点小几微秒。但这种父节点按定义不是叶子（它刚获得了一个孩子），不进入 `evictable_leaves`；而它的这个唯一新子节点是该子树里最新的条目，只有在整棵子树都被逐出后才轮到它。所以在"谁先被逐出"的排序上，这个例外不改变结果。**证毕**（在"忽略同一次下行内的微秒级时间戳差"的意义下）。

**这条命题的三个用处**：

1. **可以放心地把 SGLang 的 KV 池当成一个普通 LRU 缓存做容量规划**——§3.5.1 的经典结论（包含性质、无 Belady 异常、栈距离判据）可以直接搬过来。
2. **它解释了共享前缀为什么"自动被保护"**：一个被多条请求共享的前缀节点， 会被**每一条**经过它的请求刷新时间戳，所以它的时间戳总是等于最近一次任意共享者的访问时刻。热度是自动聚合的，不需要额外的引用计数机制来"保热"。
3. **它把逐出的粒度讲清楚了**：一次 `evict` 会把"最老的那条路径"从叶子往上逐层剥掉（每剥掉一个叶子，父节点若变成叶子就重新入堆），直到补上缺口为止。所以逐出的自然单位是**一条从叶子往上的链**，不是单个 token。

#### 3.5.3 变长条目下的重用距离:哪一半是定理,哪一半靠构造(本讲义推导)

把 §3.5.1 的栈距离改写成 token 计价。定义热前缀 $h$ 的**加权重用距离** $D_w(h)$ = 两次访问 $h$ 之间，被访问或新插入的、与 $h$ 不同的条目的 token 总量。设 $|h|$ 为 $h$ 自身的 token 数，池容量 $P$。

**方向一（定理）**：若 $D_w(h) + |h| \le P$，则第二次访问命中。 *理由*：LRU 只会逐出比 $h$ 更旧的条目；比 $h$ 新的条目总量加上 $h$ 自己都装得下，说明分配请求从未需要动到 $h$。

**方向二（不是定理，需要构造兜底）**：$D_w(h) + |h| > P$ 时**未必** miss。两个反例来源：①逐出只补缺口（common.py：135-138 明写 "evict only the shortfall"），超出部分可能恰好由比 $h$ 更旧的"压舱条目"吸收；②节点是变长的， 逐出一个大节点可能一次性释放远超缺口的空间，后续若干次分配都不再触发逐出。所以要断言 miss，必须论证"比 $h$ 更旧的压舱量不足以吸收累计逐出需求"。

**结论**：EXP-P05 里写的"LRU 命中 ⇔ 池 ≥ D"里，$\Leftarrow$ 是定理， $\Rightarrow$ 靠实验构造保证（冷流量全唯一、串行注入、轮转访问，使压舱量在稳态下被耗尽）。这个区分不是吹毛求疵——它正是"为什么同样的模型在真实流量上会看到软化曲线"的原因（§3.5.6）。

#### 3.5.4 逐段账:EXP-P05 的精确 token 收支(本讲义推导)

先看实验构造（EXP-P05，bench_evict.py）：$H=4$ 个热前缀，prefix_len=1536， total_len=2048，输出 8 token；每条热请求后跟 $c$ 条全唯一冷请求；串行并发 1。 $D = H \cdot T \cdot (1+c) = 8192\times(1+c)$，故 $H\cdot T = 8192$ 恰为最小池位。

现在把"每条请求往池里新写多少 token"逐项算清：

| 请求类型 | 输入 token | 其中已在树里 | **新增** token（含 8 输出） |
|---|---|---|---|
| 预热请求（每热前缀 1 条） | 1536 + 32 | 0 | 1576 |
| 计时热请求 | 1536 + 512 | 1536 | **520** |
| 冷请求 | 2048 | 0 | **2056** |

于是同一热前缀两次访问之间，新写入池的 token 量为

$$\Delta = H\cdot 520 + H\cdot c\cdot 2056 = 2080 + 8224c$$

再按 §3.5.3 的方向一，把"比 $h$ 更新的一切"加上 $h$ 自己：

$$D_w(h) + |h| = \underbrace{3\times1536}_{\text{另外 3 个热前缀}} + \underbrace{2080 + 8224c}_{\Delta} + \underbrace{1536}_{h} = 8224\,(1+c)$$

**对照粗模型**：粗模型给 $8192(1+c)$，精算给 $8224(1+c)$——相差 0.4%，来源就是每条请求那 8 个输出 token。**粗模型在这个构造下几乎是精确的**，这一点值得强调： $H\cdot T$ 之所以能当 $D$ 用，是因为"热请求少写的部分（前缀已在树里）"恰好被 "输出 token 多写的部分"抵消掉了。换一个 prefix_len/output_len 组合，两者就不再抵消，粗模型会偏。

**一个必须解释的边界格**：$c=0$ 时精算给 $8224 > 8192 = P$，按方向一的条件 **不能保证命中**；而实测是 1.0000（全命中）。这不是矛盾，而是 §3.5.3 方向二不成立的实例。机制如下（本讲义推导）：

1. 稳态下树里有 4 个热前缀节点（6144 token）+ 最近几条热请求的后缀节点（每条 520）；
2. 当第 5 条热请求要分配 520 个新位而只差 32 个时，`evict_from_tree_cache` 只请求逐出 32 个（common.py：138）；
3. `evict` 弹出的最小 priority **叶子**是最老的那个后缀节点（520 token）， 一次就释放 520 ≥ 32，循环立刻结束（radix_cache.py：605 的 while 条件）；
4. 热前缀节点此刻**不是叶子**（它们各自还挂着后缀子节点），根本不在堆里； 即使成为叶子，它们的时间戳也比后缀节点新（§3.5.2 的路径刷新）。

**所以 $c=0$ 保命中的真正原因不是"$D$ 恰好等于 $P$"，而是"溢出量只有 32 token， 而可供牺牲的后缀压舱有 4×520 = 2080 token"**。这是本讲义对仓内模型的一处细化： 粗模型的边界格给对了答案，但给的理由不完整。

**顺着这条精算，还能读出两个未测的预测**（本讲义推导，标注为预测）： 16384 池在 $c=1$ 时需要 $8224\times2 = 16448$，略超 16384 但溢出仅 64 token， 按上面同样的压舱逻辑**预测命中**；$c=2$ 时需 24672，远超 16384 且压舱不足， **预测崩塌**。这两格本仓未跑（EXP-P05 只在 16384 池测了 $c=4$），写在这里作为可证伪的下一步。

#### 3.5.5 为什么是阶跃而不是斜坡

四步推导（白板可复现，docs/talk/whiteboard_card_reuse_distance.md）：

1. **定义重用距离 $D$**：同一热前缀两次被访问之间，注入缓存池的 token 总量。——LRU 只看"最近"，所以决定一个条目存亡的正是"两次使用之间进来了多少"。
2. 轮转访问 $H$ 个热前缀、每条请求总长 $T$、每条热请求后跟 $c$ 条冷请求： 两次访问同一热前缀之间恰好流过 $H$ 条热请求与 $Hc$ 条冷请求， $$D = H \cdot T \cdot (1+c)$$——每条请求（热或冷）都把约 $T$ 个 token 写进池（树缓存 input+output）。
3. **LRU 命中条件**：池容量 $P \ge D$。——$P \ge D$ 时，热前缀在被重访之前不可能被挤出（比它更旧的都先走）； $P < D$ 时，它**必然**在重访之前被逐出（进来的量已超池容量）。
4. **预测阶跃而非斜坡**：轮转访问下所有热前缀的 $D$ 相同，条件对全体同时成立或同时破——命中率只有 1.0 与 ~0 两个稳态，没有中间态。这是"循环工作集 + LRU"的经典最坏搭配，与 CPU cache 的 LRU thrashing 同构。

第 4 步是整节的要害，值得再补一层理论出处：**轮转访问正是 LRU 的最坏输入**。 Sleator 与 Tarjan 证明确定性在线分页算法的竞争比下界是缓存大小 $k$，且 LRU 达到这个下界(CACM 28(2)：202-208, 1985，Theorem 6 及其配套下界论证)。达到下界的构造就是"循环访问 $k+1$ 个页面"：每次访问的都是刚被逐出的那个。本仓的 $H$ 个热前缀轮转 + 冷流量填充，是这个构造的 token 计价版本。**换句话说，EXP-P05 测到的不是 SGLang 的缺陷，是一条 1985 年就证明了的下界在 2026 年的 KV 池上的复现。**

#### 3.5.6 真实到达序会把悬崖抹平:Che 近似与它的适用范围

轮转访问是最坏例；真实流量不是。在独立引用模型（IRM）下，Che 等人提出的近似把 LRU 的命中率写成一个极简形式（Fricker， Robert， Roberts， arXiv:1202.3974， 该文给出了这个近似为何准确的数学解释）：对容量 $C$ 的 LRU 缓存，定义**特征时间** $t_C$ 为"恰好有 $C$ 个不同条目被访问所需的时间"，则条目 $i$ 的命中率近似为

$$h_i \approx 1 - e^{-\lambda_i t_C}$$

其中 $\lambda_i$ 是条目 $i$ 的请求速率。这个式子的形状是**光滑的 S 曲线**，不是阶跃——因为不同条目的 $\lambda_i$ 不同，越线时刻被拉开了。**这就是"为什么真实流量看不到悬崖"的定量解释**：悬崖需要所有热条目的 $D$ 完全相同，而 Zipf 式的流行度分布让每个条目各有各的越线点，叠加起来就是软化曲线。

**边界**：Che 近似假设 IRM（每次请求独立同分布），而 LLM serving 的到达序有强时间相关性（同一会话的多轮、同一租户的突发）。相关性会让实际命中率**高于** IRM 预测（局部性更好）。本仓两种流量形态都没测（合成负载是轮转，不是 IRM 也不是真实 trace），这一段是给读者的外推指南，标注为未在本机验证。

#### 3.5.7 实验构造与实测

**实验构造**(EXP-P05，bench_evict.py)：$H=4, T=2048$，故 $D = 8192 \times (1+cr)$；特意让 $H \cdot T = 8192$ = 最小池位，使 cr=0 时 $D$ 恰压在池边界上，cr 每 +1 把 $D$ 线性外推一个池位。三池（`--max-total-tokens` 8192 / 16384 / 默认 ≈16 万，161671，EXP-P01 启动日志） × 冷流量 cr ∈ {0,1,2,4} × 3 seeds，串行并发 1（隔离排队与 lock_ref 扰动）。

**实测**（data/derived/exp_p05_eviction_cliff.csv，seed 间 std 全为 0）：

| 池（token） | cr=0(D=8192) | cr=1(16384) | cr=2(24576) | cr=4(40960) |
|---|---|---|---|---|
| 8192 | **1.0000** | **0.0625** | 0.0625 | 0.0625 |
| 16384 | — | — | — | 0.125 |
| 默认 | 1.0000 |— |— | **1.0000** |

模型三池全符合：8192 池仅 cr=0（$D=P$，恰好）保命中；16384 池到 cr=4 ($D=40960>16384$)崩；默认池 $P \gg D$ 全保。残余 0.0625 = 1/16，即预热后的首个热请求（raw 里 `hot_cached` 序列是 `[1536, 0, 0, ...]`， data/raw/EXP-P05/20260824T164650_smallpool_cr1_s20260824.json）。仅边界格 16384@cr4 见 0.125 残余（首周期多存活一拍）。`evicted_tokens_total` 佐证： 小池随 cr 单调升（33.9 万 → 66.3 万），默认池全程无逐出（counter 不曝光， 按 0 记——EXP-P05 §7）。

**残余 0.0625 与 0.125 的机制读法**（本讲义推导）：0.0625 = 1/16 说明 16 条计时热请求里恰有 1 条命中，即预热种下的树只撑到第一条计时请求；之后每一条都在被重访前已被逐出——**稳态命中率是 0,1/16 是启动瞬态**。16384@cr4 的 0.125 = 2/16 说明瞬态多撑了一拍，与"池大一倍、压舱多一层"一致。两个残余值都不是"部分命中"， 而是"命中的请求条数"，读表时不要把它当作"命中了 6.25% 的 token"。

**工程含义**：容量规划按**热前缀重用距离**配池，不是按热集大小；冷流量占比把 $D$ 线性推过边界，是一阶变量。这条模型还是讲义 02 里路由实验（EXP-P06《路由 × 池容量》） 预注册预测的推导前提。

### 3.6 KV 池的硬件账:161671 这个数是怎么来的(硬件语义层)

池位是 token，但显存是字节。换算公式（本讲义推导）：

$$b = 2 \times L \times \mathrm{KVH} \times D \times \mathrm{sizeof(dtype)}$$

（2 = K 和 V 各一份）。代入本机两个 checkpoint 的 `config.json`：

| 模型 | $L$ | KVH | $D$ | dtype | $b$ | 每 token |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | bf16 | $2\cdot28\cdot8\cdot128\cdot2$ | 114,688 B = **112 KiB** |
| Qwen3-8B | 36 | 8 | 128 | bf16 | $2\cdot36\cdot8\cdot128\cdot2$ | 147,456 B = **144 KiB** |

两者之比恰为 36/28 = 1.286——因为 KV 头数与 head_dim 相同，**KV 显存只随层数线性增长，与模型总参数量无关**。这是一条常被搞错的直觉：8B 的参数是 0.6B 的 14 倍，但每 token 的 KV 只是 1.29 倍。

**反推池位**：EXP-P01 启动日志里的默认池 161,671 token，乘 112 KiB = 17.28 GiB。RTX 4090 有 24 GiB，减去 0.6B 权重（约 1.2 GiB bf16）、CUDA 上下文、激活与通信缓冲，17.3 GiB 的 KV 池与 SGLang 默认 `--mem-fraction-static` 的分配量级一致（本讲义折算；确切的分配公式未逐项核对，标注）。

**这条账的三个用处**：

1. **容量规划的第一步是算 $b$**，第二步才是算重用距离。同样的 `--max-total-tokens 8192`，在 0.6B 上是 0.875 GiB，在 8B 上是 1.125 GiB。
2. **它解释了为什么 EXP-P05/P06 用 0.6B 而不是 8B**：要制造"池装不下热集"的机理格，必须能把池压到很小；8B 的权重就占 16 GiB 左右，可调空间小，而且每 token 更贵，同样的 token 数占更多显存。用小模型做容量机理实验，是把变量隔离干净的正确做法。
3. **它给出了前缀缓存的显存收益上界**：$N$ 条请求共享 $L_p$ 的前缀，不共享时要存 $N\cdot L_p\cdot b$，共享后只存 $L_p \cdot b$，省 $(N-1)/N$。8B、 $L_p=1792$、$N=16$ 时省 $15/16 \times 1792 \times 144\ \mathrm{KiB} \approx 236$ MiB。**本仓没有测显存侧收益**（所有实验固定池大小，只看时延与命中），这个数字是折算，不是实测。

### 3.7 page_size = 1 凭什么可行:分页 KV 的 kernel 侧语义(硬件语义层)

SGLang 论文写明 KV"stored in a non-contiguous， paged layout， where the size of each page is equivalent to one token"(arXiv:2312.07104，§3)。一个 token 一页意味着 attention kernel 每读一个历史 token 都要过一次间接寻址。**为什么这不会把带宽打垮？**

答案在最后一维的连续性。FlashInfer 把各种 KV 布局统一成块稀疏矩阵，并明确写道： "The last dimension of the KV-Cache remains contiguous (with size of head dimension d， commonly 128 or 256)， maintaining coalesced memory access." (arXiv:2501.01005，§3.2.1)。代入本仓：head_dim = 128、bf16，单个（token， head） 的 K 或 V 是 $128\times2 = 256$ 字节的**连续**块。

对照 CUDA 的访存粒度语义：全局内存以 32/64/128 字节的对齐事务服务，且在 compute capability 6.0 及以上，数据访问单元是 32 字节（NVIDIA， "CUDA C++ Best Practices Guide"，Coalesced Access to Global Memory 一节）。256 字节 = 8 个 32 字节扇区，且天然对齐。**所以 page_size=1 付出的代价只是"多一次索引查找"， 而不是"访存变成非合并"**——真正决定合并度的最后一维始终连续。

FlashInfer 附录 B 给出了这条代价的量化：以 page size 1 的向量稀疏格式对比稠密 KV,"For decode kernels, the performance gap between sparse and dense KV-Cache is negligible (within 1%). For prefill kernels, there is approximately a 10% performance gap."（arXiv:2501.01005，附录 B）。

**读法与本仓的关系**（本讲义推导）：

- decode 侧 1% 的代价 → 逐 token 分页在解码路径上基本免费，这是 RadixAttention 能做到 token 粒度共享而不是 block 粒度的物理前提。
- prefill 侧 10% 的代价 → 命中越多、需要重算的 prefill 越少，这 10% 作用的基数也越小。本仓的收益曲线里包含这项代价（off 臂用的是同一个后端），所以 −77% 是**含分页开销**的净收益，不是理想化数字。
- 本仓用的是 flashinfer 后端（EXP-P01 启动日志），但**没有做 page_size 或后端的对照臂**，以上 1%/10% 是论文数字，不是本机实测，不能当作本仓结论。

### 3.8 魔法数总表:每个数字由谁决定

把本篇出现的所有"看起来随便定的数"归一次因（本讲义推导，依据列在最后一列）：

| 数字 | 出处 | 由谁决定 | 依据 |
|---|---|---|---|
| 匹配上限 $n-1$ | schedule_batch.py：1413 | **理论上界** | 采样需要最后位置的 logits(§3.1.2) |
| page_size = 1 | 本仓启动参数（默认） | **硬件约束允许** | 最后一维连续保证合并访存（§3.7） |
| KV 每 token 112/144 KiB | config.json 折算 | **硬件/模型结构** | $2LKVH\cdot D\cdot$sizeof(§3.6) |
| 池 8192 / 16384 | EXP-P05 协议 | **实测扫描的设计值** | 取 $H\cdot T=8192$ 使 $c=0$ 压边界（§3.5.7） |
| 池 161671（默认） | 启动日志 | **硬件约束** | 24 GiB 减权重与缓冲后按 $b$ 折算（§3.6） |
| total_len 2048 | 收益曲线协议 | **实测扫描** | 定总长扫前缀，使 prefix_len 成唯一自变量 |
| output_len 32 | 收益曲线协议 | **实测扫描** | 短输出放大 prefill 占比（§3.3.2 的负载定语） |
| 3 seeds | 全仓协议 | **统计口径** | 给轮间波动一个量尺，单轮微趋势才判得了平 |
| 16 请求/点 | 收益曲线协议 | **实测扫描** | 分位数索引近似在 16 样本下可用（§4 第 3 段） |
| 预热尾巴 8 个随机 token | bench_prefix.py：282 | **协议正确性** | 使预热请求不被计时请求全长命中（§4 第 2 段） |

**表的读法**：只有第 1、3、5 行是"不由人定"的——分别由采样语义、模型结构、显存容量决定。其余都是实验设计选择，换一个场景就该重选。把这两类混为一谈， 是"把别人的 benchmark 参数抄过来"这类错误的根源。

## 4. 代码逐段走读:scripts/bench_prefix.py 的停表与闭环

以下按"测量端 → 上游语义 → 缓存内部"的顺序走读（引用为逐字拷贝，标 文件：起-止行； 上游文件相对 /root/repos/sglang-v0.5.18）。

**第 1 段 · 单请求：停表与命中数在同一响应内闭合**(scripts/bench_prefix.py：43-68)

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

角色：全仓统一的 TTFT 停表口径就定义在这 26 行里。三个关键选择： ①`input_ids` 直传（EXP-P02 契约结论）——token 序列完全受控，`cached_tokens` 才能与 prefix_len 逐 token 对账；②停表停在**首个非空 content delta** 到达客户端，不停在 HTTP 首字节（那只是 SSE 响应头，不含 token），也不用 server 侧直方图（要的是含排队+prefill+首 token 解码的用户可感知延迟）；③ `stream_options.include_usage` 让 usage 随流返回，"快了多少"与"命中了多少" 同源。改错会怎样：若停表停在首个 chunk 而不判 `delta.content`，会被 role-only 首 chunk 提前触发，TTFT 系统性偏小；若事后查 /metrics 取命中，并发下无法归属到单个请求，闭环校验（第 4 段）就做不成。

**这个口径与论文口径的差别**（本讲义推导）：SGLang 论文的 latency 是端到端的程序完成时间（arXiv:2312.07104，§6），vLLM 论文用的是 normalized latency（每输出 token 的平均端到端时延，arXiv:2309.06180，§6.1）。本仓用 TTFT， 因为被测机制（省 prefill）只作用在首 token 之前；用端到端会被 decode 段稀释， 用 normalized latency 会把结论变成"输出越短收益越大"这个同义反复。**换口径就换结论，这是读任何 serving 论文时首先要对齐的一件事。**

**第 2 段 · 测量点前置：flush 与预热**(scripts/bench_prefix.py：74-91)

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

角色：把"树里已有前缀"设成前置条件。flush 清掉上一测量点的树（点与点零残留， 否则 prefix_lens 的扫描顺序会污染结果）；预热发 1 条 [prefix + 8 随机尾] 请求把前缀灌进树（计时外）。改错会怎样：不 flush，则后一个点会命中前一个点的残树， cached 校验（恰=prefix_len）当场 FAIL；不预热，计时臂的首请求变成"替大家种树" 的 miss，分布被污染；预热尾巴不加随机 token，预热请求自己就会被计时请求全长命中，cached 会超出 prefix_len，同样被校验捕获。

**为什么预热必须是"前缀 + 唯一尾巴"而不是"只发前缀"**（本讲义推导）：树在请求 **结束**时插入 input+output 全序列（第 8 段）。如果预热只发 prefix 本身，插进树的是 `prefix + 预热的输出 token`；由于 `max_tokens=1`，输出只有 1 个 token， 而计时请求的第 prefix_len+1 个 token 是随机的，与那 1 个输出 token 相等的概率约为 1/99000(token 空间 range(1000,100000))——小，但不是 0，且一旦相等， 该请求的 cached 就会是 prefix_len+1，硬校验立刻 FAIL 成整格作废。加 8 个随机尾 token 把这个概率压到 $(1/99000)^8$，实际上消除了这个尾部风险。**这是"用构造消灭低概率事件，而不是用容差掩盖它"的一个具体样本。**

**第 3 段 · 并发定义与分位数**(scripts/bench_prefix.py：92-113)

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

角色：并发的操作性定义（客户端在途上限，server 侧 batch 是引擎自己的事）， 以及 raw 的输出契约——注意**逐请求数组**(`cached_tokens`、`ttft_ms`)整个落盘，聚合器才可能做逐请求校验；只存分位数的 bench 事后无法逐请求复核。改错会怎样：分位数若用插值定义，16 样本下与索引定义差一个次序统计量，但因两臂同法，相对比较不受影响——这是把"分位定义"从实验变量里消掉的做法。

**"客户端信号量 = 并发"这个定义的理论后果**（本讲义推导）：它把系统变成一个 **闭环**(finite population)队列：在途请求数恒定为 $N$，一条完成才放一条进来， 思考时间 $Z=0$。于是交互式响应时间律给出 $R = N/X$(Lazowska et al.， 1984， "Fundamental Laws")，即**响应时间与吞吐互为倒数关系**，而不是开环模型里的 "到达率逼近服务率时发散"。实际含义有两条：①并发 8 的 TTFT 不会因为负载过重而无界增长，它被 $N=8$ 钉住；②两臂比较时，"更快"同时意味着"吞吐更高"——两者不是独立的两个指标。读并发臂的数字时必须记住这个耦合。

**第 4 段 · 聚合侧闭环硬校验**(scripts/aggregate_p03.py：27-38)

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

角色：数据有效性 gate。`cached_ok_all` 是"每一个请求的命中数都精确等于协议期望"的合取——任一请求违反即整格 False，该格数字不得进对外文档。`cv or 0` 处理的是 usage 语义：cached=0 时字段**缺失**而非 0（下一段）。改错会怎样： 校验若放宽为 `v >= pl*0.9` 之类的软阈值，预热失败、残树、中间层改写请求体这三类协议破坏都会漏网（讲义 02 的 EXP-P06 首轮作废正是靠同思路的硬 gate 才被发现）。

**为什么是"恰等于"而不是"至少"**（本讲义推导）：恰等于同时锁住了两个方向。下界防的是"没命中"（预热失败、树被清）；上界防的是"多命中"（预热尾巴撞上、跨点残树、请求之间意外共享后缀）。只设下界的 gate 会放过 §4 第 2 段那种 "预热请求被全长命中"的污染——而那种污染会让收益看起来**更好**，是最危险的一类，因为它不会让人起疑。

**第 5 段 · 与引擎计数器对账**(scripts/aggregate_p03.py：41-47)

```python
m = glob.glob(f"{P}/*_radix_on_metrics.txt")
if m:
    txt = open(m[0]).read()
    for pat in ["device_hit", 'mode="input"', "cache_hit_rate"]:
        for line in txt.splitlines():
            if pat in line and not line.startswith("#"):
                print("METRICS:", line.strip())
```

角色：归因三方相等（§3.4）的第二方。把 /metrics 快照里的 device_hit counter 原样打印，与客户端 Σcached 对账。metrics 快照同时是一个现成的教学反例：同一文件里 `cache_hit_rate` 是 0.0（窗口化 gauge，空闲后归零）而 `prefill_effective_tokens_total{mode="device_hit"}` 是 466944.0——累计口径只能用 counter，gauge 不能当累计命中率读（EXP-P03 §6 的坑）。

**第 6 段 · 上限来自哪一行**（上游 sglang v0.5.18， python/sglang/srt/managers/schedule_batch.py：1411-1416）

```python
    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)
```

角色：§3.1 推导的源码落点。注意第二个分支：请求要 logprob 时上限还会进一步压低到 `logprob_start_len`——要"从头给 logprob"就得从头重算，缓存复用与 logprob 是此消彼长的。EXP-P01 的 1324/1325 精确命中了第一个分支。

**第 7 段 · cached 的报告语义**（上游 python/sglang/srt/entrypoints/openai/usage_processor.py：12-15）

```python
    @staticmethod
    def _details_if_cached(count: int) -> Optional[PromptTokensDetails]:
        """Return PromptTokensDetails only when count > 0 (keeps JSON slim)."""
        return PromptTokensDetails(cached_tokens=count) if count > 0 else None
```

角色：解释为什么冷启动请求的 raw 里 `cached_tokens` 是 `null` 而不是 0（EXP-P01 首发、P03 off 臂全部如此）。details 仅在 >0 且 `--enable-cache-report` 开启时携带。改错会怎样：客户端不做 `or 0` 归一， 统计代码把 None 参与求和直接抛异常，或被当 0 静默——本仓所有脚本统一 `or 0`（EXP-P02 §6 的结论）。

**第 8 段 · 树在什么时候、插什么进去**（上游 python/sglang/srt/mem_cache/radix_cache.py：476-496）

```python
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            freed_end = result.prefix_len
```

角色：第 1 行就是 EXP-P02"意外收获"的源码依据——插入的是 `origin_input_ids + output_ids` **全序列**，所以后来者可以命中前者的**输出**， 这就是 1326 = 1325 + 1 里那个 +1。三条推论：①同一会话多轮对话天然共享，因为上一轮的输出是下一轮输入的一部分；②A 请求的输出会成为 B 请求的可命中前缀， "命中率"因此可能超出"输入共享比例"；③插入发生在请求**完成**时，所以并发同组请求会同时 miss——这条是讲义 02 里 8B fcfs 命中只有 0.757 的机理候选。

**第 9 段 · LRU 时间戳在哪一行被写：路径刷新**（上游 python/sglang/srt/mem_cache/radix_cache.py：678-702）

```python
    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node
```

角色：§3.5.2 等价性命题的源码依据。两处细节值得注意：①`access_time` 只采样 **一次**（第 2 行），路径上所有节点写同一个值——所以一次下行内的节点在 LRU 排序上完全平局；②被写的是路径上每一个节点，不只是终点——这就是"共享前缀被每个共享者自动保热"的实现。改错会怎样：若只刷新终点节点，深层热前缀的祖先（往往是更通用、被更多请求共享的段）会因为"很久没被单独当作终点"而先被逐出， 而它一旦被逐出，整棵子树都不可达——共享程度越高的段反而越先失效，完全反了。

**第 10 段 · 逐出：叶子最小堆 + 只补缺口**（上游 python/sglang/srt/mem_cache/radix_cache.py：592-620）

```python
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # Tree values are page-aligned copies of a kv row: page-exact segment.
            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
```

配合调用方（上游 python/sglang/srt/mem_cache/common.py：134-138）：

```python
    else:
        # Standard allocator: evict only the shortfall (mirrors the SWA arm)
        available_size = allocator.available_size()
        if available_size < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens - available_size))
```

角色：§3.5.4 那笔精算的两个关键机制都在这里。①**堆只从叶子建**，父节点在变成无子且无锁时才被推进堆（第 613-615 行）——逐出的自然单位是"一条自底向上的链"；②**只逐出缺口**（common.py：138 的 `num_tokens - available_size`）， 而 while 的终止条件是 `num_evicted < num_tokens`，一次弹出释放的可能远超缺口， 循环立刻结束。这两条合起来解释了 EXP-P05 的 $c=0$ 格为什么在"名义上越线"的情况下仍然全命中。改错会怎样：若改成"逐出到某个水位线"（常见的 high/low watermark 设计），$c=0$ 格就会把热前缀一起冲掉，悬崖的位置会左移。

**一处必须标注的不确定性**（本讲义推导）：`leaves = list(self.evictable_leaves)` 从一个 `set` 取序，而 `TreeNode` 没有自定义 `__hash__`，集合迭代序由对象 id 决定；`heapq` 在 priority 平局时的弹出顺序又依赖建堆时的输入序，`TreeNode.__lt__` (radix_cache.py：299-300)比较的还是同一个 `last_access_time`，平局仍是平局。 **所以"逐出顺序完全确定"并不是由构造保证的**；EXP-P05 观察到的 seed 间 std=0 是实测结果，不是可以先验推断的性质。在本仓的构造下平局节点的命运相同（要么都是压舱后缀、要么都是热前缀），所以平局不影响命中率——但换一个构造就不一定。

**第 11 段 · 逐出策略就是一个 priority 函数**（上游 python/sglang/srt/mem_cache/evict_policy.py：16-23）

```python
class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)
```

角色：整个 §3.5 的模型只依赖 `LRUStrategy` 这三行。换成 `LFUStrategy` 后排序键变成 `(hit_count, last_access_time)`——轮转热前缀的 `hit_count` 会持续增长而冷流量恒为 1，理论上 lfu 能挡住 EXP-P05 的冲刷。**本仓未测**（EXP-P05 §7 列 backlog），所以 §3.5 的模型必须带定语"在 lru 策略下"。上游 v0.5.18 一共提供六种（lru/lfu/fifo/mru/filo/priority，另有 slru）， 每一种都只是这个 priority 函数的一个不同实现——把逐出策略抽象成 "给节点排个序"是一个值得学的设计：模型层面只要换排序键就能换策略。

**第 12 段 · cache_hit_rate 为什么是窗口 gauge**（上游 python/sglang/srt/managers/scheduler_components/metrics_reporter.py：646-657）

```python
            effective_input_tokens = (
                prefill_stats.log_input_tokens
                - prefill_stats.reprocessed_log_input_tokens
            )
            effective_hit_tokens = (
                prefill_stats.log_hit_tokens - prefill_stats.reprocessed_log_hit_tokens
            )
            total_tokens = effective_input_tokens + effective_hit_tokens
            cache_hit_rate = (
                effective_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )
            self.metrics_collector.increment_effective_prefill_tokens(
```

角色：§4 第 5 段那个"同屏 0.0 与 466944.0"的源码解释。分子分母都取自 `prefill_stats.log_*`——**每个 log interval 结算并清零的窗口量**，所以空闲一个窗口后它必然回 0；而 `increment_effective_prefill_tokens` 把同一批数据累加进 counter，counter 才是累计口径。上游在 counter 的 documentation 里把正确用法写明了："Windowed prefix cache hit rate = rate(sum of *_hit) / rate(sum of all modes)"(metrics_collector.py：892-901)。注意还有一个减法项 `reprocessed_log_*`——被 retract 后重跑的请求要从统计里扣掉，否则同一批 token 会被计两次。**本仓的三方相等能成立，依赖的正是这个扣减是对的**；如果它有偏差， 466,944 就对不上协议期望。这是一条"我们的闭环反过来验证了上游统计正确性"的副产品。

## 5. 实验数据怎么读

### 5.1 fig2(8B 收益曲线)的读法

以 figures/fig2_p07_ttft_vs_prefix_8b.png（8B 收益曲线）为主样本：

- **轴与口径**：x 轴是共享前缀长度（token，总长固定 2048——定总长扫前缀， 变量只有 prefix_len）；y 轴是 TTFT p50(ms)，点值是 3 seeds 的 p50 均值， 误差条是 3 seeds 间的 std（不是 16 请求内的分布宽度——组内分布见 raw 的 ttft_ms 数组）。三条线：on·并发 1、on·并发 8、off·并发 1（反例臂）。
- **这个设计防了哪些坑**：①**反例臂**（disable-radix 起服）排除"降幅来自别的什么"（如长前缀带来的分词/调度差异）——off 臂 229.7→231.8 ms 无趋势； ②**预热计时外**排除"首请求替大家种树"的混合态；③**每点 flush**排除跨点残树；④**3 独立 seeds**给出轮间波动的量尺，单轮的 0.28 ms 微趋势（EXP-P03 off 臂）才能被按协议判为"无可区分"；⑤**逐请求 cached 硬校验 + 计数器对账** (§3.4)把"命中了多少"钉死。
- **机理账怎么列**（读图时心算）：并发 1 斜率 =(228.4−52.9)/1792 ≈ 98 µs/token，乘回去可预测任意前缀长度的 TTFT；并发 8 斜率 ≈(1068.3−234.5) /1792 ≈ 465 µs/token，斜率比 ≈ 4.8 就是排队放大系数（0.6B 版本是 5.3 → 40 µs/token，×7.6，EXP-P03 §6）。fig1(0.6B)与 fig2(8B)同图形语言， 对读即见"收益天花板随模型规模"。

### 5.2 p50 与 mean 的分工(一个仓内真实案例)

exp_p07 csv 的 on/c1/512 行 p50=180.51 而 mean=788.41——3 seeds 中一个 seed 的 1/16 请求出现 29,309 ms 孤立离群（EXP-P07 §7，根因不可考，server 日志被启动截断，按终端级证据记录）。headline 用 p50（稳健），mean 列如实保留污染值不做剔除——两列并存本身就是诚实度的展示面。c8 的 mean < p50（如 1068.3 vs csv mean 列）则是并发批内先完成者拉低均值的左偏，属预期。

**为什么不剔除离群点**（本讲义推导）：剔除需要一条跑前锁定的规则（例如"超过中位数 100 倍者剔除"），而这条规则一旦是跑后定的，就等于用结果挑数据。保留 mean 列的代价是它对这一格没有解释力，收益是读者能自己看见"有一个 29 秒的请求存在"——这个事实本身比一个干净的均值更有信息量：它说明本机在某种未知条件下会出现秒级停顿，这是一个开放问题而不是一个可以抹掉的噪点。

### 5.3 fig3(逐出悬崖)的读法

x 轴直接取重用距离 $D=8192\times(1+cr)$ 而非 cr——$D$ 才是机理变量，cr 只是构造手段；三色柱是三个池位，读图就是逐格核对"池 ≥ D ⇔ 命中 1.0"。std=0 的误差条不是"没画"，是协议确定性的结果（temperature=0、串行、固定 seed 负载下逐出顺序完全确定）。

**读这张图时要同时记住 §3.5.4 的精算**：x 轴的 $D$ 是粗模型值；精算值是 $8224(1+cr)$，与粗模型差 0.4%，不影响任何一格的判定，但把"cr=0 恰好压在边界上" 这句话变成了"cr=0 名义上越线 32 token，靠后缀压舱吸收"。图上看不出这个区别， 只有把 token 收支逐项列出来才看得见——**这是"图给结论、账给机制"的分工**。

### 5.4 std = 0 意味着什么,以及它不保证什么

P05/P06 全部格 seed 间 std=0，不是"没测出波动"，而是协议确定性的体现： temperature=0、串行注入、seed 只改变 token 内容不改变结构（热集大小/轮转序）， 逐出与路由的行为逐次完全相同。看到 std=0 应当去查协议是否确定性，而不是怀疑数据造假；反之，把这种格子的结论外推到并发/随机到达时必须重新测。

**但要注意 §4 第 10 段那条不确定性**：逐出顺序在 priority 平局时依赖集合迭代序， 并不是构造上确定的。所以正确的说法是"在本构造下观察到 std=0"，而不是"本实现保证确定性"。两者的差别在于：前者是可复现的观察，后者是一个本仓无权作出的断言。

### 5.5 口径速查:哪些数字能外推,哪些不能

| 数字 | 定语 | 能不能换定语后引用 |
|---|---|---|
| −77% / −78% | 8B、共享前缀 1792/2048、TTFT p50、并发 1/8、3 seeds | **不能**；换模型/前缀占比/并发都要重测 |
| 98 µs/token 斜率 | 8B、并发 1、total_len 2048 | 可在同模型同并发下内插不同前缀长度 |
| 112 / 144 KiB per token | 由 config.json 结构决定 | **能**，只要模型结构不变（与负载无关） |
| $D = H\cdot T(1+c)$ | 轮转访问、冷流量全唯一、lru 策略 | 形式可推广，数值必须按新构造重算（§3.5.4） |
| 466,944 | 该协议下的算术恒等式 | 只对该协议成立，是校验量不是性能量 |
| 1.0000 / 0.0625 | 8192 池、0.6B、串行 | **不能**当作"命中率随池大小的函数"，只有三个采样点 |

## 6. 误区与边界

至少踩过一次才写得出来的错误直觉（前两条是仓内被证伪/修正的真实案例）：

1. **"thinking 开关会破坏前缀共享"——本仓预注册假设，已被证伪**（EXP-P02， §3.2 全程）。教训的一般形式：模板参数落在渲染结果的哪个位置，必须做一次 CPU 渲染 diff 才知道，"参数改了所以头部变了"是想当然。边界：结论限定 Qwen3 模板族；换 system-prompt 开头非 `<think>` 的模型，+1 现象不复现。
2. **"缓存吃紧时命中率按比例下降"——本仓预注册用词"退化曲线"，被实测修正为阶跃**(EXP-P05，§3.5)。轮转工作集 + LRU 下不存在"缓存小一点、命中低一点"的软着陆；容量规划按重用距离，越线即崩。边界：非轮转的真实到达序（Zipf 偏斜、Poisson）会让各前缀的 $D$ 异质化，悬崖会被抹成分段的软化曲线——模型给的是每个前缀各自的越线条件，不是全局形状。
3. **"cached_tokens 应该等于 prompt 长度"**：上限是 $n-1$(§3.1)，而且 cached=0 时字段缺失而非 0（§4 第 7 段）。拿"命中=全长"做断言的 gate 会假 FAIL，拿 None 当 0 之外的语义会算错命中率。
4. **"cache_hit_rate 指标就是命中率"**：它是窗口化 gauge，空闲后归零—— metrics 快照里它与 device_hit=466,944 同屏出现 0.0（§4 第 5/12 段）。累计口径必须用 `prefill_effective_tokens_total` 两条 counter 差分。
5. **"并发 1 测得的收益就是缓存的价值"**：0.6B 并发 1 只有 −36%，并发 8 有 −63%(EXP-P03)——排队项把收益放大（§3.3）。反过来，拿高并发数字不带并发定语去讲"单请求快了 63%"同样是错的。
6. **"逐出是按 token 一个一个逐的"**：逐出的自然单位是**一条自底向上的链**（§4 第 10 段），而且只补缺口。所以"池差 32 个 token"不会导致"逐出 32 个 token 里最老的那些"，而是"弹出最老的那个叶子节点，可能一次释放 520 个"。把逐出想成 token 粒度，会算错边界格（§3.5.4）。
7. **"共享前缀需要额外机制来保热"**：不需要。路径刷新语义（§4 第 9 段）让祖先节点自动获得所有后代的访问时间，共享度越高的段时间戳越新。误以为需要 "pin 住系统提示词"，会去找一个上游根本没有的旋钮。
8. **"命中率高就等于省得多"**：命中率是 token 账，省时间是毫秒账，中间隔着每 token 的 prefill 成本与地板 $C$。0.6B 的 hit 可以和 8B 一样高，收益却只有一半（§3.3）。**任何把命中率直接当性能指标汇报的做法都缺一次换算**。
9. **"page_size 越小越好"**：小页给更细的命中粒度，代价是 gather 开销与页表规模。本仓 page_size=1 且没有对照臂，所以"1 是最优"不是本仓能下的结论； 能说的只有 FlashInfer 报告的 decode ~1% / prefill ~10% 的开销量级（§3.7）。
10. **"三方相等就证明了因果"**：它证明的是账目，不是因果（§3.4.1）。这是本篇里最需要克制的一句话。

**适用边界（明确列出）**：本讲义全部数字来自单 worker、RTX 4090、SGLang v0.5.18、Qwen3-0.6B/8B、input_ids 直传的合成负载（随机 token，总长 2048， 输出 32）；−77%/−78% 带定语"共享前缀 1792/2048、TTFT p50、并发 1/8、3 seeds"；真实 chat 流量（模板渲染、变长、多轮）只有 messages 形态的 probe 臂（EXP-P01/P02）覆盖，收益曲线不直接外推。显存侧收益、page_size 对照、 lfu 策略对照、多副本行为**全部未测**，不在本篇主张范围内。

## 7. 连环追问

1. **Q：RadixAttention 缓存的是文本还是 token？** token id 序列（RadixKey，radix_cache.py：59），外加 extra_key/cache_salt 命名空间。文本相同但渲染后 token 不同即 miss——EXP-P02 的矩阵就是逐格验证这件事。
2. **Q：为什么命中上限是 input_len−1？** 至少重算 1 个 token 才有最后位置的 logits 可采样（§3.1 四步推导； schedule_batch.py：1411-1416）。实测 1324/1325 精确落在上限（EXP-P01）。
3. **Q：radix tree 为什么优于"整段 prompt 哈希"？** 哈希只有全等命中，radix tree 给出任意长度的最长公共前缀；匹配中段还能分裂节点暴露精确边界（radix_cache.py：704）。代价是树维护与锁。
4. **Q：正在被别的请求使用的 KV 会被逐出吗？** 不会。lock_ref>0 的节点在 protected 段，逐出扫描跳过（inc/dec_lock_ref，radix_cache.py：622-656；叶状态维护：820）。
5. **Q：−77% 的降幅怎么归因给"省掉的 prefill"而不是别的？** 三件：off 反例臂持平；逐请求 cached 恰=prefix_len 的硬校验；engine device_hit 计数器与客户端 Σcached、协议期望三方 466,944 逐 token 相等（§3.4）。
6. **Q：为什么 0.6B 收益比 8B 小这么多？** TTFT = 地板 C + prefill；0.6B 的 C(~17 ms)占比大，可省部分小。收益天花板 ≈ prefill 占比 × 前缀占比（§3.3 的账）。
7. **Q：预热请求会污染测量吗？** 预热在计时外，且尾部加 8 个随机 token 保证它不被计时请求全长命中；聚合校验（cached 恰=prefix_len）会捕获任何预热异常（§4 第 2/4 段）。
8. **Q：flush_cache 一定清干净吗？** 只在引擎 idle 时真清，有 pending 请求会失败（theory/01 §4）；所以每臂开始前要确认 flush 成功再动，冷验证（A 发 cached 缺失）双确认（EXP-P02 §7）。
9. **Q：重用距离模型对 lfu 逐出还成立吗？** 不直接成立。模型第 3 步依赖"LRU 只看最近"的性质；lfu 按 hit_count (evict_policy.py：22)，轮转热前缀的频次高于冷流量，理论上保热更好——本仓未测（EXP-P05 §7 列 backlog），不外推。
10. **Q：cache_salt 存在的意义？** 安全语义：命中带来的时延差可作侧信道探测"别人是否问过同样的前缀"； salt 提供硬隔离（EXP-P02 salt_diff 格实测全 miss）。反过来实验里可用 salt 制造"人为 miss"对照。
11. **Q："只逐出叶子"会不会让实际逐出顺序偏离真 LRU？** 在 v0.5.18 的实现下不会。`_match_prefix_helper` 与 `_insert_helper` 对路径上每个节点写同一个时间戳，所以祖先的时间戳不小于任何后代，LRU 最小值落在叶子上（§3.5.2 的命题与证明）。唯一例外是同一次下行里新建的节点， 微秒级差异，不改变逐出次序。
12. **Q：EXP-P05 的 cr=0 格，$D$ 明明等于池容量，为什么能全命中？** 精算下需求是 8224 而池是 8192，名义上越线 32 个 token。能命中是因为逐出只补缺口（common.py：138）且弹出的是最老的**后缀叶子**（一次释放 520）， 热前缀节点此时不是叶子、也不是最老的（§3.5.4）。**粗模型给对了答案， 理由不完整。**
13. **Q：如果换成 FIFO 逐出会怎样？** FIFO 不满足包含性质，可能出现 Belady 异常（加大池反而更多 miss； Belady/Nelson/Shedler 1969）。上游确实提供 `FIFOStrategy` (evict_policy.py：26-28)，所以这不是假想问题——用 fifo 时"加池一定不变差"这条直觉失效。本仓未测。
14. **Q：为什么不用 Belady 的 MIN 做逐出？** MIN 需要知道未来访问序（Belady 1966），在线系统拿不到。它的价值是作为 **上界**：把 MIN 的命中率算出来，可以量化在线策略离最优有多远——SGLang 论文正是这么做的，报告 cache-aware 调度"approaches 96% of the optimal hit rate on average"(arXiv:2312.07104，§6.2)。本仓没有实现 MIN 基线。
15. **Q：命中率能不能不实测、直接算出来？** 原则上能：Mattson 的一次遍历法可以从访问 trace 一次算出所有容量下的命中率 (IBM Systems Journal 9(2)， 1970)。前提是拿得到逐访问的 trace，本仓没有采集 engine 侧访问序，所以只能逐档实测。这是一个明确的工具缺口。
16. **Q：为什么 prefill 的收益上限不是 100%？** 因为地板 $C$ 存在，而且匹配上限是 $n-1$。即使 $k=n-1$，还要算 1 个 token 的完整前向 + 全部固定开销。8B 上 $C \approx 24$ ms，占 miss TTFT 的 10.5%——这就是 87.5% 的前缀占比只兑现出 77% 降幅的算术来源（§3.3）。
17. **压力问 Q：466,944 三方相等，是否证明了 TTFT 降幅的因果？** 诚实答：它证明的是**命中账目**分毫不差（没有虚报/漏报的 token），因果归因还需要 off 反例臂（排除其它机制）与曲线的线性形态（斜率稳定）共同支撑。严格的"降幅逐请求分解到 prefill 段"需要 server 侧 per-request 时序（queue_time/prefill_finished_time），本仓用 counter 差分 + 反例臂的组合替代，这是口径上的取舍（theory/03 §3 明示不要拿 client TTFT 直接说"prefill 变快了"）。补一个具体的漏洞：与前缀长度相关但与 prefill 无关的机制，反例臂分离不了；要堵住它需要"前缀相同但强制 miss"的第三臂（`SGLANG_RADIX_FORCE_MISS`），本仓未做（§3.4.1）。
18. **压力问 Q：−77% 在真实业务流量上还成立吗？** 不能承诺。该数字的定语是共享前缀 87.5%(1792/2048)的合成负载——这接近 "长系统提示词 + 短用户后缀"的理想形态。真实流量前缀占比、多轮结构、到达序都不同；0.6B/8B 的对比说明收益还强依赖模型的 prefill 占比。本仓本项目对外措辞的硬约束是：数字必须连同负载定语一起说，"前缀缓存让 TTFT 降 77%"单独成句是禁用措辞。
19. **压力问 Q：你说 prefill 已经跑在算术界附近，凭什么？** 诚实答：凭一个**有内部矛盾**的折算。按白皮书 165.2 TFLOPS 算，每 token 的理论下界是 91 µs，实测斜率 98 µs，比值 93%——这个效率高于稠密 GEMM 的常见水平，说明折算的某个环节有问题（峰值口径、FLOP 计数、或斜率的语义）。本仓没有 kernel 级测量，判不了是哪一个，已标注未核实（§3.3.2）。能站住的只有定性结论：prefill 是算力受限的，decode 是带宽受限的，前缀缓存只对前者有效。**把一个不自洽的折算原样写出来，比把它调到"看起来合理"更有用。**

## 8. 工业对照与延伸

### 8.1 论文/文档声称 vs 本机实测:逐条对照

本节把"论文或官方文档说了什么"与"本仓在 RTX 4090 单机上测到什么"并排放，并诚实分析差异来源。差异不粉饰：多数来自规模、口径与硬件代际，少数来自本仓的实验范围限制。

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | SGLang §3：匹配上限的语义（源码注释"at most 1 less than the input length"） | EXP-P01 第二发 cached=1324 / prompt=1325，精确落在 $n-1$ | **完全一致**，而且是"从源码读出上限 → 实测精确命中"的闭环。本仓把一句注释量化成了一个可核对的数字 |
| 2 | SGLang §3："evicts the least recently used leaf first" | EXP-P05 三池全部符合 LRU 的包含性质（池越大越不容易崩） | 一致。本讲义补出的是**为什么"叶子"不构成限制**（§3.5.2 的路径刷新论证），论文只给结论未给论证 |
| 3 | SGLang §6.2：cache-aware 调度"approaches 96% of the optimal hit rate on average" | 本仓**未实现 MIN 基线**，无法算"距最优多远" | 口径不可比。论文的最优值来自离线 MIN；本仓只有绝对命中率（0.0625/1.0）。要复现这条需要采集访问 trace + 实现 MIN，列为工具缺口 |
| 4 | SGLang §6.3：RadixAttention 在无复用负载上的管理开销"only 0.2 seconds"（占 74.3 秒的 0.3%） | 本仓 off 臂（disable-radix）与 on 臂在 prefix=0 点的 TTFT：8B 229.7-231.8 vs 228.4；0.6B 26.59-26.87 vs 26.84 | **同向且量级一致**（差值淹没在 seed 间波动里）。但口径不同：论文测的是 ShareGPT 吞吐，本仓测的是单点 TTFT。两者都支持"无复用时开销可忽略"，本仓不能复述论文那个 0.3% |
| 5 | SGLang §6.2："The speedup is more noticeable for short outputs... For long outputs... there is almost no speedup" | 本仓 output_len 固定 32（短输出），未扫输出长度 | 不可比，但**方向被本仓的机制账支持**：收益上限 ≈ prefill 占比（§3.3），输出越长 decode 段越占主导。本仓不能主张自己验证了这条 |
| 6 | PagedAttention §1：既有系统 KV 显存利用率仅 20.4%-38.2% | 本仓**未测显存利用率**（固定池，只看时延与命中） | 不可比。论文测的是"分配了多少 vs 真正用了多少"；本仓的池是显式配置的固定值，不存在预留浪费这个变量 |
| 7 | PagedAttention §6.4：共享 1-shot 前缀（80 token）吞吐 1.67×，5-shot(341 token)3.58× | 本仓 8B 共享 1792 token 时 TTFT p50 −77%（≈4.3× 加速） | **同向，数值不可直接比**：三处口径差异——论文是吞吐（req/s）本仓是 TTFT；论文前缀 80/341 token 本仓 1792；论文 LLaMA-13B 本仓 Qwen3-8B。共同点是"共享得越多、收益越大"，且都不是线性 |
| 8 | PagedAttention §7.2：block size 16 是默认，"large enough to efficiently enable... yet small enough to avoid significant internal fragmentation" | 本仓 page_size=1，未做对照 | 两条不同的设计点。本讲义给出的是**折扣公式**（§3.1.3：期望损失 $(p-1)/2$ token），说明大页只在短前缀上致命——这解释了为什么 16 在长 prompt 场景下够用 |
| 9 | FlashInfer 附录 B：page size 1 的向量稀疏 gather，decode 内 1%、prefill 约 10% 开销 | 本仓全程 page_size=1 且无对照臂，**这项开销已包含在所有数字里** | 无冲突。含义是本仓的 −77% 是**含分页开销的净值**；若把这 10% 也省掉，收益还会更大一点 |
| 10 | Mattson 1970：满足包含性质的算法命中率随容量单调不减 | EXP-P05 三池：8192 崩得最早、16384 次之、默认全保 | **一致**，而且这不是巧合而是定理的直接后果。反过来说：如果测到"加大池反而更差"，第一反应应当是怀疑协议而不是怀疑理论 |
| 11 | Sleator-Tarjan 1985：LRU 的竞争比等于缓存大小 $k$，下界由循环访问序达到 | EXP-P05 观察到 hit 1.0 → 0.0625 的阶跃 | **一致**。本仓的构造就是那个下界构造的 token 计价版。把这条对上意味着：实测到的不是实现缺陷，是一条 1985 年的下界 |
| 12 | Ada 白皮书 Appendix A Table 2：RTX 4090 1008 GB/s、BF16 Tensor（FP32 累加）165.2 TFLOPS | 本仓无微基准，所有折算直接用规格值 | 这是 §3.3.2 那处**已知的不自洽**的来源之一：用规格值折出的 prefill 效率达 93%，高于常见水平。**标注未核实**，不作为结论 |

**这张表的读法**：十二条里只有 1、2、10、11 是严格的"预言—证实"闭环，其余多半是"不可比"或"同向但换了口径"。**论文的性能数字几乎从不能直接搬到你的机器上，能搬的是机制、判据与不等式。**

### 8.2 与生产实现的差距各在哪一层

- **vLLM(automatic prefix caching)**：同思想不同数据结构——vLLM 以固定大小 block 的 hash 链实现前缀复用（block 粒度命中），SGLang radix tree 是 token/page 粒度 + 节点分裂，长尾前缀的命中粒度更细；代价是树结构的维护与锁复杂度。vLLM 的哈希把三个分量串起来：父块哈希、本块的 token 元组、以及 "其它使这个块唯一的值，如 LoRA id、多模态输入哈希、cache salt"（vLLM 官方设计文档 "Automatic Prefix Caching"）——**注意这三个分量与 SGLang 的 RadixKey 三元组几乎一一对应**，只是一个编进哈希、一个编进字典键。两家在 "命名空间隔离"这件事上的语义是相同的。vLLM 只缓存整块（"We only cache full blocks"），这正是 §3.1.3 那条对齐折扣公式的另一种表述。本仓 page_size=1， 未测 page>1 时的对齐折扣（theory/03 §4）。
- **SGLang 上游 HiCache / hiradix**(mem_cache/hiradix_cache.py)：KV 分层到 host 内存甚至存储，`cached_tokens_total{cache_source=device|host}` 的 host 维度本仓从未触发——单机显存池内的结论，不含分层缓存的换入换出。分层之后 §3.5 的模型要改成多级栈距离（每一级各有各的容量与命中条件）， 这是一个自然但本仓未做的推广。
- **多副本 serving**：单 worker 结论到了多副本会叠加"前缀→副本映射"这一层（sgl-model-gateway 的字符级近似树与 engine 的 token 树是两棵树），见讲义 02 与 EXP-P06。SGLang 论文附录 A.4 描述的是另一种设计：router 维护 meta-tree，worker 逐出时把事件提交到队列由 router 异步消费（"Should an eviction occur at a worker node， it commits this eviction to a queue， which the router then processes to update the meta-tree during periods of low activity"，arXiv:2312.07104，§A.4）。**被测的 0.3.2 版 gateway 不是这个设计**（它用字符级近似树，不接收 worker 的逐出事件），这是讲义 02 §3.3 那 "两棵树不一致"问题的根源。
- **确定性与缓存的互斥面**：上游 `cache_finished_req` 里有 `disable_finished_insert`（确定性模式不插树）——复用与逐 bit 可复现之间存在工程权衡，本仓 temperature=0 的确定性验证（EXP-P01）未开该模式。
- **本仓刻意没做的三件事**：①显存侧收益（需要变池大小 + 测最大 batch）； ②真实 trace 回放（需要采集或获取生产访问序）；③MIN 基线（需要离线最优）。这三件加起来就是"从机理实验升级为容量规划工具"的完整路径。

### 8.3 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)

**论文**

1. Zheng， Yin， Xie， Sun， Huang， Yu， Cao， Kozyrakis， Stoica， Gonzalez， Barrett， Sheng， "SGLang： Efficient Execution of Structured Language Model Programs"， arXiv:2312.07104，§3、附录 A.2(Alg. 1)、附录 A.3（Theorem 3.1 及证明）、附录 A.4。——想知道"RadixAttention 到底是什么"、"cache-aware 调度为什么等价于 DFS"、"多副本原设计长什么样"，读这四处；§3 的"evict the least recently used leaf first"一句是本篇 §3.5.2 的出发点。
2. Kwon， Li， Zhuang， Sheng， Zheng， Yu， Gonzalez， Zhang， Stoica， "Efficient Memory Management for Large Language Model Serving with PagedAttention"， arXiv:2309.06180，§1、§4.1-4.3、§4.4(shared prefix)、§6.4、§7.2(block size)。——想弄清"另一条路线（定长 block + block table + copy-on-write）怎么做前缀共享"以及"块大小该怎么选"，读这几节；与本篇 §3.1.3 的对齐折扣公式对读。
3. Ye， Chen， Lai， Lin， Zheng， Wang， Chen， Wang， Yu， Ceze 等， "FlashInfer： Efficient and Customizable Attention Engine for LLM Inference Serving"， arXiv:2501.01005，§3.1.1、§3.1.2、§3.2.1、附录 B。——想回答"page_size=1 为什么不把带宽打垮"以及"共享前缀在 kernel 层怎么被利用"，读这四处； §3.2.1 的"last dimension remains contiguous"是本篇 §3.7 的关键句。
4. Srivatsa， He， Abhyankar， Li， Zhang， "Preble： Efficient Distributed Prompt Scheduling for LLM Serving"， arXiv:2407.00023，§1、§3.2。——想知道"把前缀亲和推到多副本会出什么问题"以及"一个带容量项的路由代价函数长什么样"， 读这两节；它是讲义 02 §3.4 那次证伪的理论对照物。
5. Mattson， Gecsei， Slutz， Traiger， "Evaluation techniques for storage hierarchies"， IBM Systems Journal 9(2)：78-117, 1970(DOI 10.1147/sj.92.0078)。——想知道"栈距离/重用距离的原始定义"与"为什么 LRU 的命中率随容量单调"， 读这一篇；本篇 §3.5.1 的两条推论全部来自它。
6. Belady， "A study of replacement algorithms for a virtual-storage computer"， IBM Systems Journal 5(2)：78-101, 1966(DOI 10.1147/sj.52.0078)。——想知道 "最优离线替换（MIN）长什么样、为什么在线系统做不到"，读这一篇；它是 SGLang 论文里"optimal hit rate"那个基线的来源。
7. Sleator & Tarjan， "Amortized efficiency of list update and paging rules"， CACM 28(2)：202-208, 1985(DOI 10.1145/2786.2793)。——想知道"LRU 最坏能坏到什么程度"以及"为什么轮转访问是最坏输入"，读这一篇；本篇 §3.5.5 把 EXP-P05 接到了它的下界构造上。
8. Fricker， Robert， Roberts， "A versatile and accurate approximation for LRU cache performance"， arXiv:1202.3974。——想知道"真实流行度分布下 LRU 的命中率曲线为什么是光滑 S 形而不是阶跃"，读这一篇；它给出本篇 §3.5.6 的定量形式与它的 IRM 前提。
9. Denning， "The working set model for program behavior"， CACM 11(5)：323-333, 1968(DOI 10.1145/363095.363141)。——想把"热集大小"这个直觉概念做严格， 以及理解"为什么按工作集配内存可以避免抖动"，读这一篇；它是 §3.5 那句 "按重用距离配池而不是按热集大小"的反面参照。
10. Kingman， "The single server queue in heavy traffic"， Proc. Cambridge Phil. Soc. 57(4)：902-904, 1961。——想知道"为什么服务时间降一点、等待时间降很多"， 读这一篇的 VUT 形式；注意它是开环模型，与本仓的闭环并发定义口径不同（§3.3.3 的标注）。
11. Little， "A Proof for the Queuing Formula： L = λW"， Operations Research 9(3)：383-387, 1961。——想把"并发 = 吞吐 × 时延"这条换算做严格，读这一篇； §4 第 3 段的闭环讨论全部建立在它上面。

**官方文档**

12. NVIDIA， "NVIDIA Ada GPU Architecture" 白皮书，Appendix A Table 2。——RTX 4090 的显存带宽、L2 容量与 Tensor 峰值的唯一权威出处；本篇 §3.3.2 与 §3.6 的所有硬件常数都取自这里（以及那处未核实的效率冲突）。
13. NVIDIA， "GPU Performance Background User's Guide"，§4 Understanding Performance；以及 "CUDA C++ Best Practices Guide" 的 Coalesced Access to Global Memory 一节。——想知道"算术强度/ops：byte 的官方定义"与"全局内存访问的 32/64/128 字节事务语义"，读这两处；§3.7 关于 page_size=1 为什么可行的论证依赖后者。
14. vLLM 官方文档 "Automatic Prefix Caching"(design/prefix_caching)。——想对比"块哈希链"与"radix 树"两种前缀缓存的实现语义（尤其是哈希的三个分量与"只缓存整块"这条约束），读这一篇；它是 §8.2 第一条的出处。
15. SGLang 上游源码 `python/sglang/srt/server_args.py` 的 `radix_eviction_policy` 字段（v0.5.18 在：919-931）。——想知道"除 lru 外还有哪些逐出策略、各自的 priority 键是什么"，从这里跳到 `mem_cache/evict_policy.py`；§4 第 11 段的六种策略全在那一个文件里。

**源码与本仓证据**

16. mem_cache/radix_cache.py：376-434（match_prefix 全注释）与：678-702（下行循环）——树操作的第一手实现，以及 §3.5.2 路径刷新论证的原文。
17. mem_cache/radix_cache.py：592-620(evict)+ mem_cache/common.py：114-138（需求驱动逐出）——重用距离模型的对手方；§3.5.4 的边界格解释全靠这两处。
18. managers/schedule_batch.py：1292 起（init_next_round_input）——请求到匹配的调度侧入口，limit/salt 如何传入。
19. managers/scheduler_components/metrics_reporter.py：646-657 与 observability/metrics_collector.py：892-901——`cache_hit_rate` 是窗口 gauge、 `prefill_effective_tokens_total` 是累计 counter 的源码依据。
20. docs/theory/01_radix_prefix_cache.md 与 03_workload_contract_pitfalls.md——本讲义的机制笔记底稿（全部 file：line 锚）。
21. records/EXP-P02_token_contract_matrix.md——预注册证伪案例的原始记录（§7 含 flush 语义的协议偏差说明）。
22. records/EXP-P05_eviction_pressure.md §5-§7 与 data/derived/exp_p05_eviction_cliff.csv——悬崖三池数据的原始形态； §3.5.4 的精算可以对着 bench_evict.py 的参数逐项复算。
