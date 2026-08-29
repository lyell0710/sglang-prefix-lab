# 讲义 02 · 调度与路由的边界:lpm/fcfs 权衡形态与"映射分散度"

> 读者：准备校招面试的作者本人，以及第一次接触 cache-aware 调度/路由的工程师。读法：不跳步。上游源码锚以 /root/repos/sglang-v0.5.18 为准（worker 侧 python/sglang/srt/，router 侧 sgl-model-gateway/）；实验数字全部带 EXP 锚。凡属论文/官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节；文档给路径 + 小节名）；凡属本讲义自己补出的推导或折算，行内标注"本讲义推导"； 无法用检索确认的说法标注"未核实"。

## 目录

- [1. 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的三条能力](#11-本篇要建立的三条能力)
  - [1.2 符号与口径约定](#12-符号与口径约定)
  - [1.3 本篇引用的一级文献(详细出处见 §8.3)](#13-本篇引用的一级文献详细出处见-83)
  - [1.4 两层 cache-aware 的职责切分](#14-两层-cache-aware-的职责切分)
- [2. 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 论文的最优性定理:它说了什么,以及它的三个前提](#21-论文的最优性定理它说了什么以及它的三个前提)
  - [2.2 代价从哪里来:守恒律,以及 lpm 为什么不只是零和](#22-代价从哪里来守恒律以及-lpm-为什么不只是零和)
  - [2.3 lpm 到底是不是 SPT:一个必须做的区分](#23-lpm-到底是不是-spt一个必须做的区分)
  - [2.4 日常类比与失效点](#24-日常类比与失效点)
  - [2.5 路由层的直觉,与它的三个经典理论框架](#25-路由层的直觉与它的三个经典理论框架)
- [3. 完整机制与两阶段认知](#3-完整机制与两阶段认知)
  - [3.1 lpm 决策链(上游源码走读,文件均为](#31-lpm-决策链上游源码走读文件均为)
  - [3.2 两阶段认知:0.6B 反劣 → 8B 分位数再分配](#32-两阶段认知06b-反劣--8b-分位数再分配)
  - [3.3 cache-aware 路由:决策伪码与失衡回退](#33-cache-aware-路由决策伪码与失衡回退)
  - [3.4 EXP-P06 双证伪全程:奇偶巧合与亲和集中](#34-exp-p06-双证伪全程奇偶巧合与亲和集中)
  - [3.5 魔法数总表:每个数字由谁决定](#35-魔法数总表每个数字由谁决定)
- [4. 代码逐段走读](#4-代码逐段走读)
- [5. 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 fig4(8B 调度权衡)的读法](#51-fig48b-调度权衡的读法)
  - [5.2 图上没有的那一列:makespan](#52-图上没有的那一列makespan)
  - [5.3 方差本身是机制证据](#53-方差本身是机制证据)
  - [5.4 P06 的表怎么读](#54-p06-的表怎么读)
  - [5.5 std=0 怎么理解](#55-std0-怎么理解)
  - [5.6 防坑清单(本组实验特有)](#56-防坑清单本组实验特有)
- [6. 误区与边界](#6-误区与边界)
- [7. 连环追问](#7-连环追问)
- [8. 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 论文/文档声称 vs 本项目实测:逐条对照](#81-论文文档声称-vs-本项目实测逐条对照)
  - [8.2 与生产实现的差距各在哪一层](#82-与生产实现的差距各在哪一层)
  - [8.3 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)](#83-延伸阅读带精确出处每条一句话说明它能解决什么疑问)

## 1. 这一篇回答什么问题

前缀缓存之上还有两层"cache-aware"：引擎内的调度（lpm 排序等待队列）与多副本前的路由（cache_aware 策略选卡）。这两层各自何时有收益、何时反噬。读完你应当能：①沿上游源码走通 lpm 的完整决策链（>128 退化、in-batch 去重、排序键）； ②解释为什么同一协议下 0.6B 的结论是"反劣"而 8B 是"分位数再分配"——模型重量如何改变权衡形态，并给出排队论直觉；③复盘 EXP-P06《路由 × 池容量》的双预测双证伪（奇偶巧合与亲和集中），说清"映射分散度 > 策略标签"这句一般化结论的推导与限度。

### 1.1 本篇要建立的三条能力

1. **机制能力**：能只凭源码说出 lpm 每一轮到底做了什么——包括它什么时候 **不做**（队列 >128 整轮退化成 fcfs）、什么时候**反着做**（in-batch 去重把同前缀请求推到队尾）。看懂"策略开着"与"策略在工作"是两件事。
2. **理论对接能力**：能把 lpm 拆成两个成分——SPT 式重排（零和，只搬运等待） 与缓存聚簇（非零和，减少系统总工作量）——并用这个二分解释为什么 0.6B 是纯负资产而 8B 是分位数再分配（§2.2、§3.2）。
3. **反例设计能力**：能设计出"用对照臂杀死巧合解释"的实验（hot5 奇数臂）， 并在拿到与源码不一致的观测时，把冲突量化后如实留着而不是圆过去（§3.4.3）。

### 1.2 符号与口径约定

| 符号 | 含义 | 本仓取值 |
|---|---|---|
| $G$ | 组数（每组共享一段前缀） | std 档 8，boundary 档 16 |
| $R$ | 每组请求数 | std 档 8，boundary 档 12 |
| $N$ | 总请求数 $G\times R$ | std 64,boundary 192 |
| $c$ | 客户端在途并发上限 | std 16，boundary 64 |
| $L_p$ | 组内共享前缀长度 | 1536（总长 2048） |
| hit_frac | 命中率，分母扣掉每组首请求 | 见 §4 第 7 段 |
| $W$ | 副本数 | 2(EXP-P06) |
| $D$ | 单卡上的热前缀重用距离（讲义 01 §3.5） | 见 §3.4 |
| makespan | 一档全部请求跑完的墙钟时间 | csv 的 `dur_s_mean` 列 |

两档负载的设计意图必须先说清：**std 档是"策略窗口内"，boundary 档是"策略窗口外"**。分界线就是下一节第一段那个硬编码的 128——boundary 档取 192 请求正是为了确定性地跨过它（EXP-P04《调度策略》 §1 预注册）。

### 1.3 本篇引用的一级文献(详细出处见 §8.3)

- cache-aware 调度的原始定义与最优性定理：Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104,§3 "Cache-aware scheduling"、Theorem 3.1、附录 A.2(Alg. 1)、A.3（证明）、A.4。
- 分布式前缀调度的代价模型：Srivatsa, He, Abhyankar, Li, Zhang, "Preble: Efficient Distributed Prompt Scheduling for LLM Serving", arXiv:2407.00023, §1、§3.2。
- 调度守恒律：Kleinrock, "A conservation law for a wide class of queueing disciplines", Naval Research Logistics Quarterly 12(2):181-192, 1965 (DOI 10.1002/nav.3800120206)。
- 最短剩余处理时间的最优性：Schrage, "A Proof of the Optimality of the Shortest Remaining Processing Time Discipline", Operations Research 16(3):687-690, 1968 (DOI 10.1287/opre.16.3.687)。
- SRPT 的公平性实证分析：Bansal & Harchol-Balter, "Analysis of SRPT scheduling: investigating unfairness", ACM SIGMETRICS Performance Evaluation Review 29(1):279-290, 2001(DOI 10.1145/384268.378792)。
- 负载均衡的两个经典结果：Azar, Broder, Karlin, Upfal, "Balanced Allocations", SIAM Journal on Computing 29(1):180-200, 1999;Mirrokni, Thorup, Zadimoghaddam, "Consistent Hashing with Bounded Loads", arXiv:1608.01350。
- 闭环系统的基本律：Little, "A Proof for the Queuing Formula: L = λW", Operations Research 9(3):383-387, 1961;Lazowska, Zahorjan, Graham, Sevcik, *Quantitative System Performance*, Prentice-Hall, 1984,"Fundamental Laws"。

### 1.4 两层 cache-aware 的职责切分

| | 引擎内调度（lpm） | 副本前路由（cache_aware） |
|---|---|---|
| 决定什么 | 已经到这张卡的请求，**先跑谁** | 请求**去哪张卡** |
| 依据的树 | engine 的 token 级 radix 树（真值） | router 的字符级近似树 |
| 作用对象 | 等待队列的顺序 | 前缀到副本的映射 |
| 失效模式 | 队列 >128 整轮退化；超窗负载下反噬 | 亲和集中，把工作集压到一张卡 |
| 本仓实验 | EXP-P04(0.6B)、EXP-P08(8B) | EXP-P06（双副本容量受限格） |

**一句话记法**：router 决定"谁看见请求"，engine 决定"看见后怎么排"。两层都叫 cache-aware，但一个改的是**空间分布**，一个改的是**时间顺序**。本篇按这个顺序讲。

## 2. 直觉与第一性原理

**没有调度策略的世界**：引擎每一步从等待队列头部取请求组 batch。到达序是客户端与网络决定的，共享前缀的请求彼此打散——A 组首请求的 KV 还没插进树（radix 树在请求**完成**时才插 input+output 全序列），同组第二条已经在跑， 本可命中的前缀 miss 掉了。调度器能做的唯一一件事：**重排等待队列**，把同前缀请求聚到一起，让后来者稳定踩在前者刚种下的树上。

SGLang 论文把这件事的动机写得很直白："When there are many requests in the waiting queue, the order in which they are executed can significantly impact the cache hit rate. For example, if the request scheduler frequently switches between different, unrelated requests, it can lead to cache thrashing and a low hit rate."(arXiv:2312.07104,§3 "Cache-aware scheduling")。同一节给出的策略就是一句话："we sort the requests by matched prefix length and prioritize requests with longer matched prefixes instead of using a first-come, first-served schedule."

### 2.1 论文的最优性定理:它说了什么,以及它的三个前提

论文为离线情形证明了一条定理，值得逐字引用（arXiv:2312.07104，§3，证明在附录 A.3）：

> **Theorem 3.1.** For a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in the depth-first search order, with a cache size ≥ the maximum request length. The longest-shared-prefix-first order is equivalent to a depth-first search order.

证明的骨架（附录 A.3，本讲义复述并补出中间步）：

1. 设批次 $R$ 建出的 radix 树为 $T$。对每条边 $e$，它对应的 KV **至少**要算一次， 故总计算量 $C \ge \sum_{e\in \mathrm{edges}(T)} |e|$。这是下界。
2. 按 DFS 顺序访问：第一次算到边 $e$ 之后，接下来会把 $e$ 的整棵子树算完；在算子树的整个过程中 $e$ 一直被命中，不产生额外计算；算完子树后 $e$ 再也不会被访问。
3. **关键的一步**：要保证"算子树期间 $e$ 不被逐出"，需要缓存容量 ≥ 最大请求长度（即树中最长路径）。论文原话："with a cache size ≥ the maximum request length， which equals the longest path in the radix tree $T$， edge $e$ will not be evicted during the computation of its subtree"。
4. 于是每条边恰好算一次，下界达到，命中率 $1 - C/\sum_r(\text{prefill tokens})$ 取到上界。
5. 再用归纳法证明"最长共享前缀优先"等价于一个合法的 DFS 序：已访问节点集合形成一条根到 $y$ 的路径 $P$，任何未访问节点与已访问集合的最近公共祖先都落在 $P$ 上； 由于 $P$ 上的节点都在缓存里，**与 $P$ 分叉点最深的那个未访问节点恰好就是共享前缀最长的那个**，选它就是一步合法的 DFS。

**这条定理的三个前提，每一个在本仓的实验里都被破坏了**（本讲义推导，这正是 EXP-P04/P08 结果不能用定理预测的原因）：

| 前提 | 定理要求 | 本仓实况 | 后果 |
|---|---|---|---|
| 离线 | 一整批请求同时可见 | 在线到达，shuffle 后陆续注入 | 论文自己说"In the online case， the DFS order will be disrupted"(§3) |
| 缓存足够大 | ≥ 最大请求长度 | boundary 档并发 64 × 2048 token 同时在跑，池被在途请求分掉 | 子树没算完 $e$ 就可能被挤掉 |
| 排序真的发生 | 每轮按匹配长度排 | 队列 >128 时**整轮退化为 fcfs**（§3.1 第 1 步） | 排序建立的假设中途失效 |

**所以"论文证明了 lpm 最优"与"本仓测到 lpm 反劣"之间没有矛盾**：定理成立的区域恰好是本仓 boundary 档跨出去的那个区域。这是本篇最重要的一次理论-实测对账， §8.1 会再逐条列一遍。

### 2.2 代价从哪里来:守恒律,以及 lpm 为什么不只是零和

重排不创造算力，只是搬运等待时间。这句话的严格版本是 Kleinrock 的守恒律 (Naval Research Logistics Quarterly 12(2)：181-192, 1965)：对 M/G/1 队列，在一大类保功（work-conserving）、非抢占、**且调度决策不使用服务时间信息**的排队规则下，$\sum_p \rho_p W_p$ 是不变量——给某一类作业的优待，必然由其它类的等待买单。

**但 lpm 恰好落在这条守恒律的适用范围之外**，而且是从两个方向出去的（本讲义推导，这是本篇的核心论点）：

1. **它使用了与服务时间相关的信息**。排序键是"对 radix 树的匹配长度"，而匹配越长 → 需要重算的 prefill 越短 → 有效服务时间越短。所以 lpm 是一个 **SPT（最短处理时间优先）的近似**，属于"用作业大小信息"的那一类，守恒律对它不成立——SPT 类规则能真正压低平均等待，不只是搬运（Schrage 1968 对 SRPT 的最优性证明给出的正是这个方向的结论）。
2. **它改变了系统的总工作量**。这是 cache-aware 调度独有的、SPT 没有的性质： 把同前缀请求聚簇会**提高命中率**，而命中的 token 是不需要算的。EXP-P08《8B 调度》的 boundary 档 hit 从 0.757 升到 0.934，等于把 17.7pp 的 prefill 工作从系统里 **删掉**了。SPT 只能改变作业的服务顺序，不能改变作业的大小；lpm 能。

**于是 lpm 的效果可以干净地分成两个成分**：

$$\text{lpm 的效果} = \underbrace{\text{SPT 式重排}}_{\text{零和：搬运等待，p50$\downarrow$ / p99$\uparrow$}} + \underbrace{\text{缓存聚簇}}_{\text{非零和：删掉工作量，全员受益}}$$

这个二分立刻解释了两阶段认知（§3.2 的实测）：

- **0.6B boundary 档**：fcfs 的 hit 已经是 0.992，几乎没有工作量可删——第二个成分接近零，只剩零和的重排，而重排在超窗负载下还被退化开关与 in-batch 降权干扰。结果就是"只剩账单没有收入"：p99 反劣 13%，hit 还掉了 2.4pp。
- **8B boundary 档**：fcfs 的 hit 只有 0.757，第二个成分很大（17.7pp）。于是 p50 −62%、makespan 也降（§5.2 的 `dur_s_mean` 账），同时第一个成分照样收它的税：p99 +64%。

**这个分解也给出了一个可操作的判据**（本讲义推导）：**在打开 lpm 之前，先看 fcfs 下的命中率**。如果它已经接近 1，lpm 只能带来零和的重排，尾延迟必然变差而没有补偿；只有当 fcfs 命中率明显低于饱和值时，聚簇成分才有发挥空间。这条判据比"lpm 好不好"这种标量问题有用得多。

### 2.3 lpm 到底是不是 SPT:一个必须做的区分

上一节说 lpm 是"SPT 的近似"，这个"近似"有多松？四点差异（本讲义推导）， 每一点都会影响能不能把 SPT 的文献结论搬过来：

1. **代理变量不是作业大小本身**。SPT 用剩余处理时间，lpm 用匹配长度。两者的关系是 $\text{有效 prefill} = n - k$，只有在 $n$ 大致相同时，$k$ 大 才等价于服务时间短。本仓的负载 total_len 固定 2048，所以这个代理在本实验里相当准； 真实流量里 $n$ 方差很大，代理会失真。
2. **排序键是动态的**。前一个请求结完账、KV 插进树之后，后面所有同组请求的匹配长度才变长。所以排序建立的假设**会随执行漂移**——SPT 的作业大小是外生常量，lpm 的排序键是自己造出来的。
3. **被延后的是整组而不是单个作业**。lpm 把匹配短的排到后面，而"匹配短"这个属性在同组内部是**高度相关**的：一整组会一起被推后。SPT 分析里通常假设作业大小独立；相关性会把尾部拉得比独立情形更长。
4. **本仓是闭环系统**。SPT/SRPT 的经典结论建立在 M/G/1（泊松到达，开环）上； 本仓并发由客户端信号量定义，在途请求数恒定（§4 第 7 段的注入端）。闭环下 "尾延迟"的含义变成"最后一批完成者"，与开环的稳态尾分位不是同一个量。

**这四点合起来解释了一处文献与实测的表面冲突**：Bansal 与 Harchol-Balter 分析 SRPT 的公平性，结论是 SRPT 对大作业的不公平"surprisingly small"——在 M/G/1、重尾作业分布下，几乎每种作业大小都更喜欢 SRPT 而不是处理器共享（PS） (SIGMETRICS PER 29(1)， 2001)。而本仓测到 p99 +64%。冲突的来源不是哪一方错了， 而是四处口径不同：①对照物不同（文献比 PS，本仓比 FCFS）；②lpm 不是 SRPT（上面四点）；③本仓是闭环饱和积压，不是开环稳态；④"p99"是 192 条请求内的次序统计量，不是稳态分布的分位。**把这四条写清楚，比引用一句"SPT 会牺牲尾部" 更有信息量。**

### 2.4 日常类比与失效点

超市收银把"只买一件"的顾客插到快速通道，平均结账时间下降，但大采购车被越插越后。类比失效点：①lpm 的"件数"（前缀匹配长度）是**动态**的——前一个请求结完账，后面所有同组请求的"件数"才变少（树插入），排序建立的假设会随执行漂移； ②收银员不会在队伍超过 128 人时突然放弃分流，lpm 会（下一节第一段源码）， 这个工程保护把"重载"变成了它的设计边界之外。

第三个失效点（本讲义推导，也是 §2.2 的类比化）：超市的快速通道不会让商品变少， 而 lpm 的聚簇会让"要扫描的商品"真的变少——同组第二位顾客的前 1536 件商品可以直接用第一位的扫描结果。**这是收银类比里没有的东西，也正是 cache-aware 调度区别于纯排队优化的地方。**

### 2.5 路由层的直觉,与它的三个经典理论框架

多副本下每张卡一棵独立的 radix 树。路由决定"谁看见哪些前缀"，等价于决定 **每张卡的重用距离**（讲义 01 §3.5）：把热前缀分散钉住，每卡只看热集的 1/N， 重用距离缩短 N 倍——这就是"cache-aware 路由 = 扩大有效池容量"的推导。 EXP-P06 证明这条推导的前提（**分散**且**稳定**的映射）恰恰不被现成策略保证——预测双双被证伪，见 §3.4。

把"前缀→副本"的映射问题放进负载均衡的经典框架里看，有三个层次（本讲义整理）：

| 框架 | 结论 | 对应到路由 |
|---|---|---|
| 纯随机（球入桶） | $n$ 球 $n$ 桶，最大负载 $\approx \ln n/\ln\ln n$ | round_robin / 随机选卡：负载均，但**缓存局部性为零** |
| 两选一（power of two choices） | 每球随机看两桶选轻的，最大负载降到 $\ln\ln n/\ln 2 + O(1)$，指数级改善(Azar/Broder/Karlin/Upfal， SICOMP 29(1)：180-200, 1999) | 用少量随机性换均衡，仍不带亲和 |
| 一致性哈希 + 有界负载 | 给定平衡参数 $c=1+\varepsilon$，保证没有桶的负载超过 $\lceil cm/n\rceil$，且每次增删只搬动常数个球（Mirrokni/Thorup/Zadimoghaddam， arXiv:1608.01350） | **亲和 + 容量上界**：正是 EXP-P06 缺的那个东西 |

**第三行是这一篇的理论终点**：EXP-P06 观察到的崩塌，本质是一个"有亲和、无容量上界"的映射。而"亲和 + 容量上界"不是本讲义发明的补丁——被测的同一个 gateway 代码库里就有一个策略实现了它，只是本仓没测（§3.4.4、§8.2）。

## 3. 完整机制与两阶段认知

### 3.1 lpm 决策链(上游源码走读,文件均为
python/sglang/srt/managers/schedule_policy.py)

因果链按执行顺序拆五步：

1. **策略判定**：每轮调度先问队列多长。`_determine_active_policy`(：290-294) 在 lpm 且等待队列 >128 时**整轮退化为 fcfs**——前缀匹配与排序是 O（队列长 × 树深） 的开销，上游选择在重载时保调度器吞吐。含义：lpm 的设计窗口就是 "等待队列 ≤128"；超出后它**不工作**，而不是"工作得差一点"。
2. **逐请求匹配**：`_compute_prefix_matches`(：314-358)对每个等待请求做一次树匹配，把匹配长度写进 `req.num_matched_prefix_tokens`。
3. **in-batch 去重**：匹配很短（≤32，CHECK_THRESHOLD：81）的请求，再到一棵 **模拟树**（等待队列自己的 radix）里查：如果队列里已有 ≥32 token 同前缀的请求（DEPRIORITIZE_THRESHOLD：88），当前请求被**临时降权**——让每个新前缀只有一个"开路者"先跑完种树，其余同伴等树种好再跑，避免同批并发把同一前缀重算 N 遍。
4. **排序**：`_sort_by_longest_prefix`(：373-384)按匹配长度降序，被降权者排到队尾（`float("inf")`）。
5. **执行漂移**：排序只发生在调度时刻；树随请求完成持续变化，fcfs 档下同组请求并发交错时，"首请求 KV 未入树、同组已在跑"的 miss 正是 8B 命中差 17.7pp 的机理候选（EXP-P08 §6，推断标注）。

#### 3.1.1 128 这个数由什么决定(魔法数溯源)

上游把理由写在注释里："Turn off the expensive prefix matching and sorting when the #queue is large."(：292)。把这句话量化（本讲义推导）：

- 第 2 步对每个等待请求做一次 `match_prefix`，单次代价 $O(\text{树深}\times \text{段长})$，实践中与前缀长度同阶；
- 第 4 步的排序是 $O(Q\log Q)$；
- 合计每轮 $O(Q\cdot L_p + Q\log Q)$，而这一整块跑在**调度器线程**上，与 GPU 前向串行。

于是存在一个平衡点：当 $Q$ 大到"排序耗时"可与"一轮 forward 耗时"相比时， 调度器自己成为瓶颈。128 是上游选定的一个**硬编码经验值**，不是 flag、不可配置（：291 的字面量）。**归类：工程保护值，由"调度器不得成为瓶颈"这条约束决定， 既不是理论上界也不是本机扫描结果。** 这也意味着它在不同硬件/模型上未必是最优分界——本仓没有扫描它，只是把 boundary 档设计成确定性地跨过它。

**退化到底省掉了什么**（本讲义推导，顺着 §4 第 2/3 段的分支分析）：它省掉的是 `_compute_prefix_matches`（逐请求树匹配 + in-batch 模拟树查询）与 `_sort_by_longest_prefix`，即上面式子里的两项全部。换句话说，退化不是"降级"， 是**整轮关闭**——这正是"lpm 开着不等于 lpm 在工作"这句话的精确版本。

#### 3.1.2 两个 32:in-batch 去重的阈值语义

`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`(：81-83)与 `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`(：87-89)默认都是 32，都可用同名环境变量覆盖。两者语义**不同**，容易混：

- CHECK：**对真树的匹配长度**若 ≤32，才值得去查模拟树。上游注释解释了为什么不设成 0："too small threshold means we cannot use in-batch prefix caching for short prefixes"(：336-338)——引擎长时间运行后，树里会积累像 "the" 这样的极短公共前缀，导致几乎没有请求满足"真树匹配为 0"；把门槛放到 32 才能让短前缀的新请求仍然进入 in-batch 去重。
- DEPRIORITIZE：**对模拟树的匹配长度**若 ≥32，就把当前请求降权。

**归类**：两者都是**经验值**，由"多短的前缀才算没命中"和"多长的重合才算同伴" 这两个语义判断决定；上游把它们做成环境变量说明它们不是理论量。本仓未扫描（EXP-P04/P08 全程用默认值），所以本仓的 boundary 档结论都带着"两阈值 = 32" 这个隐含定语。

#### 3.1.3 排序的稳定性:lpm 内部嵌着一个 fcfs

`_sort_by_longest_prefix` 用的是 Python 的 `list.sort`，而它的稳定性是语言层面的保证："The `sort()` method is guaranteed to be stable. A sort is stable if it guarantees not to change the relative order of elements that compare equal"（Python 官方文档 The Python Standard Library，`list.sort` 条目）。于是匹配长度相同的请求保持原有到达序——**lpm 内部嵌着一个 fcfs 作 tie-break**。

三条后果（本讲义推导）：

1. 同一组的 $R$ 条请求匹配长度相同时，它们按到达序执行，组内不会乱序。
2. 这解释了 §5.3 的方差证据：lpm 把 seed 间的随机性削平了，因为随机性只剩下 "组之间谁先"这一层，组内被稳定排序固定住。
3. 若把排序改成不稳定（或加随机 tie-break），std 档 lpm 的 hit std 就不会是 fcfs 的 1/4.5(0.006 vs 0.027)。**方差本身是排序在起作用的证据**，而稳定性是这条证据成立的前提。

### 3.2 两阶段认知:0.6B 反劣 → 8B 分位数再分配

同一协议（bench_groups.py：组内共享 1536/2048 前缀、全列表 shuffle 对抗到达序、std=G8×R8@c16 / boundary=G16×R12@c64、3 seeds）先后在两个模型上跑：

**第一阶段（EXP-P04，Qwen3-0.6B，data/derived/exp_p04_fcfs_vs_lpm.csv）**：

| 档 | 策略 | p50(ms) | p99(ms) | hit_frac |
|---|---|---|---|---|
| std | fcfs | 90.0±0.6 | 280±53 | 0.970±0.027 |
| std | lpm | 90.4±0.4 | 265±31 | 0.982±0.006 |
| boundary | fcfs | 249±8 | **661±36** | **0.992±0.011** |
| boundary | lpm | 233±6 | **747±20** | **0.968±0.005** |

std 档按预锁阈值判平（差值与轮间波动同量级）；boundary 档 lpm p99 反劣 13%(>2σ)且 hit −2.4pp——超出设计窗口后排序反成负资产，且没有换来什么。

**第二阶段（EXP-P08，Qwen3-8B，data/derived/exp_p08_8b_fcfs_vs_lpm.csv）**：

| 档 | 策略 | p50(ms) | p99(ms) | hit_frac |
|---|---|---|---|---|
| boundary | fcfs | 6659±274 | 9538±535 | 0.757±0.034 |
| boundary | lpm | **2505±767** | **15656±1293** | **0.934±0.011** |

同一积压档，lpm p50 **−62%**、hit **+17.7pp**，但 p99 **+64%**——不再是 "反劣"，而是延迟从中位数搬到尾部换命中率。

**为什么模型重量改变权衡形态**（排队论直觉，逐步）：

1. 排序的收益 = 每次命中省掉的 prefill 时间。0.6B 一条 2048 token 的 prefill 在 ~10 ms 量级（EXP-P03《命中收益曲线》 miss TTFT 26.84 ms 内含 ~17 ms 地板），8B 在 ~200 ms 量级（EXP-P07《8B 收益曲线》 miss TTFT 228.4 ms）——**命中价值放大约 20 倍**。
2. 排序的代价 = 被延后的组要多等的时间，同样以"别人的 prefill"计价—— **代价也同比放大**。两者放大后，原本淹没在噪声里的重排效果（0.6B std 档判平）变得可测，且分别落在不同分位数上：被聚簇提前的多数请求压低 p50， 被推后的少数组撑爆 p99。
3. 命中率差也被放大：0.6B 时 prefill 快，fcfs 下首请求很快完成入树，同组 miss 窗口短（boundary 档 fcfs hit 0.992）；8B prefill 慢，miss 窗口长（fcfs hit 掉到 0.757），lpm 的串行聚簇才有 17.7pp 的空间。
4. 0.6B boundary 档的"单纯反劣"是同一机制的退化形态：收益侧（1、3）没长大， 代价侧（>128 退化让排序假设中途失效 + in-batch 降权把重复前缀请求延后） 先出现——只剩账单没有收入。（①②两个候选机理未逐请求 trace 分离，EXP-P04 §7 如实列开放。）

#### 3.2.1 用 §2.2 的二分解读这四步

第 3 步其实就是 §2.2 的"聚簇成分"的机理：**miss 窗口的长度决定了聚簇能救回多少工作量**。把它写成一个粗糙但有用的估计（本讲义推导）：

设一组有 $R$ 条请求，首请求的 prefill 耗时 $S_1$，并发下每轮批次能容纳 $m$ 条。在首请求完成之前进入 batch 的同组请求全部 miss。fcfs 下这个数目正比于 $S_1 \times(\text{到达速率})$；8B 的 $S_1$ 大约是 0.6B 的 20 倍，所以 miss 窗口里挤进来的同组请求也多得多。$\text{hit} \approx 1 -(\text{窗口内 miss 数})/(R-1)$——这就是 0.992 与 0.757 的量级差来源。lpm 通过串行化同组请求把窗口内的并发同伴数压到接近 1，于是命中率回到 0.934。

**这段解释是与数据一致的机理推断，不是逐请求 trace 验证过的结论**（EXP-P08 §6 已如实标注）。要验证它需要 server 侧的 per-request 时序，本仓没有采集。

**结论的正确形状**："lpm 好不好"不是标量问题。答案必须带三个定语——模型（prefill 占比）、负载（是否超 128 窗口）、SLO 分位数（p50 还是 p99）。本仓本项目把无定语的"lpm 更好/更差"列为禁用措辞，原因即 P04 与 P08 的结论随定语翻转。

### 3.3 cache-aware 路由:决策伪码与失衡回退

router(sgl-model-gateway,PyPI sglang-router 0.3.2)对每个请求（sgl-model-gateway/src/policies/cache_aware.rs:387 起）:

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

三个必须知道的事实：①近似树存**原始字符**不是 token（tree.rs；文件头注释自述为省 tokenize 开销）——router 的 match_rate 与 engine 的真实命中是两棵不同的树上的两个量，通常同向但不相等；②失衡回退是安全阀：亲和性在偏斜负载下会把请求堆到一张卡，回退用最短队列换回并行度；③**回退的触发条件依赖负载计数**——串行注入（在途恒 ≤1）时 max−min 永远越不过 64 的绝对阈值， 回退**永不触发**，这正是 EXP-P06 击中的机理格。

#### 3.3.1 "近似树"到底近似在哪:四层近似

被测实现的树与 engine 的真树之间隔着四层近似（本讲义整理，均有源码依据）：

| # | 近似 | 源码依据 | 后果 |
|---|---|---|---|
| 1 | **字符级而非 token 级** | cache_aware.rs 文件头："The tree stores raw text characters instead of token IDs to avoid tokenization overhead."(：22-23) | match_rate 与 engine 的 token 命中率不是同一个量；chat template 渲染出的 token 边界完全不可见 |
| 2 | **只记"上次送去了谁"，不记"那边还在不在"** | 插入在每次决策后写回（：471）；无 worker 逐出事件回流 | worker 侧逐出后 router 仍然认为命中，亲和指向一张已经没有这段缓存的卡 |
| 3 | **时间戳按 1/8 概率更新** | tree.rs：601-606，注释自述 "Update timestamp probabilistically (1 in 8 matches) to reduce DashMap contention" | router 树自己的 LRU 逐出也是近似的 |
| 4 | **一个节点多租户时取任意一个** | tree.rs：578-582，`tenant_last_access_time.iter().next()` | 同一段前缀被两张卡都持有时，选谁没有语义保证 |

第 2 层是最重要的一层，而且**上游论文描述的原始设计不是这样**：SGLang 论文附录 A.4 写的是 router 维护 meta-tree、worker 逐出时把事件提交到队列由 router 异步消费（"Should an eviction occur at a worker node， it commits this eviction to a queue， which the router then processes to update the meta-tree during periods of low activity"，arXiv:2312.07104，§A.4）。**被测的 0.3.2 版 gateway 没有这条回流通路**，这是"论文设计 vs 被测实现"的一处实打实的差距，也是 §8.1 表里的一行。

#### 3.3.2 文件头注释与实现的一处分歧

cache_aware.rs 的文件头把低命中路径描述为"路由到树最小的 worker"：

> c. If match rate ≤ cache_threshold: Route to the worker with smallest tree size (most available cache capacity) (:29-30)

而实现里这条分支走的是**最小负载 + 随机 tie-break**（：452-466，§4 第 10 段逐字引用）。两者的差别在 EXP-P06 的机理格上是决定性的："树最小"会在冷启动后自动把新前缀推向另一张卡（因为第一张卡的树已经长大了），而"负载最小"在串行注入下两张卡恒等，只能靠 tie-break 决定。**注释描述的行为恰好可以避免亲和集中，实现的行为不能。**

本讲义不能断言这是 bug 还是有意的演进（注释可能是旧设计的遗留），只能记录： **在 v0.5.18 附带的这份源码里，注释与实现不一致，而不一致的那一处恰好落在本仓实验命中的机理格上**。

### 3.4 EXP-P06 双证伪全程:奇偶巧合与亲和集中

**预注册预测**（由讲义 01 §3.5 的重用距离模型推出，跑前锁定）：双 worker 各限池 8192 token，6 个热前缀（每请求总长 ~2150 字符形态，工作集 ~12900 token）
> 单池、< 双池之和。预测：

- H-rr：round_robin 交替分发 → 每卡看到全部 6 个前缀 → 每卡 $D \approx 6 \times 2150 > 8192$ → 双卡 thrash，hit → ~0。
- H-ca：cache_aware 把每前缀钉在一张卡 → 每卡 ~3 前缀 → $D \approx 6450 < 8192$ → hit → ~1（"路由=扩容"）。

**实测（全部格 3 seeds、seed 间 std=0，data/derived/exp_p06_routing_pool.csv）——两条预测双双反向**：

| 配置 | 策略 | hot_hit | 流量分布（worker 计数器差分） |
|---|---|---|---|
| hot6（偶） | round_robin | **1.0000**（24/24 全命中） | ~50/50(30878/30921) |
| hot6（偶） | cache_aware | **0.0020**(0/24) | **100/0**(61799/0) |
| hot5（奇，对照） | round_robin | 0.0020 | ~50/50 |
| hot5（奇，对照） | cache_aware | 0.0020 | 100/0(51507/0) |

**证伪一（rr 全命中是巧合）**：轮转周期 6 与 worker 数 2 **整除对齐**——严格轮询下第 1/3/5 个前缀永远落卡 A，第 2/4/6 个永远落卡 B，rr 意外成了完美分片， 每卡 3 前缀、$D \approx 6450 < 8192$，全命中。这不是能力而是奇偶巧合—— **hot5 对照臂**（5 与 2 互素，每个前缀轮流落两张卡）打破整除后 rr 立即崩塌， 而流量仍然均分（26589/24918，raw=data/raw/EXP-P06/20260824T165910_hot5_round_robin_s20260824.json）——"分得均匀"与"分得对"被这一格干净剥离。 **证伪二（cache_aware 亲和集中）**：worker 计数器差分直接显示全部流量落在一张卡（61799/0）。机理：串行注入下负载恒 0，失衡回退永不触发（§3.3 第③）； 冷启动时预热流量把 6 个前缀全部记到同一 worker 名下，此后 match_rate 恒高、亲和恒指向该卡——~12900 token 的工作集塞进 8192 的单池，thrash，hit 0.002（残余 3 个 token 是模板头级别的碎屑）。

#### 3.4.1 "分得均匀"与"分得对"的形式化(本讲义推导)

hot5 那一格是本实验最值钱的一格，因为它把两个平时纠缠在一起的量剥离开了。把它们写严格：

- **均衡度**：各 worker 承担的 token 量之比。rr 在两个配置下都是 ~50/50。
- **分散度**：热前缀集合被划分到各 worker 的**划分质量**。定义每卡的重用距离 $D_w =(\text{分到该卡的热前缀数}) \times(\text{单请求 token 量})$， 命中条件是 $D_w \le P$。

rr 在 hot6 下的划分是 $\{1,3,5\}$ / $\{2,4,6\}$——每卡 3 个，$D_w \approx 6450 < 8192$，命中；rr 在 hot5 下没有稳定划分，每个前缀轮流落两张卡，等价于每卡都要装下全部 5 个，$D_w \approx 10750 > 8192$，崩。**两个配置的均衡度完全相同， 命中率差 500 倍。** 结论：

$$\text{均衡度是关于流量的一阶统计量，分散度是关于映射的结构性质；前者不蕴含后者。}$$

**一般化到 $W$ 副本**（本讲义推导）：严格轮询把 $H$ 个热前缀映射到 $W$ 张卡， 映射稳定当且仅当 $W \mid H$；此时每卡分到 $H/W$ 个，命中条件是 $(H/W)\cdot T \le P$。若 $\gcd(H,W) = g < W$，则每个前缀会在 $W/g$ 张卡之间轮流，每卡都必须装下 $H\cdot g/W$ 个甚至更多——$W=2, H=5$ 时 $g=1$，每卡要装全部 5 个。**所以 rr 的"缓存友好"完全依赖 $W \mid H$ 这个数论巧合，而 $H$ 是业务决定的、$W$ 是运维决定的，两者没有理由整除。**

#### 3.4.2 修正后的一般化

cache-aware 亲和等效于扩容的前提是 tenant **分散**在多卡；它的冷启动分配在低负载下会**集中**，此时亲和与容量约束相乘为负。rr 的命中依赖热集数与副本数的整除关系，不可依赖。一句话：**容量受限的多副本里，前缀→副本映射的质量（分散且稳定）比"cache-aware"这个策略标签重要；两种现成策略都不保证这一点**。限度见 §6 条 5。

这条结论并不是本仓的独创发现，而是与分布式前缀调度文献的判断一致——Preble 在引言里把两个极端都点了名：一味均衡会把同前缀请求打散到不同 GPU 重复计算； "a naive solution that always sends requests with shared prefixes to the same GPU would result in imbalanced loads and low overall GPU utilization because the GPU that initially serves a request with a popular prefix will accumulate a huge load of new requests all trying to reuse the calculated prefix KV"(arXiv:2407.00023， §1)。**本仓的贡献不是发现这个张力，而是在一个容量受限的最小机理格里把它测出来， 并证明现成策略的两端都会掉进去。**

值得注意的是，Preble 与本仓观察到的**失效方式不同**：Preble 说的是"负载堆积"（那张卡忙不过来），本仓串行注入下根本没有负载堆积，失效的是**容量**（那张卡的池装不下）。这是同一个亲和集中现象的两种后果，而现成的失衡回退只防前者——它的判据是负载差（64/1.5），完全看不见"目标卡的 KV 池够不够"。**这正是 Preble 的代价函数里那个 $M_i$ 项（逐出代价）存在的理由，而 cache_aware 的决策式里没有对应的项**（§3.3 伪码里找不到任何与池占用有关的量）。

#### 3.4.3 一处必须留着的冲突:随机 tie-break 与 100/0

把源码读到底会遇到一个与实测对不上的地方，本讲义把它原样留下（本讲义推导）。

**源码说什么**：低命中路径（cache_aware.rs：452-466，§4 第 10 段逐字引用）在最小负载的候选里用 `.choose(&mut rand::rng())` 做 tie-break——**随机**。串行注入时两张卡负载恒为 0，所以每个新前缀的首次落点应当是一次公平硬币。

**实测说什么**：每个配置里 6 个（或 5 个）热前缀的首次落点**全部相同**，而且 3 个 seed 完全一致（csv 的 std=0）。更关键的一条细节：hot6 落在 worker1（worker0 一侧为 0），hot5 落在 worker0（worker1 一侧为 0）——**配置之间不同， seed 之间相同**。

**冲突有多硬**（数量级估计）：若首落点独立均匀，单个配置里 6 次全同的概率是 $2\times2^{-6} = 1/32$；要三个 seed 都全同且落在同一张卡，再乘两次 $2^{-6}$ 量级，单配置约 $10^{-5}$；两个配置都如此，再平方。**随机 tie-break 基本被数据排除。**

**候选解释（未核实，按可能性排序）**：

1. **装的轮子与这份 checkout 不是同一个 revision**。安装的 `sglang_router_rs.abi3.so` 里嵌着 `smg::policies::cache_aware` 的 tracing callsite 行号（431/445/455/464/479），与 checkout 里对应位置（436/450/460/ 469/484 一带）系统性偏移约 5 行——说明构建时的源码与本 checkout 有小幅差异， 低命中分支的实现可能就在这几行里不同（例如文件头注释描述的"最小树"版本）。
2. **低命中分支根本没被走到**：若 `request_text` 在 chat 形态下带有某段公共文本（模板/序列化包裹），match_rate 可能在第二个前缀起就越过 0.3，直接走亲和路径指向第一张卡。这需要打印 match_rate 才能证实。
3. **`load()` 在决策时刻并非两卡相等**：若负载来自周期性拉取的快照而非在途计数，两卡可能长期读到不同的陈旧值，min-load 就不是平局。

**下一步的最小探针**（可执行，本仓未做）：在 router 侧打开决策级日志或加一个每请求的 `X-` 响应头记录被选 worker，重跑 hot6 的 cache_aware 臂，把 6 次首落点逐次记下来。这一个探针可以同时区分上面三种解释。EXP-P06 §7 记录了两次日志级尝试均无决策级输出，raw=data/raw/EXP-P06/20260824T175130_cache_aware_debug_rationale.txt。

**为什么要把这一节写进讲义**：一个读完源码就以为自己懂了的人，会直接写下 "cache_aware 用随机 tie-break，所以冷启动是随机的"——而数据说不是。**源码是证据，不是结论；实测是证据，不是结论；两者不一致时，正确的动作是把冲突量化并设计探针，而不是挑一个信。**

#### 3.4.4 理论上正确的修法,以及它就在同一个代码库里

按 §2.5 第三行，"亲和 + 容量上界"是这个问题的标准解：一致性哈希保证映射 **稳定**（增删副本只搬动 $\approx 1/W$ 的键），有界负载保证映射**分散**（没有副本超过 $\lceil cm/W\rceil$）。

被测的同一个 gateway 里已经有一个策略实现了这套组合：`prefix_hash` (sgl-model-gateway/src/policies/prefix_hash.rs)。它的文件头把算法写成五步：

> 1. Extract first N tokens from the request (configurable prefix length) 2. Hash the token sequence using xxhash for fast, stable hashing 3. Use consistent hash ring to find the target worker 4. If worker is overloaded (load > avg * load_factor), find least loaded 5. Return least loaded worker that passes load check, or initial if all overloaded (prefix_hash.rs:9-13)

配置项的默认值是 `prefix_token_count = 256`、`load_factor = 1.25` (config/types.rs：316-321、325-331)。**`load_factor = 1.25` 就是有界负载论文里的 $c = 1+\varepsilon$ 取 $\varepsilon = 0.25$**——理论论文的参数以默认值的形态出现在了实现里（Mirrokni/Thorup/Zadimoghaddam， arXiv:1608.01350）。而且它在第 1 步就用 **token** 而不是字符，消掉了 §3.3.1 的第 1 层近似。

**本仓没有测 prefix_hash**，所以不能主张它在 EXP-P06 的机理格上会赢。能说的只有：它在设计上补齐了 cache_aware 缺的那两样（稳定映射 + 容量/负载上界）， 是这条线最自然的下一格实验。同目录下还有 `power_of_two.rs`（两选一）与 `consistent_hashing.rs`（纯会话亲和），把 §2.5 的三个理论框架凑齐了—— **一个把负载均衡三十年的经典结论逐条实现了一遍的目录，值得单独读一遍。**

### 3.5 魔法数总表:每个数字由谁决定

把本篇出现的常数归一次因（本讲义推导）：

| 数字 | 出处 | 由谁决定 | 依据 |
|---|---|---|---|
| 128（lpm 退化阈值） | schedule_policy.py：291 | **工程保护经验值** | 调度器不得成为瓶颈（§3.1.1）；硬编码，不可配 |
| 32 / 32（in-batch 两阈值） | schedule_policy.py：81-89 | **语义经验值** | "多短算没命中""多长算同伴"；可用环境变量覆盖（§3.1.2） |
| cache_threshold 0.3 | router_args.py：59 | **经验值** | 无理论依据；对照 Preble 的规则是"匹配 > 剩余"即 0.5(§8.1) |
| balance_abs 64 | router_args.py：60 | **经验值** | 防小流量误触发；串行注入下永不触发（§3.4） |
| balance_rel 1.5 | router_args.py：61 | **经验值** | 防大流量下绝对差虚高；与 abs 是"与"关系 |
| eviction_interval 60 s | router_args.py：62 | **工程值** | 近似树的后台逐出周期 |
| max_tree_size $2^{26}$ | router_args.py：63 | **容量约束** | 近似树节点上限（字符计），约 6710 万 |
| load_factor 1.25(prefix_hash) | config/types.rs:329-331 | **理论参数** | 有界负载的 $c=1+\varepsilon$,$\varepsilon=0.25$(§3.4.4) |
| 池 8192(EXP-P06) | 实验协议 | **实测设计值** | 使 6 前缀工作集 ~12900 > 单池、< 双池之和 |
| G16×R12@c64 | 实验协议 | **实测设计值** | 192 > 128，确定性跨过退化窗口（§1.2） |

**读法**：整张表里只有最后一行的 `load_factor` 有理论出处，`max_tree_size` 是容量约束，其余全是经验值或实验设计值。**这本身就是一条结论：cache-aware 路由这一层目前主要靠经验参数运行，而经验参数在没被扫描过的负载上没有任何保证。**

## 4. 代码逐段走读

按"引擎调度 → router 决策 → 测量端"的顺序。上游引用逐字拷贝（上游文件相对 /root/repos/sglang-v0.5.18）。

**第 1 段 · lpm 的重载保护**(schedule_policy.py：290-294)

```python
    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy
```

角色：整个 P04/P08 边界档设计的靶点。128 是硬编码，不是 flag。改错会怎样： 若没有这条保护，重载下每轮调度对几百个请求做树匹配 + 排序，调度器自身成为瓶颈；而有了它，"lpm 开着"不等于"lpm 在工作"——boundary 档里策略在 fcfs 与 lpm 之间随队列长度抖动，排序建立的聚簇假设中途失效（EXP-P04 §6 候选机理①）。

**注意它是逐轮判定的**：队列长度每轮都会变（有请求被取走、有新请求到达）， 所以在 boundary 档里策略不是"关掉了"，而是**在 fcfs 与 lpm 之间来回切换**。这比"一直关着"更糟：一直关着至少行为一致，来回切换意味着上一轮排好的顺序在下一轮可能被完全忽略，而 in-batch 降权造成的队尾堆积却留了下来。

**第 2 段 · 每轮调度的入口**(schedule_policy.py：237-252)

```python
    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills).
        if (
            not isinstance(policy, CacheAwarePolicy)
            and self.tree_cache.supports_fast_match_prefix()
            and get_disagg().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r, include_req=True)
```

角色：两个变量的分工在这里定下——`policy`（本轮生效，可能已被第 1 段退化）与 `self.policy`（配置）。中间那段 if 是给 cache-agnostic 策略补一次匹配、只供负载快照使用的，但**它在本仓的配置下根本不执行**：守卫条件里的 `supports_fast_match_prefix()` 在基类返回 `False` (mem_cache/base_prefix_cache.py：278-279)，而 v0.5.18 的 `RadixCache` **没有覆写它**（全仓仅此一处定义，grep 可验）。

这条冷知识对本仓有两个实际后果（本讲义推导）：①**fcfs 臂真的一次树匹配都不做**， 两臂的差别就是干净的"匹配+排序 vs 什么都不做"，这正是把它当单变量对照的依据； ②别把这段 if 当成"fcfs 也有开销"的理由——它是为将来支持快速匹配的缓存实现预留的路径，当前是死代码。**读到一个 if 时先确认它的守卫会不会为真，是读调度器代码的基本功。**

**第 3 段 · 每轮调度的分派**(schedule_policy.py：254-268)

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

角色：fcfs 的"实现"是一个提前 return（什么都不做，队列保持到达序）——两臂对比的本质是"排序 vs 不排序"，而非两套算法。注意 `policy`（本轮生效，可能已被第 1 段退化）与 `self.policy`（配置）是两个变量，fcfs 分支查的是配置——配置为 fcfs 时连匹配都不做，per-request 的 num_matched_prefix_tokens 走另一条快照路径（：246-252）。

**这里还藏着一个 boundary 档的关键细节**（本讲义推导）：第 1 段退化返回的是 `CacheAgnosticPolicy.FCFS`，赋给局部变量 `policy`；但本段第一个 if 查的是 `self.policy`（配置值，仍是 LPM），所以**不会**从这里提前 return。流程走到第二个 if，`isinstance(policy, CacheAwarePolicy)` 为假（已退化），于是 `_compute_prefix_matches` 不执行、排序也不执行，最后落到 else 链里的 `policy == CacheAgnosticPolicy.FCFS -> pass`(：273-275)。

**所以退化后的净效果与配置 fcfs 完全相同：一次匹配都不做、一次排序都不做。** 再加上第 2 段那条永假的守卫，可以下一个干净的结论：在本仓的配置下， "lpm 且队列 >128 的那一轮"与"配置 fcfs 的那一轮"在调度器里走的是不同的分支但产生**完全相同的行为**。boundary 档的 lpm 臂因此是一个逐轮在两种行为之间抖动的混合臂——这就是 EXP-P04 §6 候选机理①"排序假设中途失效"的精确含义。

**第 4 段 · in-batch 前缀去重**(schedule_policy.py：339-358)

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

角色：等待队列自己也有一棵（模拟）radix 树。真树 miss（匹配 ≤32）但队列里已有同前缀请求（模拟树命中 ≥32）的请求被降权——"让开路者先种树"。两个阈值都默认 32（：81-89，环境变量可调）。改错会怎样：去掉降权，同批 N 条同前缀请求并发跑，同一前缀被重算 N 遍；但在 G16×R12 的大批量同组负载下，这条路径也把大量请求推到队尾——EXP-P04 §6 的候选机理②，收益机制在超窗负载下的另一面。

**顺带记一个可用的实验旋钮**：第 11 行的 `SGLANG_RADIX_FORCE_MISS` 环境变量可以把匹配结果强制清零。这正是讲义 01 §3.4.1 说的那个缺失的第三臂（"前缀相同但命中被强制置零"）的现成开关。本仓未用。

**第 5 段 · 排序键**(schedule_policy.py：373-384)

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

角色：lpm 的全部"算法"就是这一个 sort：匹配长的在前（负号），被降权的最后（inf）。它是"有效 prefill 最短优先"的近似（§2 排队论直觉）。改错会怎样： 去掉负号即变成"最短匹配优先"，命中优势反转；Python sort 稳定，同匹配长度的请求保持到达序——lpm 内部嵌着 fcfs 作 tie-break，这也是它对到达序方差的削平作用（§5 的方差证据）可解释的原因之一。

**再补一句量纲上的观察**（本讲义推导）：排序键是 **token 数**，不是**时间**。两者只有在"每 token 的 prefill 成本相同"时才等价。同一模型、同一批次内这近似成立；但跨请求长度差异极大的真实流量里，一条 8000 token 匹配 7000 的请求（剩 1000 要算）会排在一条 1200 token 匹配 1100 的请求（剩 100 要算）前面——**按剩余工作量本该反过来**。真正的 SPT 键应当是 $n-k$ 而不是 $k$。上游选 $k$ 大概是因为它同时最大化命中率（Theorem 3.1 的目标），而不是最小化等待。 **目标函数不同，不是实现疏漏**；但这解释了为什么 lpm 只是 SPT 的粗近似（§2.3 第 1 点）。

**第 6 段 · router 失衡回退**(sgl-model-gateway/src/policies/cache_aware.rs:412-424)

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

角色：亲和与均衡的仲裁点，**先问负载再问缓存**。两个阈值（默认 64 / 1.5， python 侧 router_args）是"与"关系：绝对差要大**且**相对比要大。改错会怎样： 换成"或"，小流量下轻微不均就放弃亲和，命中率被随机打散；EXP-P06 的教学点相反——串行注入下 max−min ≤ 1 远小于 64，这个分支从未执行，亲和集中无人纠偏（§3.4 证伪二）。

**这个判据里缺了什么**（本讲义推导）：它只看**负载**，不看**容量**。目标 worker 的 KV 池已经装不下它被分配的工作集这件事，在这两个阈值里完全不可见。对照 Preble 的三项代价 $L_i + M_i + P_i$——其中 $M_i$ 正是"为了跑这条请求要逐出多少东西、逐出的代价有多大"(arXiv:2407.00023，§3.2)——**cache_aware 的决策式里没有任何对应于 $M_i$ 的项**。EXP-P06 崩塌的直接原因就是这个缺失项。

**第 7 段 · router 亲和路径**(cache_aware.rs:436-450, 469-471)

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

角色：匹配对象是 `text` 的**字符**；`result.tenant` 是近似树记录的"这段前缀上次被送去的 worker"——树的 tenant 归属就是路由记忆，`tree.insert` 在**每次决策后**写回，于是首个请求落在哪张卡（冷启动、低负载时由 min-load 分支决定） 会被后续同前缀请求持续强化——这就是亲和集中的自增强回路。改错会怎样：阈值 0.3 调成 0，任何一丁点字符重合都触发亲和，不同前缀会被模板头的公共字符错误地钉到同一张卡。

**分母的语义值得单独看一眼**：`match_rate = matched_char_count / input_char_count`，分母是**整条输入**的字符数，不是"可共享部分"的字符数。所以同一段系统提示词，在短用户问题后面 match_rate 高，在长用户问题后面 match_rate 低——**同一个前缀会因为后缀长度而时而触发亲和、时而不触发**。对照 Preble 的规则"matched > remaining"（等价于 match_rate > 0.5， arXiv:2407.00023，§3.2），两者形式相同而阈值不同：0.3 比 0.5 更偏向亲和。这两个阈值背后是同一个权衡，一个取了经验值，一个从"省下的算力是否超过新增的算力"推出来。

**第 8 段 · 低命中路径的随机 tie-break**(cache_aware.rs：451-467)

```rust
            } else {
                // Low cache match: use worker with minimum load. Tie break randomly.
                // Snapshot load() (live atomic count of load). Without snapshot
                // there could be no workers found matching min_load because of
                // load update.
                let loads: Vec<(usize, usize)> = healthy_indices
                    .iter()
                    .map(|&idx| (idx, workers[idx].load()))
                    .collect();
                let min_load = loads.iter().map(|&(_, load)| load).min()?;
                loads
                    .iter()
                    .copied()
                    .filter(|&(_, load)| load == min_load)
                    .map(|(idx, _)| idx)
                    .choose(&mut rand::rng())
            };
```

角色：§3.4.3 那处冲突的源码原文。**注释与实现在这里是一致的**（都说随机 tie-break），不一致的是**实测**：6 个热前缀的首落点在 3 个 seed 上完全相同， 与随机相矛盾。同时注意 `load()` 的注释自述是"live atomic count of load"——在途计数，串行注入下两卡恒为 0，平局必然发生。**读到这一段就应当预测"冷启动是随机的"，而数据说不是，这就是把探针设计出来的时刻**（§3.4.3 末尾）。

**第 9 段 · 文件头注释：一份与实现不符的设计说明**(cache_aware.rs：25-32)

```
    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow
```

角色：c 项与第 8 段的实现不符——注释说"树最小"，实现是"负载最小 + 随机" (§3.3.2)。这一段留在讲义里不是为了挑上游的错，而是为了立一条读码纪律： **注释是作者的意图，实现才是运行的东西；当你的实验结果与注释不符时，先去读实现，再去怀疑实验。** 本仓 EXP-P06 的预注册预测正是基于"路由会分散"这个直觉（与注释一致），而实现给了另一种行为。

**第 10 段 · 近似树的两处"够用就好"**(sgl-model-gateway/src/policies/tree.rs：601-607)

```rust
        // Update timestamp probabilistically (1 in 8 matches) to reduce DashMap contention.
        // LRU eviction doesn't need perfect accuracy - approximate timestamps suffice.
        let epoch = get_epoch();
        if epoch & 0x7 == 0 {
            curr.tenant_last_access_time
                .insert(Arc::clone(&tenant), epoch);
        }
```

角色：§3.3.1 第 3 层近似的原文。router 树的 LRU 时间戳只有 1/8 的机会被更新， 理由写在注释里：减少 DashMap 竞争，而"LRU 逐出不需要精确"。**这是一个典型的 "在正确性不敏感处买性能"的设计**，但它同时意味着 router 树的逐出行为是随机化的——把讲义 01 §3.5 的确定性重用距离模型直接搬到 router 树上会出错。本仓的 EXP-P06 里 router 树远未触及 $2^{26}$ 的上限，这层近似没有被激活，所以不影响结论；但换成长跑的生产场景就要重新考虑。

**第 11 段 · router 的默认值都在一个 dataclass 里**(sgl-model-gateway/bindings/python/src/sglang_router/router_args.py：59-63)

```python
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    eviction_interval_secs: int = 60
    max_tree_size: int = 2**26
```

角色：§3.5 魔法数表里 router 那几行的唯一出处，也是"本仓全程用默认值"这句定语的依据。安装的 wheel 里同名文件的对应字段取值相同（0.3 / 64 / 1.5 / 60 / $2^{26}$），所以本仓实验确实跑在这组默认值上。**注意 `max_tree_size` 的单位是近似树的节点数上限，而近似树是字符级的**——$2^{26}$ 个节点在字符粒度下并不算多， 长跑场景需要按实际 prompt 体量重新估。

**第 12 段 · 测量端的对抗设计与命中口径**(scripts/bench_groups.py：60-67, 79-82)

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

角色：shuffle 是本实验的灵魂——按组顺序注入时 fcfs 也天然聚簇，两策略没有差异空间；打散后"把同前缀请求重新聚到一起"的能力才归属调度器。hit_fraction 的分母扣掉 G（每组首请求必 miss，树里尚无该前缀），命中率才能跨配置比较。改错会怎样：分母不扣 G，理论上限变成（N·prefix_len），满命中也只能到（N−G）/N ≈ 0.92（G16×R12 时），两臂差异被系统性压缩，0.992 vs 0.968 这种 2σ 分离可能就判不出来了。

**"必 miss"这个词要精确**（本讲义推导）：严格说每组首请求必 miss 的前提是 **组间前缀互不共享**——`ids_of(prefix_len, seed*100+g)` 对不同 $g$ 用不同种子生成随机 token，首 token 相同的概率约 1/99000，所以组间共享前缀实际为 0。若换成真实语料（组间会共享系统提示词），分母扣 $G$ 就变成了低估，hit_frac 会虚高。**这是一个只在合成负载下成立的口径**。

**第 13 段 · 中间层改写的防线**(scripts/bench_route_pool.py：47-59)

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

角色：EXP-P06 首轮整批作废换来的防线。router 严格按 OpenAI schema 重序列化， **丢弃 `input_ids` 扩展字段**——首轮全部请求静默退化成 ~10 token（靠 cached=8 与 worker 计数增量反推才发现）。修正三件套：负载改文本形态（恰好也是 cache_aware 近似树的匹配对象，对齐了被测机制）、响应 `prompt_tokens` 硬 gate（低于下限直接抛异常，实验 fail-fast 而非静默继续）、每臂前置校验 router 指标的 policy 标签（首轮另一个事故：router 用 setproctitle 改名致旧身份校验落空，第一支 router 存活跨臂，cache_aware 臂实际在跑 rr）。改错会怎样：去掉这个 gate，拿到的仍是一套"看起来正常"的 json——静默退化的实验比失败的实验危险得多。

**这个事故的一般形式**（本讲义推导）：**测量链路上每一跳都可能改写你的请求， 而改写通常是静默的**。防线只有一种形式可靠：对**响应内**的硬计数设 gate。不能依赖"我发出去的是什么"——那只是你的意图；只能依赖"服务端说它收到了多少 token"。同理，第 12 段的 `cached` 逐请求校验、讲义 01 §4 第 4 段的 `cached == prefix_len` 硬校验，都是同一条纪律的实例。

## 5. 实验数据怎么读

### 5.1 fig4(8B 调度权衡)的读法

**figures/fig4_p08_sched_tradeoff.png**：横条图，p50 与 p99 两组、 fcfs/lpm 两色并排，只画 boundary 档（std 档判平，混入会稀释结论）。读法： lpm 的 p50 条比 fcfs 短（2505 vs 6659，−62%）而 p99 条比 fcfs 长（15656 vs 9538，+64%）——"换来什么/付出什么"同图可见，这就是"分位数再分配"的图形形态。误差条是 3 seeds 间 std；lpm p50 的 ±767 之大不是测量差，是机制的一部分：哪些组先被聚簇决定中位请求落点，对 seed 敏感（EXP-P08 §7）。

### 5.2 图上没有的那一列:makespan

`data/derived/exp_p08_8b_fcfs_vs_lpm.csv` 里还有一列 `dur_s_mean`（该档全部 192 条请求跑完的墙钟时间，3 seeds 均值）：boundary 档 fcfs **26.86 s**、 lpm **20.41 s**（本讲义读自该 csv）。同一档的 EXP-P04(0.6B)对应列是 fcfs 3.27 s、lpm 3.34 s(data/derived/exp_p04_fcfs_vs_lpm.csv)。

**这一列为什么重要**（本讲义推导）：在闭环、固定请求总数的负载下，makespan 就是吞吐的倒数（Little 律的直接推论：$X = N_{\text{total}}/\text{makespan}$）。于是：

- 8B boundary 档 lpm 的 makespan 更短 → **系统总工作量确实减少了**，不是纯搬运。这正是 §2.2"聚簇成分"的直接证据：hit +17.7pp 把 prefill 工作从系统里删掉了一部分。
- 0.6B boundary 档 lpm 的 makespan 略长 → **没有工作量可删**（fcfs hit 已 0.992），只剩零和重排，还倒贴了排序与降权的开销。

**所以"分位数再分配"这个说法只描述了一半**：8B 上 lpm 同时做了再分配（p99 涨） 与总量削减（makespan 降），0.6B 上只剩再分配的成本面。把这两档的 makespan 并排放，比只看 p50/p99 更能说清 lpm 到底做了什么。

**必须带的定语**：makespan 是**这一档 192 条请求**的墙钟，不是稳态吞吐； 它对负载结构（组数/组大小/并发）敏感，换协议不应期望复现。

### 5.3 方差本身是机制证据

EXP-P04 §7：std 档 fcfs 的 hit std=0.027 ≫ lpm 的 0.006——fcfs 命中依赖 shuffle 出的到达序，seed 间起伏；lpm 排序削平了这种随机性。读表时不要只看均值列：两臂方差的量级差直接指认"排序在起作用"，即使均值差判平。

**这条证据的成立依赖 §3.1.3 的稳定排序**：如果 tie-break 是随机的，lpm 的方差就不会比 fcfs 小。所以"方差小 ⇒ 排序在起作用"这个推理链完整地是： 稳定排序 → 同匹配长度保持到达序 → 组内顺序确定 → 只剩组间随机 → 方差降低。少了任何一环这条读法都不成立。

**同一现象在 8B 上翻转**：EXP-P08 boundary 档 lpm 的 p50 std 是 ±767， 远大于 fcfs 的 ±274。这不矛盾——**hit 的方差降低是排序起作用，p50 的方差升高是排序的收益本身对 seed 敏感**（哪些组先被聚簇决定了中位请求落在哪）。两个方差说的是两件事，读的时候要认准是哪一列的 std。

### 5.4 P06 的表怎么读

**data/derived/exp_p06_routing_pool.csv**：关键列是 worker0/1_traffic_mean（两 worker `prompt_tokens_total` 计数器的 before/ after 差分，3 seeds 均值）——hot6_even 的 cache_aware 行是 0 / 61658， 流量 100/0 不是推断而是计数器直读。hot_cached 逐请求序列在 raw 里：rr@hot6 是 [1597, 1625, 1609, ...](全命中;数值大于名义 1536 是文本重编码漂移, 命中率按 min(1.0, c/prefix_len) 钳制，bench_route_pool.py：109)，崩塌臂是清一色 [3, 3, 3, ...](只剩模板头碎屑)。

**还有一件容易被跳过的事**：两个 cache_aware 行粘住的**不是同一张卡**—— hot6_even 的流量全在 worker1（worker0 列为 0），hot5_odd 的流量全在 worker0（worker1 列为 0）。同一策略、同一注入方式，两个配置的 "粘住的那张卡"不同，但每个配置内部 3 个 seed 完全一致。§3.4.3 把这条观察当作证据用：它同时排除了"随机 tie-break"（seed 间会变）与"固定第一个 worker"（配置间会一样），是那处冲突里最有信息量的一格。

### 5.5 std=0 怎么理解

P05/P06 全部格 seed 间 std=0，不是"没测出波动"，而是协议确定性的体现—— temperature=0、串行注入、seed 只改变 token 内容不改变结构（热集大小/轮转序）， 逐出与路由的行为逐次完全相同。看到 std=0 应当去查协议是否确定性，而不是怀疑数据造假；反之，把这种格子的结论外推到并发/随机到达时必须重新测。

**在 P06 上，std=0 还额外承担了一个论证角色**：它把"随机 tie-break"这个源码读出来的行为证伪了（§3.4.3）。**一个本来只用于表达"协议干净"的统计量，反过来成了源码与实测冲突的关键证据**——这是保留完整逐 seed 数据的价值。

### 5.6 防坑清单(本组实验特有)

①每策略**各起一次 worker**（而非热切换），杜绝上一策略的树残留；②臂前 flush + 预热；③路由臂的 policy 标签前置 gate 与 prompt_tokens 硬 gate（§4 第 13 段）；④对照臂设计——hot5 奇数臂专为打破整除关系而设，是"用对照组杀死巧合解释"的教科书局；⑤3 seeds 换 seed 重跑而非重复同 seed（重复同 seed 的 std=0 无信息量）。

**机理账**：rr@hot6 每卡看 3 个前缀，重用距离 $D \approx 3 \times 2150 = 6450 < 8192$ → 命中；rr@hot5 每卡要装全部 5 个前缀（$D \approx 5\times2150 = 10750 > 8192$）→ 崩；cache_aware 单卡装 6 个（$\approx 12900 > 8192$） → 崩。三个格子共用讲义 01 §3.5 的同一条不等式，这正是"预注册预测可以被推导出来"（哪怕预测错了，错的是前提不是推导）的示范。

## 6. 误区与边界

1. **"开 lpm 总没错"——被 EXP-P04 证伪**。轻载（std 档）到达序已足够友好， 判平；超 128 窗口（boundary 档，0.6B）p99 反劣 13%、hit −2.4pp，单纯负资产。cache-aware 调度有收益窗口，窗口外反噬。
2. **"调度策略有标量优劣"——被 EXP-P08 修正**。8B 积压档 lpm p50 −62% 与 p99 +64% 同时成立；"哪个好"取决于 SLO 定义在哪个分位数。任何不带模型/ 负载/分位数定语的比较结论都应当被追问口径。
3. **"cache-aware 路由 = 扩大有效缓存容量"——本仓预注册假设，被 EXP-P06 证伪**。亲和等效扩容的前提是映射分散，而冷启动 + 低负载下 cache_aware 恰恰把全部热前缀集中到一张卡（计数器 100/0）。同实验的镜像误区："rr 全命中说明 rr 对缓存友好"——那是热集数与副本数整除的巧合，hot5 对照立即戳破。
4. **"router 的 match_rate 就是命中率"**。router 树是字符级近似（决定去哪张卡），engine 树是 token 级真值（决定省不省算）；router 命中只是 engine 命中的必要前置。报告命中率永远以 engine 侧计数为准（theory/02 §3）。
5. **"论文证明了 lpm 最优，所以 lpm 一定更好"**。Theorem 3.1 有三个前提： 离线批次、缓存 ≥ 最大请求长度、排序真的发生（§2.1 的表）。本仓 boundary 档三条全破。**定理没错，是应用条件被跨出去了**——这是读理论结论时最容易犯的错。
6. **"lpm 只是把等待从 p50 搬到 p99"**。这只描述了零和的那一半。8B 上 lpm 同时把 makespan 从 26.86 s 降到 20.41 s(§5.2)，说明总工作量真的减少了。只讲再分配会低估 cache-aware 调度，只讲工作量减少会掩盖 p99 的代价。
7. **"失衡回退能兜住亲和集中"**。它只看负载差（64/1.5），看不见目标卡的 KV 池够不够（§4 第 6 段）。串行注入下它一次都没触发；即使触发，它纠正的也是负载而不是容量。
8. **"策略进程活着就是策略生效"**。EXP-P06 首轮的事故：router 用 setproctitle 改名，旧身份校验落空，第一支 router 跨臂存活，cache_aware 臂实际跑的是 rr（§4 第 13 段）。策略标签必须从 router 自己的指标里前置校验。
9. **"源码怎么写就一定怎么跑"**。第 8 段的随机 tie-break 与实测的 100/0 直接冲突，数量级估计把随机解释基本排除（§3.4.3）。**源码与二进制之间还隔着一次构建**，行号偏移已经提示两者不是同一个 revision。
10. **适用边界**：①调度结论限 0.6B(EXP-P04)与 8B(EXP-P08)各自的负载档， 两记录互为限定；boundary 档机理（退化开关 vs in-batch 降权；fcfs 并发交错 miss）为与数据一致的推断，未逐请求 trace，记录里如实标注开放。 ②P06 是**串行、容量受限、冷启动**的机理格：并发到达会激活失衡回退、打破严格轮转的整除结构，高并发行为未测，不外推；cache_aware 冷启动集中于一卡的内部路径未定——源码读出的是随机 tie-break，与 std=0 的实测冲突（§3.4.3），两次日志级尝试均无决策级输出（EXP-P06 §7， raw=data/raw/EXP-P06/20260824T175130_cache_aware_debug_rationale.txt）。 ③双副本性能矩阵（吞吐/延迟 sweep）未执行，本讲义不含任何路由性能结论。 ④`prefix_hash`、`power_of_two`、`consistent_hashing` 三种策略与 lfu/ dfs-weight 调度全部未测，§3.4.4 的"理论上正确的修法"是设计层面的判断， 不是实测结论。

## 7. 连环追问

1. **Q：lpm 排序的 key 是什么？** 每请求对全局 radix 树的匹配长度取负值升序（即匹配长的在前）；in-batch 降权者置 inf 沉底（schedule_policy.py：373-384）。
2. **Q：fcfs 在代码里长什么样？** 一个提前 return——不排序，队列保持到达序（：254-259）。所以两臂对比干净： 变量只有"是否重排"。注意配置为 fcfs 时仍会为负载快照做一次匹配（：246-252，§4 第 2 段），只是结果不参与排序。
3. **Q：为什么 boundary 档取 192 请求、并发 64？** 为了确定性地跨过 lpm 的 128 等待队列退化窗口（：291），把"策略在窗口内/外" 变成实验变量（EXP-P04 §1 预注册）。
4. **Q：为什么 shuffle 后再注入？** 按组连续注入时 fcfs 也天然聚簇，策略差异没有表达空间；shuffle 把"聚簇" 的功劳完全归属调度器（bench_groups.py：67 注释，§4 第 12 段）。
5. **Q：8B 下 fcfs 命中率为什么只有 0.757？** 并发交错：同组首请求的 KV 要等它**完成**才插树（讲义 01 §3.2 的 cache_finished_req），8B prefill 慢、miss 窗口长，后续同组请求赶在树种好之前进了 batch（EXP-P08 §6，机理推断与 hit/流量数据一致，未逐请求 trace）。
6. **Q：lpm 换命中率的代价为什么落在 p99 而不是均匀摊开？** 排序是全序重排：收益摊给被提前的多数（p50），代价集中给排最后的少数组（整组延后），分布两端被同时拉开——SPT 类调度的教科书性质。补一句量化： 代价不是纯零和，因为同一次重排还提高了命中率、缩短了 makespan(§5.2)。
7. **Q：失衡回退的两个阈值为什么是"与"关系？** 绝对差（64）防小流量误触发，相对比（1.5）防大流量下绝对差虚高；单用任何一个都会在某个流量段错误触发/漏触发（cache_aware.rs：412-413）。
8. **Q：怎么证明 cache_aware 臂真的在跑 cache_aware？** 每臂前置校验 router prometheus 的 `smg_worker_selection_total{policy=...}` 标签（首轮事故后加的 gate：进程存活 ≠ 策略正确）。进程身份还要容忍 setproctitle 改名形态（svc.sh 放行三形态）。
9. **Q："映射分散度>策略标签"能推广到几副本？** 不等式形式可推广：每卡命中 ⇔ 该卡分到的热前缀重用距离 ≤ 单卡池。$W$ 副本下 rr 的"巧合分片"条件是 $W \mid H$(§3.4.1)，$\gcd(H,W)<W$ 时每卡都要装更多前缀，更脆；分散且稳定的映射（一致性哈希按前缀分片 + 负载上界）是通解方向——但本仓只测了 2 副本，推广是推导不是实测。
10. **Q：engine 的 lpm 与 router 的 cache_aware 会互相干扰吗？** 两层 cache-aware 叠加行为本仓未测（EXP-P04 §8 明确列为扩展问题）。可以说的只有机制事实：router 决定"谁看见请求"，engine 决定"看见后怎么排"。
11. **Q：SGLang 论文不是证明了 lpm 最优吗，为什么你们测出反劣？** Theorem 3.1 是**离线**结论，而且要求缓存 ≥ 最大请求长度；论文自己写了 "In the online case， the DFS order will be disrupted"(arXiv:2312.07104， §3)。本仓 boundary 档在线到达、缓存被在途请求分掉、而且队列 >128 时排序根本不执行——三个前提全破（§2.1 的表）。**定理与实测不冲突，是适用域不同。**
12. **Q：lpm 到底是不是 SPT？** 是粗近似，四处不同：排序键是匹配长度不是剩余工作量（$k$ 而非 $n-k$， §4 第 5 段）；排序键随执行漂移；被延后的是整组而非独立作业；本仓是闭环系统不是 M/G/1。所以 SPT 文献的定量结论不能直接搬（§2.3）。
13. **Q：Kleinrock 守恒律说重排是零和，你们却说总工作量减少了，矛盾吗？** 不矛盾。守恒律的适用类要求**调度决策不使用服务时间信息**；lpm 的排序键与有效服务时间相关，已经出了这个类。更根本的是，守恒律假设作业大小固定， 而 cache-aware 聚簇会**改变作业大小**（命中的 token 不用算）。makespan 26.86→20.41 s 就是这个"类外"效应的度量（§2.2、§5.2）。
14. **Q：如果 fcfs 下命中率已经很高，还该不该开 lpm？** 按 §2.2 的二分：不该。命中率饱和意味着聚簇成分接近零，只剩零和的重排， 尾延迟必然变差而没有补偿——这正是 0.6B boundary 档的实测形态。**开关的判据是"fcfs 下的命中率离饱和有多远"，不是"要不要 cache-aware"。**
15. **Q：cache_threshold 为什么是 0.3？有理论依据吗？** 没有检索到理论依据，是经验默认值（router_args.py：59）。作为对照，Preble 的规则是"匹配 token 数 > 剩余 token 数"才选择亲和（arXiv:2407.00023， §3.2），等价于阈值 0.5，且它是从"省下的算力是否超过新增的算力"推出来的。 0.3 比 0.5 更偏向亲和——在容量充裕时更好，在容量受限时更危险。
16. **Q：那按理论该怎么改路由？** 亲和 + 容量上界：一致性哈希保证映射稳定，有界负载保证不集中（arXiv:1608.01350）。这不是空想——同一个 gateway 里的 `prefix_hash` 策略就是这么做的：prefix token 哈希 + 一致性哈希环 + `load_factor` 超载绕行，默认 `load_factor = 1.25` 恰是有界负载的 $c=1+\varepsilon$ (§3.4.4)。**本仓未测，所以只能说它在设计上补齐了缺口，不能说它会赢。**
17. **压力问 Q：lpm p50=2505±767，std 这么大，−62% 可信吗？** 诚实答：3 seeds 的区间（约 1738-3272）与 fcfs 的 6659±274 无重叠，方向与量级站得住；但 ±767 说明中位数本身对聚簇顺序敏感，−62% 是三轮均值的点估计，换负载结构（组数/组大小/并发）不应期望复现这个具体数值。方差大是机制（哪组先被聚簇）而非噪声，这一点记录里如实标注（EXP-P08 §7）。补一条独立佐证：makespan 也从 26.86 降到 20.41 s(§5.2)，两个方向一致的量同时改善，比单看 p50 更可信。
18. **压力问 Q：P06 的结论会不会只是 sglang-router 0.3.2 一个版本的 bug？** 可能性无法排除——冷启动集中的内部路径未定（源码读出的是随机 tie-break， 与 3 seeds 全同的实测冲突，§3.4.3；日志级两次尝试无决策输出）， 上游新一代实现（experimental/sgl-router 的 cache_aware_zmq，block-hash of token ids + ZMQ 事件）已是不同设计。但本仓结论的一般化部分（"亲和等效扩容以映射分散为前提；整除巧合不可依赖"）是从重用距离不等式推出的， 不依赖该版本的具体实现；版本特定的部分（冷启动全钉一卡）已限定版本号并保留为潜在上游 issue 素材（EXP-P06 §8）。
19. **压力问 Q：你读了源码却解释不了自己的数据，这不是说明实验有问题吗？** 诚实答：两种可能都在，而且本仓给出了区分它们的探针（§3.4.3 末尾：决策级日志或响应头记录被选 worker）。在探针跑通之前，能确定的是**观测事实**（流量 100/0、3 seeds 全同、两配置粘住的卡不同）和**源码事实**（低命中路径写的是随机 tie-break），以及两者不相容。**把不相容原样留在讲义里， 比挑一边圆过去更接近工程现实**——生产系统里"源码这么写但行为不是这样" 是常态，通常答案就在构建版本、配置注入或某条没读到的分支里。

## 8. 工业对照与延伸

### 8.1 论文/文档声称 vs 本项目实测:逐条对照

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | SGLang Theorem 3.1：最长共享前缀优先等价于 DFS，离线且缓存 ≥ 最大请求长度时取到最优命中率 | 0.6B boundary 档 lpm hit **低于** fcfs(0.968 vs 0.992，EXP-P04) | **不矛盾，是适用域外**：三个前提（离线/缓存足够/排序真的发生）在 boundary 档全破（§2.1）。论文自己写了在线情形 DFS 序会被打断 |
| 2 | SGLang §3：排序按 matched prefix length 降序 | 源码 `_sort_by_longest_prefix` 完全一致（：373-384） | **一致**。本讲义补出的是量纲观察：键是 token 数不是剩余工作量，$k$ 而非 $n-k$（§4 第 5 段） |
| 3 | SGLang §6.3 消融："FCFS Schedule"相对 cache-aware 调度性能更差 | 0.6B boundary 档方向**相反**（lpm p99 反劣 13%）；8B boundary 档方向**一致**(p50 −62%) | 负载与规模不同。论文的消融跑在有充足复用机会、缓存不吃紧的 benchmark 上；本仓 boundary 档专门跨出了 128 窗口。**同一开关在两个 regime 里符号相反，这正是本仓两阶段认知的价值** |
| 4 | SGLang §A.4：多副本设计为 router meta-tree + worker 逐出事件回流 | 被测 0.3.2 gateway 用字符级近似树，**无逐出事件回流**（§3.3.1 第 2 层） | 论文设计与被测实现是两代东西。缺了回流通路，router 会对着一张已经逐出了该前缀的卡继续保持亲和——EXP-P06 崩塌的放大器 |
| 5 | cache_aware.rs 文件头 c 项：低命中时路由到"树最小"的 worker(：29-30) | 实现走的是"负载最小 + 随机 tie-break"(：452-466) | **注释与实现不符**，且分歧点恰在本仓机理格上。注释描述的行为可避免亲和集中，实现的不能（§3.3.2） |
| 6 | cache_aware.rs：452 注释：低命中路径"Tie break randomly" | 6（或 5）个热前缀首落点 3 seeds 全同，两配置各粘一张卡（std=0） | **直接冲突**。随机解释在数量级上被排除（§3.4.3）。候选原因：wheel 与 checkout 非同一 revision（tracing 行号系统性偏移约 5 行）、亲和路径提前触发、load() 非平局。**未核实，留探针** |
| 7 | Preble §1：一味把同前缀请求送同一 GPU 会造成负载不均与低利用率 | 本仓测到的是**容量**失效（单池装不下工作集），不是负载堆积（串行注入下负载恒 0） | 同一现象的两种后果。现成的失衡回退只防负载（64/1.5），看不见容量——对应 Preble 代价函数里缺失的 $M_i$ 项（§3.4.2） |
| 8 | Preble §3.2：匹配 token 数 > 剩余 token 数时才选亲和（等价阈值 0.5） | 被测实现用固定 `cache_threshold = 0.3`(router_args.py：59) | 形式相同、来源不同：Preble 从"省下的算力 vs 新增的算力"推出；0.3 是经验默认值。0.3 更偏向亲和，在容量受限时更危险 |
| 9 | 有界负载论文：平衡参数 $c=1+\varepsilon$ 可保证无副本超过 $\lceil cm/n\rceil$(arXiv:1608.01350) | 本仓未测；但同代码库的 `prefix_hash` 策略默认 `load_factor = 1.25`(config/types.rs：329-331) | **理论参数以默认值形态出现在实现里**。本仓没有测这条策略，所以只能指出它在设计上补齐了 cache_aware 缺的两样，不能主张它在 P06 格上会赢 |
| 10 | Bansal & Harchol-Balter：SRPT 对大作业的不公平"surprisingly small" | 8B boundary 档 lpm p99 +64% | 四处口径不同：对照物（PS vs FCFS）、策略（SRPT vs lpm 近似）、系统（开环 M/G/1 vs 闭环饱和）、统计量（稳态分位 vs 192 样本的次序统计量）。**不是文献错了，是四条定语都不一样**(§2.3) |
| 11 | Kleinrock 守恒律：一大类规则下 $\sum\rho_p W_p$ 不变 | 8B boundary 档 lpm 的 makespan 从 26.86 降到 20.41 s | **lpm 在守恒律的适用类之外**（它用服务时间相关信息，并且改变作业大小）。makespan 下降就是"类外"效应的度量（§2.2、§5.2） |
| 12 | SGLang v0.4 发布说明：cache-aware 负载均衡器带来"up to 1.9x throughput increase and 3.8x hit rate improvement" | 本仓 cache_aware 臂在容量受限格里 hit **崩到 0.0020** | **不可比**：发布说明的场景是容量充裕的多副本 serving，本仓是刻意压小池的机理格。两个数字回答的不是同一个问题——一个问"顺风时能多快"，一个问"逆风时会不会翻车" |

**这张表的读法**：十二条里只有第 2 条是干净的"文档—实现"一致；第 1、3、10、11 是"适用域不同"；第 5、6 是**实现内部或实现与观测之间的不一致**。**面试里最值钱的不是背下论文结论，而是能指出一条结论的适用前提在你的场景里破了哪一条。**

### 8.2 与生产实现的差距各在哪一层

- **上游 SGLang 调度**：除 lpm/fcfs 外还有 dfs-weight（按树的 DFS 权重）、 lof、random、routing-key（schedule_policy.py：200-215 两个枚举）。本仓只测了 lpm/fcfs 一对；dfs-weight 对深树负载的行为是自然的下一格——而且它更贴近 Theorem 3.1 的 DFS 表述，理论上应当比 lpm 更接近定理的最优构造。
- **router 的策略货架**：同一目录下还有 `prefix_hash`（prefix token 哈希 + 一致性哈希环 + 负载上界）、`power_of_two`（两选一）、`consistent_hashing`（会话亲和）、`bucket`、`round_robin`、`random`。**§2.5 的三个经典理论框架在这个目录里各有一个实现**，本仓只测了 cache_aware 与 round_robin 两端。把这六个策略在同一个容量受限格上跑一遍，是本条线最直接的下一步。
- **router 新老两代**：被测的 sgl-model-gateway（字符近似树，HTTP 侧）与 experimental/sgl-router 的 `cache_aware_zmq`（token block-hash + worker ZMQ KV 事件上报）是两代设计——后者的树与 engine 真值同源，理论上消除了 §3.3.1 的第 1、2 层近似（字符级、无逐出回流），但引入事件流的时延与可靠性问题；本仓未测。
- **vLLM 侧**：vLLM 的调度以 continuous batching + 优先级/FCFS 为主线， 前缀感知调度不是默认路径；多副本亲和通常由外部网关（如各家 gateway 的 session/prefix 亲和）承担——"engine 内调度"与"gateway 路由"的职责切分与 SGLang 同构，对比面试可讲两家 gateway 对"树"的不同近似。
- **与排队论文献的接口**：lpm ≈ SPT 的近似这一映射，把 P08 的"分位数再分配" 接到调度理论的标准结论上（SPT 最小化平均等待、牺牲尾部）；但必须同时讲清 lpm 多出来的那一半——它会改变总工作量，而经典调度理论里作业大小是外生的（§2.2）。**这一半才是 cache-aware 调度区别于普通队列优化的地方，也是面试里最能讲出深度的一点。**
- **本仓刻意没做的三件事**：①两层 cache-aware 叠加（router + engine 同时开）； ②并发到达下的路由行为（失衡回退会被激活，整除结构会被打破）；③双副本性能矩阵。三件加起来就是"从机理格升级为可用容量规划"的完整路径。

### 8.3 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)

**论文**

1. Zheng， Yin， Xie， Sun， Huang， Yu， Cao， Kozyrakis， Stoica， Gonzalez， Barrett， Sheng， "SGLang： Efficient Execution of Structured Language Model Programs"， arXiv:2312.07104，§3 "Cache-aware scheduling"、Theorem 3.1、附录 A.2、A.3、A.4。——想知道 lpm 的原始定义、它凭什么"最优"、以及那个最优性有哪三个前提，读这四处；§A.4 还给出了多副本的原始设计，与被测 gateway 对读即见代差。
2. Srivatsa， He， Abhyankar， Li， Zhang， "Preble： Efficient Distributed Prompt Scheduling for LLM Serving"， arXiv:2407.00023，§1、§3.2（E2 算法与三项代价 $L_i + M_i + P_i$）。——想知道"一个把缓存与负载放进同一个代价函数里的路由长什么样"，以及本仓 §3.4.2 说的那个缺失项 $M_i$ 具体怎么算，读这两节。
3. Kleinrock， "A conservation law for a wide class of queueing disciplines"， Naval Research Logistics Quarterly 12(2)：181-192, 1965。——想把"重排是零和" 这句直觉做严格，以及知道**这条守恒律的适用类要求调度不使用服务时间信息**（从而 lpm 出了这个类），读这一篇。
4. Schrage， "A Proof of the Optimality of the Shortest Remaining Processing Time Discipline"， Operations Research 16(3)：687-690, 1968。——想知道"最短剩余优先在什么意义上最优"（最小化任意时刻的在系统作业数），读这一篇；它是 §2.3 说 lpm 是"SPT 近似"时对标的那个精确结论。
5. Bansal & Harchol-Balter， "Analysis of SRPT scheduling： investigating unfairness"， ACM SIGMETRICS PER 29(1)：279-290, 2001。——想知道"SPT 类调度对大作业到底有多不公平"，读这一篇；它的结论（不公平"surprisingly small"） 与本仓 p99 +64% 表面冲突，四处口径差异见 §2.3，是练习"读文献先对齐口径"的好材料。
6. Azar， Broder， Karlin， Upfal， "Balanced Allocations"， SIAM Journal on Computing 29(1)：180-200, 1999。——想知道"为什么只多看一个桶就能把最大负载从 $\ln n/\ln\ln n$ 降到 $\ln\ln n/\ln 2$"，读这一篇；它是 `power_of_two.rs` 那个策略的理论出处。
7. Mirrokni， Thorup， Zadimoghaddam， "Consistent Hashing with Bounded Loads"， arXiv:1608.01350。——想知道"怎么同时要亲和与均衡"的标准答案，读这一篇； 它的平衡参数 $c=1+\varepsilon$ 就是 `prefix_hash` 策略里那个默认 `load_factor = 1.25`(§3.4.4)。
8. Little， "A Proof for the Queuing Formula： L = λW"， Operations Research 9(3)：383-387, 1961；以及 Lazowska， Zahorjan， Graham， Sevcik， *Quantitative System Performance*， Prentice-Hall， 1984，"Fundamental Laws" 一章（$R = N/X - Z$）。——想把"makespan 就是吞吐的倒数""闭环并发不会让时延发散" 这两句话做严格，读这两处；§5.2 的 makespan 读法全部建立在它们上面。
9. Qin et al.， "Mooncake： A KVCache-centric Disaggregated Architecture for LLM Serving"， arXiv:2407.00079。——想看一个生产系统怎么把"KVCache 为中心"的调度做到集群级（含过载时的预测式早拒），读这一篇；它是本仓这条线在规模上的延伸方向。

**官方文档与源码**

10. python/sglang/srt/managers/schedule_policy.py:237-300(calc_priority 全流程
    + 退化开关)与：314-384（in-batch 去重与排序）——本讲义 §4 前五段的原文， 以及 128 / 32 / 32 三个常数的字面出处。
11. sgl-model-gateway/src/policies/cache_aware.rs：1-61（文件头设计注释）与：385-480（select_worker 全流程）——把注释与实现并排读一遍，§3.3.2 那处分歧会自己跳出来。
12. sgl-model-gateway/src/policies/tree.rs：531-607(prefix_match_with_counts)——字符级近似树的匹配、tenant 选取与 1/8 概率时间戳全在这一段，§3.3.1 的四层近似逐条可对。
13. sgl-model-gateway/src/policies/prefix_hash.rs：1-31（算法五步与对比表） 与 src/config/types.rs：308-331（默认值）——想看"亲和 + 容量上界"在同一个代码库里是怎么写的，读这两处；它是 §3.4.4 那条"理论上正确的修法"的实体。
14. sgl-model-gateway/src/policies/power_of_two.rs 与 consistent_hashing.rs——§2.5 表格里另外两个理论框架的实现；三个文件加起来就是负载均衡经典结论的一次代码化巡礼。
15. experimental/sgl-router/src/policies/cache_aware_zmq.rs——新一代 token 级路由树，与被测实现对读，可直接看到 §3.3.1 前两层近似是怎么被消掉的。

**本仓证据**

16. docs/theory/02_router_cache_aware.md——router 机制笔记（观测端点、 smg_* 指标、无 cache-hit-rate 指标的坑）。
17. records/EXP-P06_routing_pool_capacity.md §7——首轮作废的完整事故记录（input_ids 丢弃、setproctitle、日志级排查），工程防线的第一手材料； §8 留有交叉复核位。
18. records/EXP-P04_lpm_vs_fcfs.md §6-§7 与 records/EXP-P08_8b_scheduling_tradeoff.md §6-§7——两阶段认知的原始记录，含候选机理与开放问题的如实标注； 配 data/derived/ 的两份 csv 可逐列复算，包括本篇 §5.2 用到的 `dur_s_mean`。
