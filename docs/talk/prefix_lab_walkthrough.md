# 面试讲稿 · SGLang 前缀缓存实验室(P01-P08 成稿)

> 讲法:每段"机制一句话 → 我测到什么 → 反直觉点/反例"。所有数字指 records/EXP-Pxx
> 与 data/derived/,不脱稿报数。

## 0. 三十秒版本

「我在 2×4090 上把 SGLang v0.5.18 的 RadixAttention 从源码机制做到可复现测量:
证明了命中判定是 token 级的并踩实了 chat template 的坑;测出共享前缀 1792/2048 时
TTFT p50 降 36%(并发 1)到 63%(并发 8),且用引擎计数器做到**逐 token 归因闭环**;
再往边界推:lpm 调度在积压超过其 128 窗口时 p99 反劣 13%,LRU 逐出在池小于
"热前缀重用距离"时命中从 1.0 阶跃崩到 0.06——不是衰减,是悬崖。每个实验带
预注册假设、反例臂和 3 轮 raw;有一处假设被证伪并当场修正了理论笔记。」

## 1. 机制层(P01/P02):缓存的是 token 序列,不是文本

- radix 树 value=KV 块索引,匹配 page 对齐,**至少重算 1 token** → 实测 cached
  精确 = prompt−1(1324/1325),不是"约等于"。
- **被证伪的假设**(诚实亮点,主动讲):我预注册"Qwen3 thinking 开关改 system 段
  → 全 miss";实测 B 端命中 1326/1329。CPU 渲染对比钉死:开关是**generation prompt
  尾部的纯扩展**(`<think>\n\n</think>`),前缀共享完好。多出的 +1 还顺带证明
  radix 缓存的是 **input+output 全序列**(咬进上一请求的首个输出 token)。
- salt 命名空间隔离实测全 miss;input_ids 直传绕模板 = 实验的契约形态。
- 追问预案:"cached_tokens 哪来的" → usage_processor 只在 --enable-cache-report
  时带 details,0 时字段缺失(不是 0,是 absent——脚本要 `or 0`)。

## 2. 收益层(P03):命中值多少毫秒,以及为什么并发放大

- 曲线:c1 26.84→17.27ms(−36%);c8 115.14→42.73ms(−63%);OFF 臂
  (--disable-radix-cache)全线打平 = 因果反例。
- **归因闭环(最硬的一张牌)**:engine `prefill_effective_tokens_total{device_hit}
  = 466,944`,与客户端全部请求 cached 之和**逐 token 相等**(3 seed×2 并发×
  Σ16×prefix_len)。TTFT 降幅确实且仅由省掉的 prefill 兑现。
- 反直觉点:c1 收益只有 −36%,因为 0.6B 有 ~17ms 与 prefill 无关的地板;c8 下
  prefill 是队列瓶颈,省的算力同时缩短排队 → 斜率从 5.3µs/token 放大到
  40µs/token(×7.6)。**"低并发测不出缓存价值"是这套数据最好的一句话**。
- 坑:`sglang:cache_hit_rate` 是窗口 gauge,空闲归零;累计口径用 counter。

## 3. 边界层(P04/P05):两个"不是免费午餐"

- **调度(0.6B→8B 是一次完整的认知升级,面试重点讲)**:0.6B 轻载判平、积压档
  lpm 单纯反劣(p99 +13%);换 8B 后同一积压档 **lpm p50 −62%、命中 +17.7pp、
  p99 +64%**(EXP-P08《8B 调度》)——排序聚簇让同组请求稳定吃到前者的 KV(fcfs 并发交错时
  同组 KV 未入树就 miss),代价是靠后的组整组饿死。**调度收益不是标量,是延迟在
  分位数间的再分配;答"lpm 好不好"先反问 SLO 定义在 p50 还是 p99。**
- **逐出**:LRU 命中 ⇔ 池 ≥ 重用距离 D=8192×(1+cold_ratio)。三池验证
  (8192/16384/大池)无一例外;悬崖两侧 std=0。轮转+LRU=循环工作集病态,与
  CPU cache thrashing 同构。**容量规划按重用距离配池,不是按热集大小**。
- **P06(用模型做预测,然后被打脸两次——最值得讲的一段)**:预测"rr 全崩、
  cache-aware 获救",实测**正好相反**:rr@6 热前缀全命中(轮转周期与 2 卡整除,
  巧合分片,hot5 奇数对照立即崩塌);cache-aware 把 100% 流量钉一张卡
  (计数器 61799/0)塞爆单池。修正结论:**亲和≠扩容,映射分散度才是本质**;
  顺带抓到 router 丢 input_ids 扩展字段、setproctitle 改名两个工程陷阱。

## 4. 方法论(所有项目共用的那套)

预注册假设+判定阈值 → 反例臂 → ≥3 seed → raw+provenance → 计数器交叉归因 →
被证伪当场改理论笔记(EXP-P02《token 契约矩阵》)。工程细节:uv venv 符号链接的进程身份校验、
/proc 竞态下的 preflight、与并行 agent 的 GPU 相撞与制度化避撞(preflight 硬 gate)。

## 5. 红线(不说的话)

不说"生产级/集群";不说 8B 结论(只测 0.6B);路由只讲 P06 的容量机理格,不讲性能矩阵(sibling 仓范围);
lpm 边界档机理二选一未定,只报实测差。
