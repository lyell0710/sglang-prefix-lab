---
topic: SGLang 双副本路由 —— round_robin vs cache_aware 的机制与回退
status: 源码级完成(file:line 锚 sgl-model-gateway = PyPI sglang-router 0.3.2 的实现)
verified_against: 本仓 EXP-S04(策略矩阵)——待跑
---

# 02 · cache-aware 路由:把同前缀送到持有 KV 的那张卡,以及它何时"故意不这么做"

## 0. 先厘清:v0.5.18 里有两个 router

- **sgl-model-gateway**(`sgl-model-gateway/`,PyPI `sglang-router==0.3.2` 装的就是它):
  经典 KV-aware,近似树用**原始字符**,策略含 `cache_aware`。**本项目用这个**。
- experimental/sgl-router(新 slim 版):策略叫 `cache_aware_zmq`,树用 **block-hash of
  token ids**,靠 worker 的 ZMQ KV 事件喂,无 Python 绑定。本项目不用,但面试可提"新老两代"。

下面全部针对 **sgl-model-gateway**(Python 入口 `python -m sglang_router.launch_router`)。

## 1. 一句话结论

`cache_aware` 每来一个请求,先看**负载是否已失衡**:失衡就退化成"最短队列"(丢掉
亲和性保平衡);不失衡才查近似前缀树,匹配率 > 阈值就送到**持有该前缀的 worker**,
否则送最小负载。**收益来自把热前缀钉在同一张卡上放大 engine 侧 radix 命中;回退
机制是它在高并发下不至于把所有请求堆到一张卡的安全阀**——这个"回退点"正是 S05 要
定位的边界。

## 2. 决策流程(cache_aware.rs:387-526,伪代码带行号)

```
healthy = 健康 worker                                   # :393
(min_load, max_load) = 各 worker 当前负载               # :405-409
is_imbalanced = (max-min) > balance_abs_threshold
             && max > min * balance_rel_threshold       # :412-413
if is_imbalanced -> 选最小负载(随机 tie-break),仍插树  # :415-424
tree = trees[pool::model]                               # :431
match_rate = 匹配字符数 / 输入字符数                     # :436-441
if match_rate > cache_threshold -> 送 tree 记录的那个 worker(若健康)  # :444-450
else                            -> 送最小负载(随机 tie-break)          # :456-467
插树(text -> chosen.url);chosen 处理数 +1              # :471,:487
```

**关键参数与默认**(Python 侧 router_args.py):
| 参数 | flag | 默认 |
|---|---|---|
| 匹配阈值 | `--cache-threshold` | 0.3 |
| 失衡绝对阈值 | `--balance-abs-threshold` | 64 |
| 失衡相对阈值 | `--balance-rel-threshold` | 1.5 |
| 逐出周期 | `--eviction-interval-secs` | 60 |
| 树上限 | `--max-tree-size` | 2^26 |

读法:`is_imbalanced` 要**同时**满足绝对差 >64 **且** 最大 > 1.5×最小,才触发平衡回退。
→ S05 的 sweep 就是把并发/热前缀偏斜推到跨过这条线,看 TTFT 收益如何翻转。

## 3. 近似树用字符不用 token(cache_aware.rs:1-61, tree.rs)

- 注释自述:"存原始字符而非 token id 以省 tokenize 开销"(:22-23)。HTTP 路径
  `request_text: text, tokens: None`(router.rs:171-179)。
- 含义:router 的匹配率是**字符级近似**,engine 的真实命中是**token 级**;两者通常
  同向但不严格相等。实验里 router 的 `overlap`/selection 指标只作"路由决策证据",
  真实收益仍以 engine 侧 `cache_hit_rate` / TTFT 为准(别把 router 近似当命中率报)。

## 4. 观测(写实验必须知道的端点与指标)

- 端点(server.rs):`/health`、`/flush_cache`(fan-out 所有 worker)、
  `/get_loads` + `/v1/loads`、`/workers`(GET 列出/POST 加),Prometheus 独立端口
  默认 **29000**(`--prometheus-port`)。
- 指标(`smg_*`):`smg_worker_selection_total{worker,policy,...}`(每 worker 被选次数——
  **直接量化路由分布**)、`smg_worker_requests_active{worker}`、
  `smg_worker_routing_keys_active`、`smg_router_ttft_seconds`。
  ⚠ **sgl-model-gateway 没有 cache-hit-rate 指标**——命中率只能从 engine 侧读。
- 每请求哪张卡服务:sgl-model-gateway 的路由理由是 **DEBUG 级**日志(cache_aware.rs:496,
  :532-536),无响应头直接标注 worker。→ 要逐请求归卡,靠 `smg_worker_selection_total`
  的增量 + 两 worker 各自 `/metrics` 的 per-worker 计数交叉验证(与 vllm/experiments
  EXP-013 的"逐请求双端匹配"同法)。

## 5. 启动(两 worker 已跑时,本项目 40000 端口)

```bash
python -m sglang_router.launch_router \
  --worker-urls http://127.0.0.1:28000 http://127.0.0.1:28001 \
  --policy cache_aware --host 127.0.0.1 --port 40000 \
  --cache-threshold 0.3 --balance-abs-threshold 64 --balance-rel-threshold 1.5 \
  --prometheus-port 29000 --prometheus-host 127.0.0.1
```
换 `--policy round_robin` 得对照臂(其余参数不变)。**同一 manifest、同 seed、
两臂 worker 冷启动等价**(见 protocol)。

## 6. 面试追问 Q&A

- **Q:cache-aware 一定更快吗?** 不。低并发+强共享前缀时它把热前缀钉住放大命中→TTFT 降;
  但热前缀极度偏斜+高并发时,亲和性把请求全堆到一张卡,另一张闲置,整体 p99 反而更差——
  这时平衡阈值触发回退。**"更快"是有区间的,S04/S05 就是把这个区间量出来**,并保留
  unique-prefix 反例证明收益确来自前缀复用而非别的。
- **Q:router 树和 engine 树什么关系?** 两棵独立的树:router(字符,决定去哪张卡)、
  engine(token,决定省不省算)。router 命中 ≠ engine 命中,但 router 命中是 engine
  命中的**必要前置**(送错卡则那张卡没有这段 KV)。
- **Q:为什么要负载失衡回退?** 纯亲和性在偏斜负载下退化成单卡串行;回退用"最短队列"
  换回并行度。这是 cache-locality 与 load-balance 的经典取舍。

## 7. 延伸(源码锚)

sgl-model-gateway/src/policies/cache_aware.rs、tree.rs、src/server.rs、
src/observability/metrics.rs;新一代对照 experimental/sgl-router/src/policies/cache_aware_zmq.rs。
