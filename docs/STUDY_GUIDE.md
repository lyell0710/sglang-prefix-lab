# 学习路线与面试检查表

本项目每阶段都要能回答“机制是什么、怎么知道、反例是什么”。读源码和跑实验并行，禁止只背结论。

## S01：单 worker 数据流

- OpenAI 请求从 HTTP server 到 scheduler、model runner 的调用链在哪里？
- prefill 与 decode 的批处理单位、KV page/block 和 RadixAttention 各解决什么问题？
- 为什么服务启动成功不等于目标 CUDA backend 被命中？怎样从 log/trace 证明？

## S02：可控 workload

- “字符串看起来相同”为什么不够？chat template、BOS/role token 如何改变真实 token prefix？
- 如何保证两个 policy 看到字节级相同且 tokenized 长度已验证的 manifest？
- 冷 cache、warm cache、JIT warmup 分别怎样污染 TTFT？

## S03/S04：双副本路由

- round-robin、shortest-queue、cache-aware 分别优化什么目标？
- 近似 radix tree 如何估计 prefix overlap？cache-aware 为什么需要负载失衡回退？
- worker request 分布更不均衡时，为什么整体 p99 仍可能更好或更差？

## S05：归因

- TTFT 能拆成 client/router queue、engine queue、prefill compute 的哪些可观测部分？
- 为什么 profiler 数据不能直接拿来报 latency？
- unique-prefix 反例、共享前缀正例、负载反转点如何组成因果证据链？

## 每个数字的五问

1. raw 在哪里，首行 provenance 是什么？
2. baseline 是谁，除了目标变量还有什么不同？
3. correctness/codepath/thermal gates 是否全过？
4. 有几轮，mean/std 与尾延迟各是多少？
5. 哪个反例能推翻更宽泛的说法？

