# LAB JOURNAL · sglang-prefix-lab

每段追加:做了什么/为什么/关键数字/产物路径/下一步。时间正序。

---

## 2026-08-24 · 接手、避撞与建仓

- **做了什么**:接手用户"搭 SGLang 学习+简历项目"的任务。现场审计发现:①附件
  `sglang 内容.txt` 从未到达任何 agent(Codex 会话 15:48 同样记录缺失);②sibling
  agent(Codex)已在 `/root/projects/sglang-inference-lab` 建仓并推进(bootstrap
  15:57,EXP-S01 提交 16:21),期间我在其未提交工作树里误写文件并与其 GPU worker
  相撞(我方 smoke 在 16:12 CUDA graph OOM——同刻其 worker 占 GPU0 ~15.7GB,
  这是 avail=9.17GB 之谜的答案);③venv `/root/venvs/sglang-lab` 由其 uv 安装,
  16:09 完成,栈健康(torch 2.13.0+cu13, CUDA 可用双卡)。
- **处置**:把我写入 sibling 仓的文件全部退出其工作树(其 16:21 提交已扫入我的
  4 个文件,不改写其历史,工作树恢复到 HEAD);另建本仓,端口错开
  (28000/28001/40000/29000),preflight 加"外来 GPU 进程即中止"硬 gate。
- **选题决策**:sibling 做 router 策略矩阵 → 本仓做其未覆盖的 **engine 侧
  RadixAttention 机理**(单卡为主,避抢卡),双副本仅留交叉复核位(P06)。
- **产物**:仓骨架 + docs/theory/01-03(radix 机制/路由机制/负载契约,全部
  file:line 锚)+ docs/tooling/bench_serving 笔记 + PLAN(P01-P06 预注册)+
  scripts(svc/preflight/provenance)+ records/data freeze。
- **下一步**:commit → preflight → EXP-P01 smoke(GPU0, 28000)。

## 2026-08-24 · EXP-P01/P02:radix 首证与契约矩阵(含一处证伪)

- **做了什么**:P01 单 worker smoke(确定性+cached=n−1 首证);P02 五格契约矩阵
  (messages/input_ids/thinking/salt),thinking_flip 格证伪预注册假设,CPU 渲染
  对比钉死机制(纯尾部扩展),theory/01/03 当场修订。
- **关键数字**:cached=1324/1325(=n−1);hit_rate 0.9992;thinking_flip 命中
  1326/1329(+1 来自树缓存 input+output 全序列咬进上一请求首输出 token)。
- **排障**:svc.sh stop 身份校验误判(uv venv python 符号链接→readlink -f 归一);
  preflight /proc 竞态(段内 set +e);transformers 5.x BatchEncoding。
- **产物**:records/EXP-P01、EXP-P02 + data/raw 配对;theory 修订。
- **下一步**:P03 收益曲线(radix on/off × 并发 1/8 × 3 seed)。

## 2026-08-24 · EXP-P03:收益曲线 + token 级归因闭环

- **做了什么**:TTFT vs 共享前缀(0..1792/2048)双臂(radix on/off)×并发(1/8)
  ×3 seed;聚合出 derived 表。
- **关键数字**:c1 −36%(26.84→17.27ms)、c8 −63%(115.14→42.73ms);OFF 臂平;
  engine device_hit 466,944 = Σcached 逐 token 闭环;c1 斜率 5.3µs/token,
  c8 40µs/token(排队放大)。
- **坑**:`cache_hit_rate` 是窗口 gauge,空闲归零,累计口径要用
  prefill_effective_tokens_total counter。
- **产物**:records/EXP-P03 + raw 9 jsonl + derived csv。
- **下一步**:P04 fcfs vs lpm(含 >128 等待队列的 lpm 退化边界档)。

## 2026-08-24 · EXP-P04:调度收益窗口的两面

- **做了什么**:fcfs vs lpm × {std 64req@c16, boundary 192req@c64} × 3 seed。
- **关键数字**:std 档差异与波动同量级(如实判平);boundary 档 lpm p99 747±20 vs
  fcfs 661±36(+13%,>2σ),hit 0.968 vs 0.992——lpm 超出设计窗口反成负资产。
- **方法论**:fcfs 命中方差(0.027)≫ lpm(0.006),方差差异本身即"排序削平到达
  序随机性"的机制证据。
- **产物**:records/EXP-P04 + raw 12 json + derived csv。
- **下一步**:P05 逐出压力(小池 vs 默认池)。
