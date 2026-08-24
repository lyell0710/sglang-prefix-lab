# EXP-S00 · bootstrap 现场审计

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 协议 | 建仓前现场审计；不产生性能数字 |
| 环境 | host + upstream `main@76d1401881f2`；SGLang venv 尚未建立 |
| 状态 | 完成；incident 根因部分为终端级证据 |
| 关联清单项 | `docs/PLAN.md#exp-s00--bootstrap-audit` |

## 1. 目的、假设与预锁阈值

在安装或启动服务前判断截图中的 60 秒 subprocess 超时是否由 GPU kernel、服务端口、普通外网、半成品环境或
host I/O 阻塞导致。允许输出“最可能归因”，不把相关性写成已证明因果。

## 2. 环境与配置

- host：2×RTX 4090 24 GB，CUDA toolkit 13.2，driver 610.57.04。
- 已有 upstream：`/root/repos/sglang`，clean `main@76d1401881f2593b0f781146b87ce27583b6209a`。
- 现有 venv 均未安装 SGLang；`/root/venvs/main` 属于 vLLM，不允许复用。
- 端口检查范围：18000、18001、30000。

## 3. 可复现步骤

```bash
cd /root/projects/sglang-inference-lab
bash scripts/preflight.sh data/raw/EXP-S00/20260824T155701_host_preflight.txt
```

现场诊断阶段另执行了 `ps`、`vmstat` 和 HTTPS HEAD 探测。确认一个 `rg` 是此前只读审计遗留后，只向其子进程
PID 579648 和父 shell PID 579576 发送 TERM；未停止 Cursor/Claude 认证进程和任何用户服务。

## 4. 原始数据

- `data/raw/EXP-S00/20260824T155701_host_preflight.txt`：清理遗留只读扫描后的完整 host 快照，首行 provenance。
- 事故发生时的 D-state 数量、`vmstat` 和 HTTPS 返回只存在于本会话终端，标记为**终端级证据**，不作性能主张。

## 5. 直接测量结果

raw 快照显示：两张卡各 1 MiB、0% utilization、P8，无 compute process；三个项目端口全空闲；无 SGLang
进程；剩余磁盘 163 GB；upstream clean。

事故诊断时的终端级证据：一个从 `/root` 发起的全盘 `rg` 有 12 个线程与两条 `claude auth status --json`
卡在 `folio_wait_bit_common`，I/O wait 一度约 32–37%；GitHub/Hugging Face HTTPS 均返回 200。遗留 `rg`
停止后，15:57 的 raw 中已无 D-state task。

## 6. 分析与结论

- **实测**：没有 CUDA kernel/GPU 进程，没有端口冲突，也没有已安装一半的 SGLang venv。
- **推断**：截图的通用“authentication and network connectivity”提示并不符合现场主要症状；60 秒超时更可能由
  overlay/filesystem I/O 阻塞放大，而不是 GPU 占用或普通 HTTPS 断网。
- 结论边界：没有 syscall trace，因此只能称“最可能归因”，不能称已证明根因。

## 7. 异常、偏差与开放问题

- 事故高峰没有按项目格式即时存 raw，只能降级为终端级证据。
- 若超时再次发生，第一动作应在**不全盘扫描**的前提下同步捕获 `ps -eLo`、`vmstat`、目标 subprocess cwd/cmd
  和精确网络探测，再决定是否停止进程。

## 8. 下游影响

- 解锁创建独立 venv 和单 worker smoke。
- 服务生命周期脚本禁止 `killall`/模糊 `pkill`，只管理项目 PID 文件。
- 此 EXP 不解锁任何简历性能措辞。

