## 中文对照与当前状态（2026-10-31）

> 本文保留下方英文原文作为阶段性运行记录。原文的“未发布本地补丁”“不训练”和早期关机约束描述的是最初先导阶段，不能当作整个后续批次的当前状态。代码、配置与匿名聚合结果现已发布；后续匹配低剂量 FBP 残差 U-Net 训练与评估也已完成。原始完整 `best.pt` / `last.pt` 已作为 [Release v1.1-p1-dev-20261031](https://github.com/ZeroDevi1/flow-ct-guidance/releases/tag/v1.1-p1-dev-20261031) 附件发布，不写入 git。机制诊断尚未完成。

### 本目录的用途

这是基于历史提交 `2311351b40adafd47cc922200a2c249cc22650e3` 准备的受预算约束 P0/P1 先导实验。新增实验使用独立目录，不覆盖历史采样器、原始指标或既有结果。最初先导不训练模型；后续训练与最终开发验证结果见 [本轮复现证据](../../../results/2026-10-31-budget-p1/README.md)。不要将先导范围、后续完整批次和历史 test200 混为同一评测。

### 先导实验范围

1. 合成 P0 控制在相同**绝对 λ**下比较稀疏 M3 与未测角度权重为零的全网格 DC，并保留使用相同相对 λ 时的历史差异。代数控制对齐已测角度和偏移，同时报告独立构造网格的原始差异。float32 与有限步 CG 在部分设置会放大微小的角度舍入误差，因此不能由此宣称未修改的 `dcfull` 自动等价于 M3。
2. 读取 `index_val.csv`，核对患者 ID 与训练/测试索引不重叠，不加载测试图像数组。按种子 20261031 固定随机排序后的四名验证患者，每人选两张均匀分布切片。
3. `sv60`、`la120` 比较 measured-only DC，λ 为 0.3/1/3；`ld180_i1e5` 比较普通 DC 与观测计数 WLS，λ 同为 0.3/1/3。初始先导使用 NFE20、CG5、一个配对种子，同一图像跨方法/λ 共用测量噪声与初始潜变量噪声。
4. PSNR、SSIM 采用相同圆形 ROI，并包含 FBP。记录吞吐量、CUDA 峰值分配显存、加权/未加权残差和计数下限占比。这个小规模验证先导不是最终质量结论，也不足以证明泛化。

WLS 精度权重与估计观测计数成正比，即对数噪声方差倒数的 delta 近似；再按**每幅图像均值归一化为 1**。当前仅使用 sigma=0，截断后的零/一个计数不可区分。这会丢失绝对精度尺度，未加权 `ata_scale` 也不一定等于加权算子的迹。应在开发验证上选择 λ，不能把相同 λ 解释为相同贝叶斯先验强度。本实现不宣称精确 Poisson 似然或已校准后验。

### 历史启动方式及安全限制

在已授权的 AutoDL GPU 实例上使用独立目录，通过获准的文件传输方式上传代码包，不能覆盖历史 `MedFusionProjects` 目录。启动封装不执行 pip 安装，预期环境已有 torch、numpy、yaml、skimage、Pillow。反序列化先验前，必须确认其 SHA-256 与已核实的 Release 一致。

英文原文中的 `run_pilot.sh` 命令属于**历史先导方式**：它会在退出时请求整台实例关机，并带命令超时和独立的 1800 秒监控。仅适用于明确授权的实例，不适用于共享服务器。原先要求启动前验证 AutoDL 控制台的关机截止时间不晚于先导启动后 30 分钟；容器内存在 `shutdown` 命令，不代表有真实断电权限。不能用容器内监控替代平台硬停止保障。

`PRIOR`、`PROCESSED`、`PILOT_OUT` 可按实际环境设置，输出目录不得预先存在。关机应在 AutoDL 控制台确认，SSH 断连或命令返回成功都不等于计费已停止。历史费率为 2.88 元/小时，半小时约 1.44 元，不含启动前准备；新运行必须核对当时费率、实际开机时间与账户费用，不能照搬旧日期或旧授权。

原先导不自动重试、不重复训练、不自动下载或上传结果、不创建 API key。原始患者级输出和预览图不属于本次公开材料；后续经审查公开的是代码、匿名聚合结果和已明确授权的原始完整训练断点，详见 [发布范围与署名](../../../results/2026-10-31-budget-p1/CHECKPOINT_PUBLICATION.md)。

### 恢复、失败与数值门控

验证脚本没有 resume 开关。残缺 JSONL/summary 证据会保留，后续获准批次必须使用新的输出目录，避免重复计数或覆盖失败。

预测、输入、FBP 或指标一旦出现非有限值，会立即写入失败状态并终止，不输出成功结论。固定队列不完整时保留原始行，但不计算质量汇总或排名。`comparison_status.json` 只在所有待比较方法/λ 具有完全相同且完整的患者/切片队列后允许比较。超时不是方法性能结果，完成的子集不能冒充完整扫描。

### 后续修订：修复过程中保持 GPU 开机

英文原文下半部分记录了授权变更：修复会话使用 `run_stage.sh`，**不再使用 `run_pilot.sh`**。旧封装只作为历史证据保留。新 runner 本身不关机；操作者须先核实 8 小时内的绝对平台截止时间和可用预算，再设置 `PLATFORM_DEADLINE_VERIFIED=YES` 与 `PLATFORM_DEADLINE_EPOCH=<UTC epoch>`。

`diagnostic`、`pilot` 阶段的历史进程超时分别为 120 秒、1200 秒；可用 `STAGE_TIMEOUT_SECONDS` 覆盖，最大 1500 秒。失败阶段保存 `launcher_status.json` 和日志。GPU 空闲仍计费，诊断输出不构成绕过或放宽 P0 门控的理由。

**当前复现入口还须阅读后续 `run_stage_v2.sh` 与批次报告。** 以上是不同阶段的历史协议，不自动授权任何新租机、费用或实验。先固定新的预算、平台停止保障和实验协议，再运行适用脚本。

---

## Original English record / 英文原文

> Publication note (2026-10-31): the user authorized publishing reviewed code and anonymous aggregates. Historical preparation notes below describe the original private run. Patient-level experiment outputs remain excluded. After clarification that checkpoint cohort IDs belong to the public dataset, the user additionally authorized original full checkpoints as Release assets; see the publication policy in the result directory.

# Budgeted P0/P1 pilot (2026-10-31)

This is an **unpublished local experiment patch** on repository commit
`2311351b40adafd47cc922200a2c249cc22650e3`. It does not modify the historical
sampler, source metrics, published results, or final test images. No training.

## Scope

1. Synthetic P0 controls compare sparse M3 against zero-unmeasured-weight full-grid
   DC using the same **absolute** lambda, then report the historical same-relative
   lambda discrepancy. Exact measured angles/offsets are aligned for the algebraic
   control; the raw independently constructed grid discrepancy is reported too.
   Float32 finite-CG amplifies tiny angle rounding in some cases, so this is NOT
   evidence the current unmodified `dcfull` automatically equals M3.
2. Read `index_val.csv`, verify patient IDs do not overlap train/test index files
   (no test array loading), freeze four randomly ordered validation patients and
   two evenly spaced slices each, selection seed 20261031.
3. `sv60`, `la120`: measured-only DC, lambda .3/1/3. `ld180_i1e5`:
   unweighted DC vs observed-count WLS, lambda .3/1/3. NFE20, CG5, one paired seed.
   Same measurement and latent noise for each image across methods/lambda.
4. PSNR and SSIM use the same circle ROI. FBP is included. Record throughput,
   peak allocated CUDA memory, unweighted and weighted residuals, and count-floor
   fraction. This small validation pilot is not a final quality result or adequate
   statistical evidence of generalization.

WLS precision is proportional to estimated observed counts (inverse log-noise
variance, delta approximation), normalized to mean one **per image**. It uses
sigma=0 only; clipped zero/one counts are indistinguishable. This discards absolute
precision, and unweighted ata_scale need not equal weighted trace. Tune lambda on
validation rather than interpreting equal lambda as equal Bayesian prior strength.
No claim of an exact Poisson likelihood or calibrated posterior.

## Launch on existing authorized AutoDL GPU instance

Use a separate directory, never overwrite the historical MedFusionProjects tree.
Upload this code bundle through a permitted file-transfer mechanism. No pip install
is performed by the wrapper; existing torch/numpy/yaml/skimage/Pillow are expected.
Prior file must match the verified release SHA256 before deserialization.

```sh
cd /root/autodl-tmp/flowct_budget_20261031/code
AUTODL_BUDGET_SHUTDOWN_APPROVED=YES \
  bash experiments/budget_pilot/run_pilot.sh
```

Set `PRIOR`, `PROCESSED` or `PILOT_OUT` if needed. Output must not already exist.
**Required before launch:** enable and verify an AutoDL console/control-plane
shutdown deadline no later than 30 minutes after pilot start. The existence of
`shutdown` inside a container does not prove permission or power-off capability.
Do not substitute the in-container watchdog for this external hard-stop condition.

The wrapper requests **whole-instance shutdown on exit**, has command timeouts,
and a separate 1800-second watchdog. It is only for this authorized instance;
not for a shared server. The operator must verify shutdown in the AutoDL console:
SSH loss or a successful shutdown command alone is not billing confirmation.
At 2.88 CNY/hour the 30-minute cap is about 1.44 CNY after launching, excluding
setup time before launch. Track actual boot time and whole-account charges.
Do not launch if the shutdown command or the control-plane fallback is unavailable.

No unconditional auto-retry, no repeat training, no auto-download, no uploading
results, no new API keys. Results include selected patient IDs and private preview
images; keep private pending user approval and privacy review.

The validation script has no resume switch: partial JSONL/summary evidence is
preserved and further approved batches must use new output directories. This
avoids double-counting or silently overwriting failures.

Numerical safeguards: non-finite predictions, inputs, FBP or metrics immediately
write a failed status and abort; no successful conclusion is emitted. Incomplete
fixed-cohort groups retain raw rows but have no quality aggregate or ranking.
`comparison_status.json` permits a setting comparison only after every requested
method/lambda has the identical complete patient/slice cohort. A timeout is not a
method result, and completed subsets must not be mistaken for the full sweep.

## Revised operating authorization: keep the GPU instance on while fixing

Use `run_stage.sh`, **not `run_pilot.sh`**, for the revised session. The old
wrapper remains historical evidence and powers off on exit. The new runner
never powers off; the operator must first verify an absolute platform shutdown
deadline within 8 hours and the available monetary budget, then supply
`PLATFORM_DEADLINE_VERIFIED=YES PLATFORM_DEADLINE_EPOCH=<UTC epoch>`.
Stages `diagnostic` and `pilot` have process time limits (120 and 1200 seconds
respectively, override `STAGE_TIMEOUT_SECONDS`, maximum 1500); failed stages
write `launcher_status.json` and retain logs. GPU idle time still costs money.
Diagnostic output is not permission to bypass or relax P0; review first.
