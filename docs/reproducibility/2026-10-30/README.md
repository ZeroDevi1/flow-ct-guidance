# Phase 1 历史实验资料恢复与来源核验

**恢复日期：2026-10-30。仅归档既有资料，没有运行新训练、重建或数值复现实验。**

本目录从原 AutoDL 实例的既有 `MedFusionProjects/01_diffusion_inverse`、相关 checkpoint 目录恢复文本证据，并与仓库固定提交 [`a9c29b4`](https://github.com/ZeroDevi1/flow-ct-guidance/tree/a9c29b4e6f2db2745034151f3d7a94dae699faa6) 比较。它补充复现所需的来源链，不替代论文主表，也不是完整数据盘备份。

## 本次实际核实了什么

- 下载的原始采集包为 **229,590 字节**，SHA256 为 `99b28bed28355de7722af207a51643c8e3d359d43fd4b1dc4ffe010f934ad673`。包内原始文本的大小、SHA256 均重新核对。
- 原项目目录未发现 `.git`，因此**无法提供原运行目录的历史 commit**，也不能证明恢复到的每份源码就是每个历史结果运行瞬间的版本。文件修改时间仅为辅助线索，不是防篡改时间证明。
- 258 份原始源码、配置及 JSON/Markdown 记录中：**228 份**与上述固定提交的对应文件逐字节相同，**3 份**不同，**23 份**在对应路径不存在，另 **4 份**属于旧 benchmark 目录、没有对应映射。比较采用 Git blob SHA-1，另保留采集文件 SHA256。
- 228 份相同文件不重复复制；[MANIFEST.json](MANIFEST.json) 给出固定提交链接、blob SHA、采集 SHA256、大小和时间。3 份不同源码以及 27 份额外历史记录保留在 [historical/](historical/)，**未覆盖现有 `code/` 或 `results/`**。
- 原盘上的先验 EMA 及四个 FBPConvNet 权重，均与 [Release v1.0-phase1](https://github.com/ZeroDevi1/flow-ct-guidance/releases/tag/v1.0-phase1) 的文件大小和 SHA256 一致。此次没有重新上传这些已公开的二进制，也没有反序列化 checkpoint。

## 三份不同源码：区别不应被隐藏

| 历史文件（相对原项目） | 当前仓库与恢复版本的差异 |
|---|---|
| `src/sample_posterior.py` | 当前版本加入 Phase 1b 的 `dcfull`、补全器、置信度及 HLCC 接口；恢复版本仅包含原 Phase 1 路径。 |
| `src/physics/radon.py` | 当前版本增加加权全角度 CG、已测掩码和缺失角度插值函数。 |
| `scripts/analyze.py` | 当前版本增加样本均值的配对比较，并修订圆域显示/误差、窗宽窗位、展示切片与图形刻度。 |

这说明不能把“源码全部与 main 相同”作为历史来源声明。需要研究旧执行行为时，可在上述固定提交的独立副本中，按 manifest 指向的历史文件替换这三份；这只是构造恢复快照的方法，**不证明所有历史运行均使用该快照，也未经本轮数值复现验证**。不要用它覆盖日常开发目录。

## 历史 smoke / benchmark / stale 的使用边界

补充文件包括 `smoke*`、`bench_prior256*` 以及 `main_sv30_dps_stale1` / `main_sv60_dps_stale1` 的配置、日志或指标。这些资料用于保留开发和试运行轨迹：

- `stale1` 按原目录名保留；其具体被替换原因没有在本次采集资料中完整记录，**不得擅自解释成算法改进证据**。
- 不把这些记录混入主表，不重新选择最优结果，不据其覆盖既有发布指标。
- 有些历史记录仅有汇总指标而无逐图 CSV，不能据此补造患者级置信区间。
- `01_diffusion_inverse_old_1792370884` 的 benchmark 只保留其原始目录归属，不等同最终模型的训练记录。

## 原始训练与评估控制台日志

另恢复了 13 份命名日志，见 [training_logs/](training_logs/) 与 [TRAINING_LOG_MANIFEST.json](TRAINING_LOG_MANIFEST.json)：包括 prior256、FBP、matrix / matrix2 / matrix3、smoke、precheck、benchmark 和 ZIP 完整性检查。原始日志按字节保留，包括警告、失败和重试，不清洗成“全部成功”的叙事。第二采集包为 83,063 字节，SHA256 `465bdd0bece4063e0fe5464272c04e1eac98cf50fd51df7644c3d5cefe8fff42`。

日志中的历史测试成功或耗时只说明当时记录的事件；本轮没有重新执行。`precheck*`、中断或重复矩阵不能当作额外独立实验，命令出现过也不证明该命令最终完成。排除了通用 exec 管理日志、下载认证日志及 setup 日志；本次公布的命名日志经凭据/个人信息筛查，不包含医学图像或逐患者映射。

## 环境：恢复时快照，不是历史锁定文件

[CURRENT_ENVIRONMENT.json](CURRENT_ENVIRONMENT.json) 记录 **2026-10-30 13:48 UTC** 恢复时的 Python 与已安装包名/版本，仅包含版本信息，不包含凭据、环境变量或私有包源 URL。

其中可见 Python 3.12.3、PyTorch `2.12.1+cu130`、torchvision `0.27.1+cu130`。这**不独立证明 10 月 19 日训练时的每个包版本相同**，也不是在其他机器上验证过的 lockfile。首次复现仍应按仓库安装说明检查依赖与 GPU 兼容性；本轮未运行 GPU 验证。原环境脚本仍含旧 `MedFusionProjects/shared/env` 布局，应按实际目录审查，不应原样盲执行。

## 数据校验，不分发医学影像

[PROCESSED_ARRAY_MANIFEST.json](PROCESSED_ARRAY_MANIFEST.json) 保存原盘四个数组的文件名、shape、dtype、字节数及 SHA256：

| 文件 | shape | dtype |
|---|---|---|
| `slices_train.npy` | 43220 × 256 × 256 | float16 |
| `slices_val.npy` | 2280 × 256 × 256 | float16 |
| `slices_test.npy` | 3560 × 256 × 256 | float16 |
| `slices_test200.npy` | 200 × 256 × 256 | float16 |

四文件合计 **6,456,607,232 字节**，本目录只包含校验元数据，不含数组、CT、图像、原始数据包或新增患者映射。数据应依据原数据集及镜像的许可，从其来源取得，再使用仓库预处理流程。校验和可用于检查相同输入和处理流程所得文件是否一致，不能单独证明无泄漏或跨版本数值等价。已有匿名评估索引仍位于仓库原 `results/`，本次不扩展其范围。

未来新实验的数据隔离与最终测试使用遵循 [Research Roadmap](../../../ROADMAP.md)，本归档不授权重新调参已查看的测试集。

## 权重与断点恢复的边界

[MANIFEST.json](MANIFEST.json) 列出在原盘观察到的 **14 个 checkpoint** 的大小、SHA256 及是否与已发布 Release 匹配。`included: false` 表示本次没有取回或发布该 checkpoint 二进制；有 Release 匹配链接的 5 个模型已有可下载副本。

- 已发布的 `prior_256_ema_final.pt` 足够用于旧模型推理或冻结先验的新方法研究，不等于完整训练断点。
- 原盘 `prior_256/last.pt` 为 **317,566,637 字节**，SHA256 `549e0af807eb45c93ad07e52e7c9b396f52440d575f0ee41cda58fe3fbd73068`；**本次只保留元数据，未备份其二进制**。
- 从保存代码可见，先验 `last.pt` 保存 model、EMA、optimizer、step 和 cfg，但未保存全部 RNG / 数据加载游标；只能说明恢复机制的边界，不能保证精确续接同一训练轨迹。
- FBPConvNet 现有保存格式及恢复接口不构成完整断点续训；Phase 1b 补全器的规划也不能被当作已完成实验或已备份权重。

因此，**“学术复现文本证据和已发布模型已核对”不等于“原盘全部最新进度已永久备份”**。未发布断点和医学数组如需长期保存，应另选授权的私有持久存储，不能仅保留本 manifest 后删除原盘。

## 本次验证记录

完成的是：采集包 SHA256、258 份原始文本哈希、固定提交逐文件对比、5 个 Release 权重大小/哈希匹配、UTF-8/JSON 读取、公开内容敏感字段检查以及 GitHub 回读核验。没有执行训练、数值复现实验、模型反序列化或重算论文指标。原项目没有提供可验证的完整历史环境锁文件/每次运行源码快照；这些缺口在此明确保留，不补造证据。
