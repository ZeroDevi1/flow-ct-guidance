# 2026-10-31 P0/P1 复现证据（匿名聚合）

本目录由用户明确授权公开。仅包含代码可复算的匿名汇总、训练曲线、数值控制与 checkpoint 哈希；**不含医学图像、患者 ID、切片索引、原始逐图记录或完整训练 checkpoint**。旧仓库其他目录不在本次新增材料隐私审查范围。

## 已完成，不代表独立验证

- 同一冻结 flow prior；20 名开发验证患者 × 2 片 × 3 个配对种子。
- 180 角平行束、Poisson I0=1e5、sigma=0、mu_scale=.06；NFE50 / CG30，DC λ3 与 WLS λ.3。
- 20 人包含4名先导调参患者；剩余37名验证患者图像、本轮89名测试患者图像均未读取。该表述**仅限本轮**，不重写历史 test200 的已查看事实，也不对先验历史选模过程作额外保证。
- 匹配低剂量 FBP 残差 U-Net 从头训练至20k上限，固定 best 在同20人/3seed评估。不是“已证明收敛”。
- 主结果：FBP / DC / WLS / FBP网络 PSNR 分别27.2668 /25.8361 /31.7144 /37.2846 dB。WLS 对弱 DC 的优势不等同超过强基线。
- 同图 NFE20→50 退步、float32/CG12 单位权重门控失败均保留；不改阈值掩盖失败，不挑漂亮 seed。

## 文件入口

- [`p1/deidentified_result.json`](p1/deidentified_result.json)：完整240次评估；患者级配对差与探索性 bootstrap。
- [`p1/deidentified_cohort_audit.json`](p1/deidentified_cohort_audit.json)：精确同图 NFE/队列审计，不能简单把下降归因解剖。
- [`baseline/evaluation/summary_anonymous.json`](baseline/evaluation/summary_anonymous.json)：固定 best 的120次配对预测。
- [`baseline/training/`](baseline/training/)：每50步抽样训练曲线/完整验证曲线、停止原因、完整私有 checkpoint 的校验值（不是可公开权重资产）。
- [`numerics/`](numerics/)：原始失败与同 solver 重复性；[`pilot/`](pilot/)：修订双门控与先导结果；[`cg_sensitivity/`](cg_sensitivity/)：真实开发子集 CG 敏感性。
- [`reproducibility_manifest.json`](reproducibility_manifest.json)：源代码、数据文件校验值、环境和隐私范围；[`SHA256SUMS.txt`](SHA256SUMS.txt)：本目录数据文件校验。

## 复现与核验

运行入口在 [`code/experiments/budget_pilot`](../../code/experiments/budget_pilot/README.md)。先准备合法取得的 TotalSegmentator 数据、复核数据 hash 和原先验 hash，再按固定患者排序+seed20261031 洗牌选择队列。公开的是选择算法和输入哈希，不公开患者清单；独立从匿名汇总无法重新计算全部患者 bootstrap，需要在有权访问原数据的环境重跑。

实验采用 GPU torch2.12.1+cu130/RTX5090；CPU测试是另一环境，不能冒充 GPU 确定性证明。代码保持历史阶段脚本用于溯源：**旧 `run_pilot.sh` 会请求关机；当前重现应阅读 `run_stage_v2.sh`/批次说明并设置自己的预算截止。** 示例日期是历史记录，不应原样用于新租机。

完整checkpoint含模型、优化器、RNG、公开数据集编号和通用数据路径。用户已在知悉公开范围后澄清这些编号来自公开数据集，并授权发布原始完整 best/last 断点。秘密与官方数据许可检查通过；作为 GitHub Release 资产上传，不写入 git。文件 SHA 保持不变，见 [发布范围与数据署名](CHECKPOINT_PUBLICATION.md)；实际下载链接以最终 Release 为准。

训练逐步原始JSONL的19800行未改写：git收录每50步原行以避免大API请求；原始行数/SHA见 `baseline/training/train_log_provenance.json`，完整匿名日志仍保留在源证据归档中。
