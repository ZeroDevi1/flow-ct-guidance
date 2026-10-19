# 新颖性核查 · 方案 A（角度序列 SSM 补全器）+ 扩展 1（HLCC）+ 扩展 2（限角外推）

**日期**：2026-10-19　**方法**：arXiv 官方 API，21 组关键词组合（摘要/全字段，覆盖 2007–2026-10），去重后约 70 篇，逐篇读摘要；另核实顾问报告引用的 12 篇（全部存在）。相关条目已入 Zotero 分组「MedFusion P1 · Diffusion/Flow CT Inverse」，标签 `P2-novelty-check`。
**局限**：只查了 arXiv。期刊独有的论文（如 ViewTrans，Digital Signal Processing 2026）、会议未挂 arXiv 的工作、未公开的投稿都查不到。"未检索到"不等于"没有人做过"，投稿前需要在 Google Scholar / IEEE Xplore 上补查一次。

## 结论

| 组成 | 精确重合 | 最接近的已有工作 | 我们的区别 |
|---|---|---|---|
| A 核心：以 t 为条件、沿 θ 扫描的 SSM，在**图像域流先验的 ODE 采样器内**补全正弦图，并输出逐角度置信度，用于加权全角度 DC | **未检索到** | LUCID 2606.16212（流匹配 + 投影域 DC，按稀疏度自适应）；EFMCT 2603.00205；FCDM 2409.06714、HRSino 2506.08809、MR-SDE 2505.19385 / RNSD⁺ 2507.05647（**正弦图域**扩散补全，独立模型，不在图像先验采样器内）；TomoTransformer 2609.37605（确定性、注意力视角预测）；RCL-Mamba 2606.31353 / MambaRoll 2412.09331（监督双域 SSM，非生成式） | 补全器只有约 2M 参数，复用冻结的图像先验；补全随 t 变化，与 ODE 轨迹耦合；逐角度置信度直接作为 DC 的权重；用 SSM 的周期边界 s(θ+π,r)=s(θ,−r) 建模角度序列 |
| 扩展 1：Helgason–Ludwig 一致性条件（HLCC）作为**生成式采样内**的硬投影 / 训练约束 | **未检索到** | 1609.06604（经典 HLCC 滤波器，非学习，插出中间投影后做 FBP）；Epi-NAF 2411.06181（锥束对极一致性作为神经场损失） | HLCC 用在学习式补全器的输出和训练损失上，并与流采样结合 |
| 扩展 2：上述补全器从**插值**（稀疏视角）推广到**外推**（限角） | **未检索到**（SSM / 流相关） | 2505.19385、2507.05647（限角的正弦图扩散补全）；ExtraPolationNet 2103.05255；2009.01782（循环网络 + 正弦图一致性层） | 同一个补全器、同一个先验，用一次训练覆盖稀疏视角与限角 |

**判断**：A + 1 + 2 的**整体组合**在 arXiv 上没有找到相同的工作，可以推进。但相邻方向很拥挤，至少包括「流匹配 + 投影域 DC」「正弦图扩散补全」「双域 Mamba」这三类。论文必须把 LUCID、FCDM / RNSD⁺、TomoTransformer 作为直接对比或重点讨论的对象，新意要表述为"**采样器内、随时间变化、带置信度的角度序列补全，以及 HLCC 约束**"，而不是"首次把 Mamba 用于 CT"。

## 需要预先说明的技术风险（影响论文叙事）

1. **HLCC 可能与 DC 有重复作用**：加权最小二乘 DC 求解的是 argmin_x ‖W(A_full x − ŷ)‖²，结果天然落在 range(A) 内，而 range(A) 本身就满足 HLCC。所以在稀疏视角下，HLCC 投影带来的增益可能很小。它真正起作用的地方是：(a) 作为补全器的**训练损失**，约束低阶矩；(b) **限角外推**，缺失角度的低频矩可以由 HLCC 从已测角度推出来，这正是经典 HLCC 外推的用武之地。所以扩展 1 和扩展 2 应当一起做、一起评估。
2. **真实数据缺口**：全部实验都基于平行束仿真。可以免申请使用的真实投影数据有：2DeteCT（2306.05907，实验室扇束扫描，样品是坚果、果干，非人体）、DM4CT 的同步辐射数据集（zenodo 15420527）。两者都需要另写扇束或对应几何的算子。

## 检索式（便于复现）

`arxiv_search.py` 的 21 组检索式，记录在会话 scratchpad 中。主要关键词组合：sinogram × {mamba, state space, inpainting/completion/view synthesis, flow matching}；sparse-view / limited-angle × {mamba, flow matching, extrapolation}；Helgason-Ludwig / HLCC / consistency conditions / range conditions × {learning, diffusion}；dual-domain / projection domain × {diffusion, flow}。
