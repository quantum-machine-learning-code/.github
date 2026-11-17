# Quantum Machine Learning Code

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar-dark.png">
    <img src="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar.png" alt="Quantum Machine Learning Code mark" width="120">
  </picture>
</p>

> 公开的量子机器学习实验仓库组织。
> **Reproducible quantum–classical ML baselines — measured, not marketed.**

![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> **重要 · 对照口径修正（2026-11-16，优先于本页下方所有历史数值）**：
> 审查发现 `quantum-llm` 课题一的历史「量子 vs 经典」对照**不是干净消融** ——
> 两臂除中间变换外还差读出缩放（`scale` 5.0 vs 1.0）、初始化
> （`near_identity_init` True vs False）与中间层容量（12 vs 240 参数）；
> grokking 长训练里的 `RICH/*` 实际是经典臂；课题一评估只取最后位置
> logits，是一次性分类而非自回归生成；量子层逐位置运行、不承载跨步状态。
> 因此本页所有「量子层无可证实正贡献」「不宣称量子优势」「Holm 后 0 显著」
> 等表述**一律降级为「不同配置之间的观察」** —— 既不支持「没有量子优势」，
> 现已新增 `--matched` 严格对照：代码层硬断言两臂仅差 `transform_type`
> （13 项配置相同、两臂都不加小初始化补丁、不启用增益不等的 `LayerNorm`），
> 并在报差值前强制检查装置有效性，见
> [`docs/REVIEW.md` 第 14 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)。
> **装置诊断（2026-11-16 续）**：v1 的「对称小初始化补丁」本身造成两臂初始
> 输出标准差差 16 倍；v2 的 `LayerNorm` 对两臂有效增益 `2.57× / 40.31×`，
> 并非对称操作，两臂一起退到 `0.0062`；v3 两臂都退回默认初始化后尺度对齐
> （`0.258 / 0.286`），但统一 `lr=0.015` 下经典臂仍 4/5 seed 停在基线。
> 三版归档的 `device_validity.valid` 均为 `false`，只作配置诊断。
> 公开记录的小预算网格显示决定因素是 **lr 而非 scale**：`lr=0.003` 下两臂
> 正式 5 seed × 2000 步（[`docs/matched_v4_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/matched_v4_metrics.json)）
> **第一次通过装置有效性检查**：两臂各 5/5 seed 越过门槛，量子
> `0.3251±0.0464` / 经典 `0.2767±0.0917`。主指标 val acc 配对 Δ `+0.0485`
> 的 CI `[-0.0677,+0.1647]` 跨 0；Holm 后唯一显著项是
> `generalization_gap`（量子臂训练–验证间隙更大），不是准确率优势。
> 容量差与架构缺口仍未消除，因此这里只得到「两种变换实现都学动了」，
> **不是量子层因果效应，也不写作「无量子优势」**。
> **容量匹配**：`n_layers=20` 时量子臂 240 参数 = 经典 240（单步 `0.3865 s`），
> 但层数同时改变线路深度 / 表达力 / 梯度，参数量相等不等于唯一自变量已隔离。
> **更靠前的问题（§15）**：现有量子层逐位置独立、不承载跨步状态，课题一还是
> 一次性分类，因此再干净的对照也只测到「两种 pointwise 非线性」。已落地
> L1 测量反馈型量子循环原型 `qml_lab/recurrent.py` 与 L0/L1/L2 + P1–P4 架构
> 方案 [`docs/ARCHITECTURE_PROPOSAL.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/ARCHITECTURE_PROPOSAL.md)；
> 第 14 节全部归档统一标为「经典瓶颈基线」，不升级、不重新解释。
> **L1 短预算 pilot（2026-11-17，§16）**：`experiments/recurrent_pilot.py` 在
> 标准下一 token 预测（独立语料 k=4 覆盖率 0.3867）+ 延迟回忆任务上做
> 2×2 状态消融，代码层断言初始权重逐元素相同、只差 `feedback`。
> **P1–P4 结构契约 4/4 通过**（关反馈后前缀敏感度与跨步梯度严格归零）；
> **但功能有效性未证明**：延迟回忆上量子开/关都停在 `0.297`（随机基线
> `0.25`），只有经典正对照到 `0.594` —— 既不能写成「量子循环不能记忆」，
> 也不能写成「经典更好」。`T=8` 量子 `0.194 s/步`、外推 2000 步
> `0.11 h/seed`，单进程峰值 333 MB。见
> [`docs/recurrent_pilot.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/recurrent_pilot.json)。

## 我们在做什么

这里存放**可复现、可核对**的量子机器学习实验代码，聚焦量子计算与 AI 的交叉方向：把参数化量子电路放进经典深度学习训练循环，再和同规模经典模型放在同一套配置下比较。

**每个方向一个独立仓库**，各自采用 MIT 许可、代码自包含、可单独 clone 运行。

| 方向 | 仓库 | 内容 |
| --- | --- | --- |
| 量子视觉分类 | [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 变分量子分类器（VQC / 局部 Patch VQC / 混合 CNN+VQC / 量子核 QSVC）做手写数字识别，与经典 MLP / CNN / SVM 对照 |
| 量子 + 符号推理 / 语言模型 | [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 参数化量子电路作为**一个可微分层**：课题一为模 112 加法（117-token 词表、零重叠组合泛化检验），课题二为字符级自回归语言模型（含 $k$-gram 熵下界证明、多可观测量读出与因果上下文混合的玩具级验证） |

## 仓库

| 仓库 | 说明 | 状态 |
| --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | VQC / 振幅编码 / 最多 165 维多可观测量读出 / 局部 Patch 权重共享 VQC / Pérez-Salinas 可训练重上传 / 量子核 QSVC、8 轮多 seed 统计对照与**全量 MNIST（60000/10000）实跑**，附真实实测指标、对比图与方法学审查 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | 活跃 |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 玩具级量子增强字符 LM：修复无记忆 Bigram 瓶颈（`1.9701` nats 下界）与批次混淆，提供 5-seed 受控实验（48 维两体 Pauli 读出 + 因果窗口 $k=4$ 突破 Bigram 墙至 train `0.8271` nats / val `1.4494` nats；该留出集与训练集 k=4 上下文重叠 90.2%）；2026-11-13 补句级 disjoint 独立语料重跑（重叠率降至 **38.7%**），量子层训练集优势显著但独立验证句上经典替身数值更好、量子泛化间隙显著更大（5-seed `+0.2044` nats，20-seed 修正为 `+0.1433`）；2026-11-13 稳健性复核（20 seed × 1200 步）确认是稳态过拟合而非欠拟合，附方法学审查 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)；2026-11-13 新增**课题一：模 112 加法**（117-token 词表、验证集 $k=4$ 上下文重叠率 **0%**、5-seed × 2000 步；量子 val `0.1817 ± 0.0423` vs 经典消融 `0.1365 ± 0.0410`，3 个主检验 Holm 后 0 显著，量子层无可证实正贡献），见 [`docs/REVIEW.md` 第 10–12 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)（第 11 节为 embedding 三模式消融，第 12 节为 grokking 长训练扫描与正对照） | 活跃 |
| `.github` | 组织 profile 与共享文档（本仓库） | — |

## 我们的原则

1. **不宣称量子优势。** 量子模型在这个规模下常常打平甚至落后于经典基线，量子与经典两组数字会一起公开 —— 两个仓库的实测结果都是如此。
2. **说清边界。** 状态向量模拟是 $2^n$ 内存墙，4–12 qubits 是笔记本舒适区；当前条件下不存在可行的量子 LLM。`quantum-llm` 只是沿用方向命名的**玩具级可微分层验证**，README 开头就写明了这一点，它不是能对话的大模型；课题一的模 112 加法同样只是玩具级可微分层验证，不是算术求解器。
3. **可复现，且承认错误。** 固定种子、离线数据、精确复现命令；结论性数字带多 seed 均值与标准差。`quantum-mnist` 与 `quantum-llm` 均做过系统的方法学审查：定位并修复了种子与批次混淆、单步评估噪声、无记忆展平（`(B, T, Q) -> (B*T, Q)`）等问题，把问题清单、实测证据与消融对照公开写在各自的 `docs/REVIEW.md` 里。2026-11-07 的第八轮外部审查又补进四条结论：「量子核等于经典二次核」「量子层消融打平」「20 项配对检验 Bonferroni 后仅 2 项显著」「验证集与训练语料 90.2% 上下文重叠」，见 [`quantum-mnist/docs/REVIEW.md` 第 15 节](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) 与 [`quantum-llm/docs/REVIEW.md` 第 7 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)；README 与本主页的表述已据此逐条收紧。2026-11-13 又补上独立语料重跑：把验证集 $k=4$ 上下文重叠率从 90.2% 降到 38.7% 后，量子层在训练集上的显著优势在独立验证句上消失并方向反转（两次 val CI 都跨 0），见 [`quantum-llm/docs/REVIEW.md` 第 8 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)。2026-11-13 的稳健性复核又把它扩到 20 seed × 1200 步：泛化间隙差值由 `+0.1433` 扩大到 `+0.3263`（Holm 后仍显著），量子 val loss 单调上升而 train loss 持续下降，排除「训练不足」解释；但在极端分布偏移的合成语料上未复现，见第 9 节。
4. **不需要真机。** 全部实验在经典模拟器上完成，无需 IBM Quantum 等平台账号或 token。

## 技术栈

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn / SciPy / NumPy / Matplotlib；Python 3.11+；默认 `default.qubit` 状态向量模拟，解析求导，不接真机。两个实验仓库共享同一套核心实现，但各自保留所需模块，避免跨仓库依赖影响复现。

## 进展与路线图

### 1. `quantum-mnist`（8 轮实测与归因）
- **已完成（Rounds 1–7）**：amplitude encoding 无压缩对照、输入维度与电路深度容量扫描、方法学审查、5 seed 重复实验 + 配对差值置信区间（`train_multiseed.py`）、改良分类头与输出缩放、9 组 × 5 seed 瓶颈分解与量子读出信息探针（`diagnose_ceiling.py`）、11 组 × 5 seed 多可观测量 / 数据重上传 / 单独 lr 对照（`break_readout.py`）、**全两体关联读出（最多 165 维）**、**信息上界分解诊断**（`probe_information.py`）、**9 组 × 5 seed 加宽读出 + 数据规模对照**（`push_ceiling.py`）、**等数据 / 近似等参数预算经典对照**（`classical_budget.py`）、**六轮剩余差距归因**（`why_low_acc.py`）与**全量 MNIST（60000/10000）实跑**（`train_full_mnist.py`）。
- **读出瓶颈解开与全量实跑**：10 维 `PauliZ` 读出可分性上限仅 0.679（像素探针 0.855）；扩到 40 维后 5 seed 准确率 `0.6640 → 0.7720`，再扩到 165 维（全部 ZZ/XX/YY 两体关联）达 **`0.8450 ± 0.0100`**（配对 `+7.60 ± 2.38 pt`），1800 张真图达 **`0.8660 ± 0.0185`**。全量 MNIST（60000 训练 / 10000 测试，3 epochs × 3 seeds）上，量子 165 维达 **`0.9421 ± 0.0008`**（40 维 0.9014），等预算经典 `PCA-165 + MLP` 为 **`0.9554 ± 0.0028`**（差距收窄到 **1.33 pt**，配对 CI `[+0.45, +2.20]`，经典仍显著领先且快约 1000 倍）。冻结随机酉 165 维探针 `0.9440` 与端到端 `0.9421` 打平，说明剩余差距来自全局齐次二次型缺少局部空间归纳偏置（1px 平移扰动达类内离散度的 1.20 倍）。详见 [`quantum-mnist/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) 第 10–13 节。
- **第八轮突破（`resolve_open_issues.py`，解决遗留问题与补齐基线）**：
  1. **解决 Hybrid CNN + VQC 训练崩溃**：定位崩溃根因是轻量 CNN 池化后跨样本特征标准差（`~0.003`）比线性层随机偏置小 50 倍且缺乏归一化；通过 `BatchNorm1d + Tanh*(π/2) + 近恒等初始化 N(0, 0.1²) + 32 维读出`（纯串联、**无跨量子层残差跳连**），`hyb_light_stab` 从 `0.5980 ± 0.1062` 提升至 **`0.7650 ± 0.0464`**（配对 `+16.70 ± 11.70 pt`，未校正 CI `[+2.18, +31.22]`，0/5 seed 崩溃；20 项 Bonferroni 校正后不显著），追平跳连基线 `nq_light`（`0.7440 ± 0.0651`）。
  2. **Pérez-Salinas 可训练数据重上传**：实现逐层可训练输入缩放与偏置 $\text{RY}(\omega_{l,i}\tilde{x}_i + \theta_{l,i})$，在 Z-only 读出下从朴素重上传的 `0.5220 ± 0.0652` 恢复到 **`0.6580 ± 0.0246`**（`+13.60 ± 7.41 pt`，未校正），配合 40 维读出达 **`0.7470 ± 0.0214`**（较单次上传 `a10_z` `+11.50 ± 4.65 pt`，同样未校正；两项经 Bonferroni 校正后均不显著）。
  3. **补齐解析核（QSVC）第三条基线**：5-seed 实测 10-qubit 全局振幅保真度核 `qkernel_amp` 为 **`0.9210 ± 0.0074`**，8-qubit 局部 9-patch 保真度核 `qkernel_patch` 为 **`0.9310 ± 0.0164`**。**审查修正**：这两项都是解析可算的经典二次型核（`qkernel_amp` 与不加 `StandardScaler` 的经典 poly2 核在 5/5 seed 上数值完全相同），`svm_poly2` `0.8980` 是另一套预处理口径下的数字，两者不可混比；因此不能表述为「量子核显著领先经典」，也不应与端到端量子模型混排成同一列准确率。详见 [`docs/REVIEW.md` 第 15.1 节](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md)。
  4. **局部空间归纳偏置（`PatchVQCClassifier`）**：将 $28\times 28$ 图像切为 $3\times 3$ 重叠 $14\times 14$ 局部 patch（stride 7），共享同一个 8-qubit 振幅编码电路（仅 48 个量子参数）并保留相对 patch 范数，5-seed 准确率达 **`0.8800 ± 0.0112`**（对 `amp_b165` `+3.20 ± 2.02 pt`、对 `ref_pca165` `+3.90 ± 1.71 pt`，均为**未校正** CI 不跨 0，Bonferroni 校正后不显著）。**量子层消融**：把该 8-qubit 线路换成经典 `Linear(196,32)+Tanh`、其余逐项不变，得 `0.8920 ± 0.0067`（配对 +1.20 pt，CI 跨 0），还快约 50 倍 —— 提升主要来自经典 2D 局部窗口归纳偏置，**量子层无可证实正贡献**。比 `amp_b165` 快 5.7 倍只是 8-qubit 相对 10-qubit 态向量的模拟优势。详见 [`quantum-mnist/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) 第 14–15 节。

### 2. `quantum-llm`（课题一：模 112 加法；课题二：字符 LM 方法学审查）
- **定位并证明无记忆 Bigram 墙**：初版 `TinyLM` 将 `(B, T, Q)` 展平为 `(B*T, Q)` 逐位置独立过变换层，`--seq-len 16` 完全未跨位置交互（有效上下文 $k=1$）。内置语料（1668 字符，$V=31$）的精确 $k=1$ 条件熵下界为 **`1.9701` nats（困惑度 `7.17`）**，且精确 Bigram Oracle 也会因 `' ' -> 't' -> 'h' -> 'e' -> ' '` 最高频环确定性陷入 `"A qubit can hold the the the..."` 死循环；而 $k=4$ 因果上下文的理论熵下界骤降至 **`0.1000` nats（困惑度 `1.11`）**。
- **修复混淆变量与 5-Seed 受控对照（`diagnose_llm.py`）**：修复了量子/经典使用不同 batch 随机种子（`seed` vs `seed+1`）、共享层初值不同、`iter_batches` 漏采末尾字符以及单步 batch loss 噪声，引入全语料确定性评估 `evaluate_lm` 与留出句验证集（`_HELDOUT_VAL_CORPUS`，382 字符；语料审查显示其在 $k=4$ 上下文上与训练集重叠 90.2%，并非独立未见分布）：
  - **单字符模式（$k=1$）**：修复 logit 收缩（`scale=5.0` + `tanh*π/2`）并将 4-qubit 读出从 4 维 `PauliZ` 扩至 48 维全两体 Pauli 基（`z+x+y+zzall+xxall+yyall`）后，`tok_q_rich48` 训练 loss 从 `2.3864 ± 0.0553`（ppl `10.89`）降至 **`2.0631 ± 0.0148`（ppl `7.87`）**，逼近 $k=1$ 理论下界 `1.9701` nats。
  - **因果上下文模式（$k=4$，`CausalContextEncoder` 严格左填充因果卷积）**：`caus_q_rich48`（4639 参数）将训练 loss 推进至 **`0.8271 ± 0.0717` nats（ppl `2.29 ± 0.16`）**、留出句验证 loss 达 **`1.4494 ± 0.1290` nats（ppl `4.29 ± 0.57`）**；训练集上优于无量子层消融 `caus_identity`（train `1.6549` / val `1.9394`）与等预算经典基线 `caus_classical`（4867 参数，train `1.1483 ± 0.0617` / val `1.6266 ± 0.1166`；验证集配对差值 `-0.1772` nats、95% CI `[-0.4019, +0.0475]` 跨 0 不显著；且该留出集 90.2% 的 $k=4$ 上下文训练时已见过，`val PPL 4.29` 不是独立泛化证据），并彻底跳出 `"the the the"` 贪心循环。**独立语料复核（2026-11-13，[`independent_corpus.py`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/experiments/independent_corpus.py)）**：换成 44 句训练 / 14 句主题不同、句级 disjoint 的新语料（3171 / 947 字符，$k=4$ 上下文重叠率 **38.7%**，新 4-gram 69.3%，验证集零 UNK），其余配置逐字不变、5 seed：量子 `caus_q_rich48` train **`2.1045 ± 0.0200`** vs 经典 `caus_classical` `2.2115 ± 0.0169`（配对 `−0.1069`，4 检验 Bonferroni 后仍显著），但独立 val 为 `2.5756 ± 0.0692` vs **`2.4782 ± 0.0385`**（配对 `+0.0974`，CI 跨 0），泛化间隙 `0.4711` vs `0.2667`（差值 **`+0.2044`**，Bonferroni 后显著）。即 **量子层的可复现优势只在拟合训练集上成立，不是泛化能力**；旧语料 90.2% 的重叠把 val 伪装成了泛化证据。归档案 [`docs/independent_corpus_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/independent_corpus_metrics.json)。详见 [`quantum-llm/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) 第 7–8 节。**稳健性复核（20 seed × 1200 步，[归档](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/robustness_metrics.json)）**：泛化间隙差值由 `+0.1433`（300 步）扩大到 `+0.3263`（1200 步，Holm 后显著），5-seed 的 `+0.2044` 属小样本高估；量子 val loss 从 `2.5305` 单调升到 `2.7431`、train loss 从 `2.2982` 持续降到 `1.8922`，**排除 300 步欠拟合假象，是持续加剧的过拟合**。20 seed 下 val 差值也变为显著（`+0.0599`，Holm `p=0.0022`）。详见第 9 节。

### 3. 诚实边界与下一步
- **仍不宣称量子优势**：`quantum-mnist` 全量下经典仍领先 1.33 pt 且快约 1000 倍，`PatchVQC` 的提升主要来自经典 2D 滑动窗口局部性与二次型保真度特征；`quantum-llm` 的 48 维两体 Pauli 读出等价于 4-qubit 因果特征上的固定二阶多项式核展开，状态向量模拟比经典层慢约 40 倍。课题一的 3 个主检验 Holm 校正后同样 0 个显著，量子臂 val 准确率虽高 `4.51 pt` 但配对 CI `[-0.0003, +0.0905]` 跨 0。全部结果来自 `default.qubit` 解析模拟，不含真机状态制备与有限 shots 成本。
- **第八轮外部审查收紧了四条结论**：① `qkernel_amp` / `qkernel_patch` 是经典可算的解析二次核，不是量子模型优势证据；② `PatchVQC` 换成经典线性投影后 `0.8920 ± 0.0067` 打平量子 `0.8800 ± 0.0112`（CI 跨 0，快约 50 倍），量子层无可证实正贡献；③ 20 个配对检验 Bonferroni 校正后由 13/20 变为 2/20 显著，README 中的「显著」已逐条降级；④ `quantum-llm` 训练语料是 417 字符重复 4 遍、留出集 $k=4$ 上下文 90.2% 训练时已见过，`val PPL 4.29` 只能读作同分布重组句迁移。详见两仓 `docs/REVIEW.md`。
- **独立语料证据链已补齐（2026-11-13，原优先级①）**：`quantum-llm` 换用句级 disjoint、主题分离语料重跑并报告逐 $k$ 上下文重叠率（$k=4$ 由 90.2% 降到 **38.7%**）。结论：量子层在训练集上显著更好（`−0.1069` nats），但独立验证集上经典替身数值更好（`+0.0974` nats，CI 跨 0），且量子层泛化间隙显著更大（`+0.2044` nats，Bonferroni 后仍显著）。**不宣称量子优势，也不宣称量子显著更差**；字符级英文仍共享 38.7% 功能 4-gram，不是严格 OOD。顺带修复 `audit_corpus.py` 的 `max_context` 参数名错误，使 `corpus_audit.json` 重新归档条件熵下界。
- **稳健性复核已部分完成（2026-11-13，原优先级③）**：独立语料扩到 **20 seed × 1200 步**，并新增 `synthetic` 受控合成语料（`k=4` 覆盖率 **0.24%**、字符 unigram JS **0.2538**，约为 `independent` 的 27 倍）。结论：① 5-seed 的 gap `+0.2044` 修正为 20-seed `+0.1433`（seed 0–4 子集逐位复现原值，高估完全来自抽样）；② 延长训练后差值扩大到 `+0.3263`（Holm 后显著），**不是欠拟合假象**；③ 20 seed 下独立语料 val 差值显著（`+0.0599`，Holm `p=0.0022`）。但 `synthetic` 上两臂 val loss 均崩到 5+ nats、gap `+0.5602` 的 CI `[-0.1406, +1.2609]` 跨 0 不显著 —— **结论有语料边界，不推广为普遍的「量子泛化更差」**。51 个检验 Bonferroni 后 35 显著，6 个主检验 Holm 后 4 显著；两臂始终同配置、**均未各自调参**。见 [`docs/REVIEW.md` 第 9 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) 与 [`docs/robustness_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/robustness_metrics.json)。
- **课题一：模 112 加法（2026-11-13 新增）**：因第 7–9 节反复暴露自写自然语言语料太小、训练/验证上下文重叠过高，新增一个**可穷举、零重叠**的符号任务作为 `quantum-llm` 的第一个课题。词表 **117 = 112 个数字 token + `+ - * /` 四个运算符 + `=` 一个等号**，`111+1=0`（回绕）与 `0+0=0`（恒等元）编码为 5 个 token；答案位为 112 类单 token 分类（随机基线 `1/112`）。全空间 12544 个有序对按 `(a,b)` 组合严格 disjoint：训练 10035 / 验证 2509，**验证集 $k=4$ 上下文覆盖率 0.0%**（而 k=1/k=2 为 100%，即数字都见过、组合一个没见过），把历史语料的 90.2% / 38.7% 彻底归零。5 seed × 2000 步：量子 `add_q_rich48` 未见组合 val **`0.1817 ± 0.0423`**、经典消融 `add_classical` **`0.1365 ± 0.0410`**，两臂均显著高于随机基线（二项检验 $p\approx0$），但准确率仅 18% / 14%、曲线仍上升，**任务远未解决**；val 准确率配对 `+0.0451`、95% CI `[-0.0003, +0.0905]` 跨 0，**3 个主检验 Holm 后 0 个显著**，`val_loss` 虽在 Bonferroni 后显著但不是主指标 —— **量子层无可证实的正贡献**。两个必要前提：随机 embedding 下扫过 3 个学习率、跑到 25.5 epochs 两臂都卡在随机基线，因此加了**两臂共享的循环角度编码**（$\sin/\cos(2\pi n/112)$）这一经典先验，它不是量子优势；两臂始终同配置、**均未各自调参**。课题一的泛化间隙差值 `−0.0023`（CI 跨 0）与课题二第 9 节的 `+0.3263` 相反，进一步印证「量子泛化更差」不是跨任务的普遍性质。见 [`docs/REVIEW.md` 第 10 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) 与 [`docs/addition_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/addition_metrics.json)。
- **课题一 embedding 三模式消融（2026-11-13 新增）**：先澄清一个措辞问题 —— 循环角度编码只是**初始化**，默认 `trainable_circular` 下 embedding 本来就在训练（2000 步后 RMS 漂移 `0.2816` / `0.3674`），「放开再训练」并不是新设置。用统一预算（5 seed × 2000 步、同切分与同 batch 顺序）跑 `frozen_circular` / `trainable_circular` / `trainable_random` 三模式，前两者初始权重逐位相同、只差一个 `requires_grad`：冻结时量子 val `0.2418 ± 0.0488`、经典 `0.1839 ± 0.0427`，高于默认的 `0.1835 ± 0.0463` / `0.1462 ± 0.0447`；放开效应配对 `−0.0583` CI `[-0.1318, +0.0153]` / `−0.0377` CI `[-0.1187, +0.0433]`，两臂方向一致但 **CI 跨 0，不宣称显著有害**。纯随机 embedding 即使放开，loss 仍停在 `4.735 ≈ ln(112)`（val `0.0062`），真正显著的是循环先验（`+0.1773` / `+0.1400`，CI 均不跨 0）。**9 个主检验 Holm 后 0 个显著** —— 放开或冻结都不改变「量子层无可证实正贡献」的结论；且设计选择的影响（−5.8 pt）与量子层差异（+5.8 pt）同量级，说明未做设计敏感性分析前不宜把两臂差值归因于量子层。2000 步仅约 6.4 epochs、曲线仍上升，「冻结更好」有明确的训练预算边界。见 [`docs/REVIEW.md` 第 11 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) 与 [`docs/embedding_ablation_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/embedding_ablation_metrics.json)。
- **课题一 grokking 长训练扫描（2026-11-13 新增）**：按 `(a,b)` 组合留出 + 随机可训练 embedding 正是 Power et al. (2022, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)) 的 **grokking** 原始设置；而第 10 节手工注入的循环角度编码恰好是 Nanda et al. (2023, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)) 发现模型最终自己学出来的 $\sin/\cos$ 傅里叶表征，等于把终点当起点。本轮核实文献并实跑 1-seed pilot（19 条件 × 20000 步）→ 5-seed 确认（40000 步）→ 量子臂有限预算示意，另加 Liu et al. (2022, [arXiv:2205.10343](https://arxiv.org/abs/2205.10343)) 式的**二次 MLP 正对照**与傅里叶 progress measure。**grokking 确实复现了，但不在本仓库模型里**：正对照在 20% 训练数据 + AdamW `wd=1.0` 下 5/5 seed 出现「1000 步完成记忆 → 1500 步突发泛化（单步跳变 `+0.667`）」，val **`0.9944 ± 0.0006`**；同一任务、同一随机 embedding、同一零重叠切分下，把 `RichQuantumLM` 预算从 2000 步拉到 **40000 步（20 倍）**并扫 6 档 weight decay，仍 **0/5 seed 出现相变**（val `0.0255 ± 0.0411`）。同切分架构差距 `+0.9910` CI `[+0.9877,+0.9943]`，是 3 个主检验 Holm 后**唯一显著项**。**⚠ 上述「瓶颈是架构」的结论已撤回（2026-11-16 修正）**：正确的对应关系是「加循环角度先验 = 把 grokking 最终学到的 sin/cos 表征直接注入、跳过 grokking；不加先验 = grokking 起跑线」。第 12 节答非所问地另造了一个能 grok 的二次 MLP 来论证本仓库架构不行，且那组随机 embedding **连记忆阶段都没进入**（40000 步后 train acc 仅 `0.0288`）、两架构的 `lr` / `wd` / 训练数据占比 / 步数四项全不同，`+0.9910` 不是干净的架构效应。修正实验 `grokking_no_prior.py` 给两架构完全同超参并推到 $10^5$ 步，见 [`docs/REVIEW.md` 第 13 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)。**修正复验结果（60 条件 1-seed 网格 + 2 条件 × 5 seed × $10^5$ 步确认，两架构完全同 `lr=0.003 / vf=0.8 / batch / seed / 切分`）**：不加先验的 `RichQuantumLM` 在 `wd=0` 下 train 从 `0.09` 爬到 **`0.6565 ± 0.0469`**、val 全程 `≈0.01` —— 这正是 grokking 的「记忆中、尚未泛化」阶段，10 万步**连第一步（记忆）都没走完**；同超参正对照 `wd=1` 组 5/5 完成泛化 val **`0.9918 ± 0.0030`**。最硬的反证是：同一个架构、同样不加先验，只把 `lr` 从 `0.015` 降到 `0.003`，train acc 就从 `0.009` 变成 `0.66`，可见上一轮「架构瓶颈」主要混淆了超参。**现在仍不能下架构结论**：grokking 要求先记忆，rich 连 `train≥0.95` 都没到，0 相变是必然结果而非证据；课题一的定位应改写为「**人工注入 grokking 终点表征后的模加法分类**」，其量子 vs 经典对照都是在跳过 grokking 的任务上做的。（stage2 `eval_every=2000` 漏掉正对照 1000–1500 步的突发跳变，严格相变判定全 0/5 属采样伪影；装置有效性由 val acc 与 stage1 细采样 `grok=True` 证明。）见 [`docs/grokking_no_prior.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/grokking_no_prior.json)。附带结论：冻结在 40000 步后仍未反转（`0.4207` vs `0.2736`，`p=0.041` 但 Holm 后 `p=0.082` 不显著），前 4 维相位保持度恒 `1.000` vs 降到 `0.614` 且与准确率相关（探索性 `r=0.69`），证实了第 11 节猜测的机制；量子臂实测 `0.1264 s/步` vs 经典 `0.0038 s/步`（**33.6×**），$10^5$ 步需 `3.5 h` vs `0.1 h`，本轮只跑 2500 步（2.5%）仅作成本示意。边界：pilot 为 1-seed 探索、存在选择偏差；相变条件用了 80% 留出切分，干净架构比较用同切分那组；此处相变发生在 500–1500 步，比 transformer 文献快几个数量级，只能说现象同类。见 [`docs/REVIEW.md` 第 12 节](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) 与 [`docs/grokking_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/grokking_metrics.json)。
- **当前优先级（2026-11-17 更新，覆盖下方旧 roadmap）**：架构线已推进到
  L1 量子循环。`recurrent_pilot.py` 证明 P1–P4 结构契约 4/4 通过，但延迟回忆
  上量子两臂仍停在随机基线 —— **先解决状态的功能有效性**（延长训练、扫状态维
  Q / 线路深度 / lr，用延迟回忆当装置门槛），通过之后才做 A2/A3 同构对照
  （≥5 seed、配对 CI、Holm）与 L2 隐量子态延续。这些都是长训练，需先确认
  预算与内存窗口。下方 ①–⑥ 是 2026-11-13 的历史 roadmap，保留供追溯，
  但其中「用能 grok 的架构做量子层消融」仍受 §14 对照口径修正约束。
- **下一步（按优先级）**：① **把不加先验的 `RichQuantumLM` 推到记忆完成**（$10^6$ 步量级，或调容量 / lr / wd 组合），并用 `eval_every≤250` 细采样重跑以捕捉突发相变 —— 第 13 节已确认 $10^5$ 步时 train 仅 `0.66`、连记忆都没完成，此时谈「架构表示能力」为时过早；② **给两臂各自调参**（统一 `lr=0.015` 已被证实是主要混淆，同架构降到 `0.003` 后 train 从 `0.009` 变成 `0.66`），并重跑课题一、课题二独立语料与 1200 步收敛扫描；③ 用能 grok 的架构做量子层消融，才谈得上「量子层对相变有无影响」；④ 课题一 `holdout_operand` 数值外推、换减法 / 乘法（`/` 需处理 $\mathbb{Z}_{112}$ 非域问题），课题二换真实外部**自然**语言语料并预注册 primary endpoints；⑤ 把全量 MNIST 对照扩到 $\ge 20$ seed 并做多重比较校正，实跑 `PatchVQCClassifier` 与 `qkernel_patch` Nyström 近似；⑥ 探索量子自注意力打分（Quantum Attention）与有限 shots（按 QWC 3 基分组测量）/ 退极化噪声基准，并把量子臂预算推到 $10^4$ 步以上。

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线与无量子层（`Identity`）消融对照、在**构造模型之前**固定随机种子，并在 README 与 `docs/REVIEW.md` 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。结论性差异需要多 seed 均值与置信区间支撑（小样本效应量可能高估，扩充 seed 后请复报）；≥10 个配对检验请报告 Bonferroni/Holm 校正后的显著数；报告验证集泛化时给出与训练集的 n-gram 上下文重叠率；「量子核」「量子层有效」类结论必须附同预处理经典核与量子层消融对照；报「训练变好」时同时报独立验证集结果与泛化间隙，不要把拟合改善写成泛化改善；报「泛化更差」时要延长训练排除欠拟合解释，并说明结论的语料边界。两个实验仓库均为 MIT 许可。
