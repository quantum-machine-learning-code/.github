# Quantum Machine Learning Code

> 公开的量子机器学习实验仓库组织。
> **Reproducible quantum–classical ML baselines — measured, not marketed.**

![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

## 我们在做什么

这里存放**可复现、可核对**的量子机器学习实验代码，聚焦量子计算与 AI 的交叉方向：把参数化量子电路放进经典深度学习训练循环，再和同规模经典模型放在同一套配置下比较。

**每个方向一个独立仓库**，各自采用 MIT 许可、代码自包含、可单独 clone 运行。

| 方向 | 仓库 | 内容 |
| --- | --- | --- |
| 量子视觉分类 | [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 变分量子分类器（VQC / 局部 Patch VQC / 混合 CNN+VQC / 量子核 QSVC）做手写数字识别，与经典 MLP / CNN / SVM 对照 |
| 量子 + 语言模型 | [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 参数化量子电路作为**一个可微分层**嵌入字符级自回归语言模型（含 $k$-gram 熵下界证明、多可观测量读出与因果上下文混合的玩具级验证） |

## 仓库

| 仓库 | 说明 | 状态 |
| --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | VQC / 振幅编码 / 最多 165 维多可观测量读出 / 局部 Patch 权重共享 VQC / Pérez-Salinas 可训练重上传 / 量子核 QSVC、8 轮多 seed 统计对照与**全量 MNIST（60000/10000）实跑**，附真实实测指标、对比图与方法学审查 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | 活跃 |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 玩具级量子增强字符 LM：修复无记忆 Bigram 瓶颈（`1.9701` nats 下界）与批次混淆，提供 5-seed 受控实验（48 维两体 Pauli 读出 + 因果窗口 $k=4$ 突破 Bigram 墙至 train `0.8271` nats / val `1.4494` nats），附方法学审查 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) | 活跃 |
| `.github` | 组织 profile 与共享文档（本仓库） | — |

## 我们的原则

1. **不宣称量子优势。** 量子模型在这个规模下常常打平甚至落后于经典基线，量子与经典两组数字会一起公开 —— 两个仓库的实测结果都是如此。
2. **说清边界。** 状态向量模拟是 $2^n$ 内存墙，4–12 qubits 是笔记本舒适区；当前条件下不存在可行的量子 LLM。`quantum-llm` 只是沿用方向命名的**玩具级可微分层验证**，README 开头就写明了这一点，它不是能对话的大模型。
3. **可复现，且承认错误。** 固定种子、离线数据、精确复现命令；结论性数字带多 seed 均值与标准差。`quantum-mnist` 与 `quantum-llm` 均做过系统的方法学审查：定位并修复了种子与批次混淆、单步评估噪声、无记忆展平（`(B, T, Q) -> (B*T, Q)`）等问题，把问题清单、实测证据与消融对照公开写在各自的 `docs/REVIEW.md` 里。
4. **不需要真机。** 全部实验在经典模拟器上完成，无需 IBM Quantum 等平台账号或 token。

## 技术栈

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn / SciPy / NumPy / Matplotlib；Python 3.11+；默认 `default.qubit` 状态向量模拟，解析求导，不接真机。两个实验仓库共享同一套核心实现，但各自保留所需模块，避免跨仓库依赖影响复现。

## 进展与路线图

### 1. `quantum-mnist`（8 轮实测与归因）
- **已完成（Rounds 1–7）**：amplitude encoding 无压缩对照、输入维度与电路深度容量扫描、方法学审查、5 seed 重复实验 + 配对差值置信区间（`train_multiseed.py`）、改良分类头与输出缩放、9 组 × 5 seed 瓶颈分解与量子读出信息探针（`diagnose_ceiling.py`）、11 组 × 5 seed 多可观测量 / 数据重上传 / 单独 lr 对照（`break_readout.py`）、**全两体关联读出（最多 165 维）**、**信息上界分解诊断**（`probe_information.py`）、**9 组 × 5 seed 加宽读出 + 数据规模对照**（`push_ceiling.py`）、**等数据 / 近似等参数预算经典对照**（`classical_budget.py`）、**六轮剩余差距归因**（`why_low_acc.py`）与**全量 MNIST（60000/10000）实跑**（`train_full_mnist.py`）。
- **读出瓶颈解开与全量实跑**：10 维 `PauliZ` 读出可分性上限仅 0.679（像素探针 0.855）；扩到 40 维后 5 seed 准确率 `0.6640 → 0.7720`，再扩到 165 维（全部 ZZ/XX/YY 两体关联）达 **`0.8450 ± 0.0100`**（配对 `+7.60 ± 2.38 pt`），1800 张真图达 **`0.8660 ± 0.0185`**。全量 MNIST（60000 训练 / 10000 测试，3 epochs × 3 seeds）上，量子 165 维达 **`0.9421 ± 0.0008`**（40 维 0.9014），等预算经典 `PCA-165 + MLP` 为 **`0.9554 ± 0.0028`**（差距收窄到 **1.33 pt**，配对 CI `[+0.45, +2.20]`，经典仍显著领先且快约 1000 倍）。冻结随机酉 165 维探针 `0.9440` 与端到端 `0.9421` 打平，说明剩余差距来自全局齐次二次型缺少局部空间归纳偏置（1px 平移扰动达类内离散度的 1.20 倍）。详见 [`quantum-mnist/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) 第 10–13 节。
- **第八轮突破（`resolve_open_issues.py`，解决遗留问题与补齐基线）**：
  1. **解决 Hybrid CNN + VQC 训练崩溃**：定位崩溃根因是轻量 CNN 池化后跨样本特征标准差（`~0.003`）比线性层随机偏置小 50 倍且缺乏归一化；通过 `BatchNorm1d + Tanh*(π/2) + 近恒等初始化 N(0, 0.1²) + 32 维读出`（纯串联、**无跨量子层残差跳连**），`hyb_light_stab` 从 `0.5980 ± 0.1062` 提升至 **`0.7650 ± 0.0464`**（配对 `+16.70 ± 11.70 pt`，CI `[+2.18, +31.22]`，0/5 seed 崩溃），追平跳连基线 `nq_light`（`0.7440 ± 0.0651`）。
  2. **Pérez-Salinas 可训练数据重上传**：实现逐层可训练输入缩放与偏置 $\text{RY}(\omega_{l,i}\tilde{x}_i + \theta_{l,i})$，在 Z-only 读出下从朴素重上传的 `0.5220 ± 0.0652` 恢复到 **`0.6580 ± 0.0246`**（`+13.60 ± 7.41 pt`），配合 40 维读出达 **`0.7470 ± 0.0214`**（较单次上传 `a10_z` `+11.50 ± 4.65 pt`）。
  3. **补齐量子核（QSVC）第三条基线**：5-seed 实测 10-qubit 全局振幅核 `qkernel_amp` 达 **`0.9210 ± 0.0074`**，8-qubit 局部 9-patch 保真度核 `qkernel_patch` 达 **`0.9310 ± 0.0164`**，在 600 张小样本上显著领先同划分经典 `svm_rbf`（`0.8650 ± 0.0226`，`+6.60 pt`）与 `svm_poly2`（`0.8980 ± 0.0225`）。
  4. **局部空间归纳偏置（`PatchVQCClassifier`）**：将 $28\times 28$ 图像切为 $3\times 3$ 重叠 $14\times 14$ 局部 patch（stride 7），共享同一个 8-qubit 振幅编码电路（仅 48 个量子参数）并保留相对 patch 范数，5-seed 准确率达 **`0.8800 ± 0.0112`**，显著超越 10-qubit 全局 165 维 `amp_b165`（`0.8480 ± 0.0148`，`+3.20 ± 2.02 pt`）与同特征数 `ref_pca165`（`0.8410 ± 0.0124`，`+3.90 ± 1.71 pt`），与经典 `MLP-784`（`0.8640 ± 0.0065`）统计打平，且因状态向量从 $2^{10}$ 降为 $2^8$ 比 `amp_b165` **快 5.7 倍**。详见 [`quantum-mnist/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) 第 14 节。

### 2. `quantum-llm`（方法学审查与 5-Seed 因果上下文升级）
- **定位并证明无记忆 Bigram 墙**：初版 `TinyLM` 将 `(B, T, Q)` 展平为 `(B*T, Q)` 逐位置独立过变换层，`--seq-len 16` 完全未跨位置交互（有效上下文 $k=1$）。内置语料（1668 字符，$V=31$）的精确 $k=1$ 条件熵下界为 **`1.9701` nats（困惑度 `7.17`）**，且精确 Bigram Oracle 也会因 `' ' -> 't' -> 'h' -> 'e' -> ' '` 最高频环确定性陷入 `"A qubit can hold the the the..."` 死循环；而 $k=4$ 因果上下文的理论熵下界骤降至 **`0.1000` nats（困惑度 `1.11`）**。
- **修复混淆变量与 5-Seed 受控对照（`diagnose_llm.py`）**：修复了量子/经典使用不同 batch 随机种子（`seed` vs `seed+1`）、共享层初值不同、`iter_batches` 漏采末尾字符以及单步 batch loss 噪声，引入全语料确定性评估 `evaluate_lm` 与独立未见句验证集（`_HELDOUT_VAL_CORPUS`，382 字符）：
  - **单字符模式（$k=1$）**：修复 logit 收缩（`scale=5.0` + `tanh*π/2`）并将 4-qubit 读出从 4 维 `PauliZ` 扩至 48 维全两体 Pauli 基（`z+x+y+zzall+xxall+yyall`）后，`tok_q_rich48` 训练 loss 从 `2.3864 ± 0.0553`（ppl `10.89`）降至 **`2.0631 ± 0.0148`（ppl `7.87`）**，逼近 $k=1$ 理论下界 `1.9701` nats。
  - **因果上下文模式（$k=4$，`CausalContextEncoder` 严格左填充因果卷积）**：`caus_q_rich48`（4639 参数）将训练 loss 推进至 **`0.8271 ± 0.0717` nats（ppl `2.29 ± 0.16`）**、未见句验证 loss 达 **`1.4494 ± 0.1290` nats（ppl `4.29 ± 0.57`）**，显著优于无量子层消融 `caus_identity`（train `1.6549` / val `1.9394`）与等预算经典基线 `caus_classical`（4867 参数，train `1.1483 ± 0.0617` / val `1.6266 ± 0.1166`；验证集配对 CI `[-0.2582, -0.0962]` nats），并彻底跳出 `"the the the"` 贪心循环。详见 [`quantum-llm/docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md)。

### 3. 诚实边界与下一步
- **仍不宣称量子优势**：`quantum-mnist` 全量下经典仍领先 1.33 pt 且快约 1000 倍，`PatchVQC` 的提升主要来自经典 2D 滑动窗口局部性与二次型保真度特征；`quantum-llm` 的 48 维两体 Pauli 读出等价于 4-qubit 因果特征上的固定二阶多项式核展开，状态向量模拟比经典层慢约 40 倍。全部结果来自 `default.qubit` 解析模拟，不含真机状态制备与有限 shots 成本。
- **下一步**：在全量 MNIST 上实跑 `PatchVQCClassifier` 与 `qkernel_patch` Nyström 近似；全量扩到 $\ge 20$ seed + 多重比较校正；在 `quantum-llm` 中探索量子自注意力打分（Quantum Attention）与更大非重复语料；引入有限 shots（按 QWC 3 基分组测量）与退极化噪声基准。

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线与无量子层（`Identity`）消融对照、在**构造模型之前**固定随机种子，并在 README 与 `docs/REVIEW.md` 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。结论性差异需要多 seed 均值与置信区间支撑。两个实验仓库均为 MIT 许可。
