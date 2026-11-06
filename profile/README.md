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
| 量子视觉分类 | [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 变分量子分类器（VQC）做手写数字识别，与经典 MLP 对照 |
| 量子 + 语言模型 | [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 参数化量子电路作为**一个可微分层**嵌入字符级自回归语言模型（玩具级验证） |

## 仓库

| 仓库 | 说明 | 状态 |
| --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | VQC / 振幅编码 / 容量扫描与多 seed 统计对照，附真实 MNIST 实测指标、对比图与方法学审查 `docs/REVIEW.md` | 活跃 |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 玩具级量子增强字符 LM，附 loss / 困惑度 / 耗时对照与贪心生成样本 | 活跃 |
| `.github` | 组织 profile 与共享文档（本仓库） | — |

## 我们的原则

1. **不宣称量子优势。** 量子模型在这个规模下常常打平甚至落后于经典基线，量子与经典两组数字会一起公开 —— 两个仓库的实测结果都是如此。
2. **说清边界。** 状态向量模拟是 2ⁿ 内存墙，4–12 qubits 是笔记本舒适区；当前条件下不存在可行的量子 LLM。`quantum-llm` 只是沿用方向命名的**玩具级可微分层验证**，README 开头就写明了这一点，它不是能对话的大模型。
3. **可复现，且承认错误。** 固定种子、离线数据、精确复现命令；结论性数字带多 seed 均值与标准差。`quantum-mnist` 在 2026-11-06 做过一次方法学审查：发现种子 bug、撤回了一个基于单次运行的「+15.5 pt」结论，并把问题清单、实测证据与参考实现对照公开写在 `docs/REVIEW.md` 里。
4. **不需要真机。** 全部实验在经典模拟器上完成，无需 IBM Quantum 等平台账号或 token。

## 技术栈

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn / NumPy / Matplotlib；Python 3.11+；默认 `default.qubit` 状态向量模拟，解析求导，不接真机。两个实验仓库共享同一套核心实现，但各自保留所需模块，避免跨仓库依赖影响复现。

## 路线图

- **已完成（quantum-mnist）**：amplitude encoding 无压缩对照、输入维度与电路深度容量扫描、方法学审查、5 seed 重复实验 + 配对差值置信区间（`experiments/train_multiseed.py`）、改良分类头与输出缩放、**9 组 × 5 seed 的瓶颈分解与量子读出信息探针**（`experiments/diagnose_ceiling.py`，轻量/强特征两种混合 CNN 前端）
- **已量出的瓶颈**：最佳 VQC 的 0.664 基本等于其 10 维 `PauliZ` 读出的可分性上限 0.679（原始像素探针 0.855）；训练/数据预算翻倍只值 +1.3 / +4.0 pt，换经典 CNN 特征前端可到 0.863–0.934。详见 `docs/REVIEW.md` 第 8 节
- 等参数预算对照与量子核 QSVC 第三条基线
- 数据重上传、增加读出可观测量、给量子层单独调参（轻量前端有 1/5 seed 训练崩溃）
- 噪声模型与有限 shots
- Lightning GPU 后端与更多 qubits
- 条件成熟时再接入真实量子硬件

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线、在**构造模型之前**固定随机种子，并在 README 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。结论性差异需要多 seed 均值与方差支撑。两个实验仓库均为 MIT 许可。
