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
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | VQC / 振幅编码 / 多可观测量读出 / 容量扫描与三轮 5 seed 统计对照，附真实 MNIST 实测指标、对比图与方法学审查 `docs/REVIEW.md` | 活跃 |
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

- **已完成（quantum-mnist）**：amplitude encoding 无压缩对照、输入维度与电路深度容量扫描、方法学审查、5 seed 重复实验 + 配对差值置信区间（`train_multiseed.py`）、改良分类头与输出缩放、**9 组 × 5 seed 瓶颈分解与量子读出信息探针**（`diagnose_ceiling.py`）、**11 组 × 5 seed 多可观测量 / 数据重上传 / 单独 lr 对照**（`break_readout.py`）
- **瓶颈已定位并部分解开**：0.664 约等于 10 维 `PauliZ` 读出的可分性上限 0.679（像素探针 0.855）；把读出扩到 40 维（Z/X/Y + 相邻 ZZ）后，5 seed 准确率 **0.6640 → 0.7720**（配对 +10.80 ± 1.35 pt，CI 不跨 0），冻结探针 **0.679 → 0.811**，回收约 13.2 pt 的可分信息。训练/数据预算翻倍只值 +1.3 / +4.0 pt，换经典 CNN 前端可到 0.863–0.934。详见 `docs/REVIEW.md` 第 8、9 节
- **已试过但无效 / 未解决**：给量子层单独 lr 没有消除混合前端的训练崩溃；朴素 data re-uploading 在 Z-only 读出下反而掉 11 pt（实现与文献原始设计不同，不外推）
- 量子层在混合架构里的初始化 / 深度 / lr 系统调参；按文献原始设计重做 data re-uploading
- 严格等总参数预算对照与量子核 QSVC 第三条基线；≥20 seed + 多重比较校正
- 噪声模型与有限 shots（多可观测量在真机上需分组测量）、Lightning GPU 后端与更多 qubits
- 条件成熟时再接入真实量子硬件

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线、在**构造模型之前**固定随机种子，并在 README 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。结论性差异需要多 seed 均值与方差支撑。两个实验仓库均为 MIT 许可。
