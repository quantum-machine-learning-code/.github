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
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | PCA 降维 + AngleEmbedding + StronglyEntanglingLayers 的 VQC，附真实 MNIST 实测指标与对比图 | 活跃 |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 玩具级量子增强字符 LM，附 loss / 困惑度 / 耗时对照与贪心生成样本 | 活跃 |
| `.github` | 组织 profile 与共享文档（本仓库） | — |

## 我们的原则

1. **不宣称量子优势。** 量子模型在这个规模下常常打平甚至落后于经典基线，量子与经典两组数字会一起公开 —— 两个仓库的实测结果都是如此。
2. **说清边界。** 状态向量模拟是 2ⁿ 内存墙，4–12 qubits 是笔记本舒适区；当前条件下不存在可行的量子 LLM。`quantum-llm` 只是沿用方向命名的**玩具级可微分层验证**，README 开头就写明了这一点，它不是能对话的大模型。
3. **可复现。** 固定种子、离线数据与内置语料、精确复现命令；单次运行的结果会明确标注「未做统计重复」。
4. **不需要真机。** 全部实验在经典模拟器上完成，无需 IBM Quantum 等平台账号或 token。

## 技术栈

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn / NumPy / Matplotlib；Python 3.11+；默认 `default.qubit` 状态向量模拟，解析求导，不接真机。两个实验仓库共享同一套核心实现，但各自保留所需模块，避免跨仓库依赖影响复现。

## 路线图

- 多 `--seed` 重复实验与统计检验、等参数预算对照
- amplitude encoding / 数据重上传、量子核 QSVC 作为第三条基线
- 噪声模型与有限 shots
- Lightning GPU 后端与更多 qubits
- 条件成熟时再接入真实量子硬件

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线、固定随机种子，并在 README 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。两个实验仓库均为 MIT 许可。
