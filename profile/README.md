# Quantum Machine Learning Code

> 公开的量子机器学习实验仓库组织。
> **Reproducible quantum–classical ML baselines — measured, not marketed.**

![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Simulator](https://img.shields.io/badge/backend-statevector%20simulator-lightgrey)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

## 我们在做什么

这里存放**可复现、可核对**的量子机器学习实验代码，聚焦量子计算与 AI 的交叉方向：把参数化量子电路放进经典深度学习训练循环，再和同规模经典模型放在同一套配置下比较。

| 方向 | 内容 |
| --- | --- |
| 量子视觉分类 | 变分量子分类器（VQC）做手写数字识别，与经典 MLP 对照 |
| 量子 + 语言模型 | 把参数化量子电路作为**一个可微分层**嵌入字符级自回归语言模型（玩具级验证） |
| 对照与评测 | 固定随机种子、等规模经典基线、准确率 / loss / 困惑度 / 参数量 / 耗时 |

## 仓库

| 仓库 | 说明 | 状态 |
| --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 上述两个方向的完整实现、实测指标、对比图与精确复现命令 | 活跃 |

## 我们的原则

1. **不宣称量子优势。** 量子模型在这个规模下常常打平甚至落后于经典基线，量子与经典两组数字会一起公开 —— 首个仓库的实测结果就是如此。
2. **说清边界。** 状态向量模拟是 2ⁿ 内存墙，4–12 qubits 是笔记本舒适区；当前条件下的「量子 LLM」只能是玩具级可微分层验证，**不是能对话的大模型**。
3. **可复现。** 固定种子、支持离线 MNIST 数据、给出精确复现命令；单次运行的结果会明确标注「未做统计重复」。
4. **不需要真机。** 全部实验在经典模拟器上完成，无需 IBM Quantum 等平台账号或 token。

## 技术栈

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn / NumPy / Matplotlib；Python 3.11+；默认 `default.qubit` 状态向量模拟，解析求导，不接真机。

## 路线图

- 多 `--seed` 重复实验与统计检验、等参数预算对照
- amplitude encoding / 数据重上传、量子核 QSVC 作为第三条基线
- 噪声模型与有限 shots
- Lightning GPU 后端与更多 qubits
- 条件成熟时再接入真实量子硬件

## 贡献

欢迎 issue 与 PR。新增实验请保留经典基线、固定随机种子，并在 README 里写清配置、边界与全部结果 —— 不要只报好看的那一组数字。
