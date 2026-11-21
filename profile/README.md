# Quantum Machine Learning Code

[English](README_EN.md) | 简体中文

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar-dark.png">
    <img src="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar.png" alt="Quantum Machine Learning Code mark" width="120">
  </picture>
</p>

> 公开的量子机器学习受控基准组织。
> **Reproducible quantum–classical ML baselines — measured, not marketed.**

![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

---

## 我们在做什么：三范式 × 双模态统一框架

本组织将量子机器学习实验按**算法与物理机制**清晰划分为三类范式，在**视觉/量子态分类**（[`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)）与**序列/语言建模**（[`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)）两条独立仓库线上执行严格受控、可离线复现的多 seed 对比：

| 范式 | 定义 | 视觉 / 量子态赛道（`quantum-mnist`） | 序列 / 语言赛道（`quantum-llm`） |
| --- | --- | --- | --- |
| **Regime I · 经典算法在经典上** | 精确去量子化替身、同预处理与全量参数对齐基线 | `poly2` 精确去量子化核、48 参经典滤波、双 BN 等宽 CNN、96 参三角网、局域观测量 & 全态 Oracle | 12 参 `ClassicalTrigBilinearLayer`、对齐 `Tanh`、非负实数 MPS（禁用干涉）、经典 RNN 与 $SO(9)$ `OrthogonalSSM` |
| **Regime II · 经典算法移植到量子** | 将小参数 VQC 瓶颈嵌入经典主干，严格单变量消融 | 10q 振幅 VQC（含 200/1000 有限 shots）、权重共享 8q `PatchVQC`、双 BN 稳定化 `HybridCNN+VQC` | L0 因果卷积 + 4q VQC（独立字符 LM 与零重叠模 112 加法）；L1 测量反馈量子循环（修复 `RZ+Y` / `RY+Z` 状态通道） |
| **Regime III · 量子原生在量子上** | 零经典 MLP 拐杖，以相干态/纠缠/玻恩干涉为核心机制 | 1D Cluster-Ising $n=8$ SPT 相 **Cong-et-al. QCNN**（36q / 0c）、6q 纠缠 IQP 核、4q 深度重上传纯 Born 分类器（96q / 0c） | **自回归 MPS 玻恩机**（Periodic-4 复振幅与 Dyck-1 有符号干涉）、**真正 L2 幺正状态空间模型**（3q `QuantumUnitarySSM`，POVM 读出） |

---

## 核心实测结论一览（全部 5 独立 Seed，Holm-Bonferroni 校正）

### 1. [`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)（完整数据：[`three_regimes_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/three_regimes_metrics.json)）

统一协议：独立分层互不相交 `Train / Val / Test` 切分、零测试集泄露、显式拆解 `量子 / 经典封装 / PCA 拟合` 参数、`qml.pauli.group_observables` 实测 QWC 基数。

| 赛道与任务 | 代表模型（范式 / 参量 / QWC 基） | 关键实测指标（Test Acc） | 严格对照结论 |
| --- | --- | --- | --- |
| **Track A · MNIST 10 类** (600/200/400) | `patch_vqc_matched`（II · `48q / 20,316c` · 3 QWC） | `0.8705 ± 0.0262` | 与等参 `classical_patch_lowrank`（`0.8715 ± 0.0157`）持平（`−0.10 ± 1.82 pt`，`p_holm=1.0`，量子仅占 `0.24%`）；远低于精确去量子化 `classical_poly2_unscaled`（**`0.9155 ± 0.0124`**）。 |
| **Track A · 混合与有限 Shots** | `vqc_amp_matched`（II · 10q 40d · `60q / 1,642c` · 3 QWC） | 解析 `0.7760 ± 0.0314`<br>200 shots `0.4090` / 1000 `0.6360` | 高于未训练 `classical_random_quad`（`0.6515`，`p_holm=0.024`），但 **200 shots 下暴跌 `−36.70 pt`**（`p_holm=0.000336`）；双 BN 对齐后 `hybrid_noq`（`0.8310`）高于 `hybrid_vqc`（`0.7600`）。 |
| **Track A · 零封装原生模型** | `qnative_reupload_fidelity`（III · 4q L=4 · `96q / 0c` · 1 QWC）<br>`qnative_iqp_kernel`（III · 6q 2-rep · 1 QWC） | `0.6175 ± 0.0307`<br>`0.7380 ± 0.0339` | 零 MLP 封装可独立收敛，但与等参 `classical_trig_net`（`0.6165`，`p_holm=1.0`）及同 PCA-6 `poly2/rbf`（`0.7415 / 0.7545`）持平；高 $\alpha$ 下实测核集中（`offdiag 0.875 → 0.019`）。 |
| **Track B · 1D Cluster-Ising SPT 相** ($n=8$ PBC，120/60/120) | **`qcnn_spt`（III · Cong-et-al. QCNN · `36q / 0c` · 1 QWC）** | 解析 **`0.9967 ± 0.0045`**<br>1000 shots **`0.9950`** / 200 **`0.9633`** | **Holm 显著超越局域单比特基线**（`1body_logreg 0.6917`，`+30.50 pt`，`p_holm=0.00678`；`1body_mlp 0.9167`，`+8.00 pt`，`p_holm=0.000235`），无显著采样退化（`p_holm=0.198`），以 36 参逼近 $O(2^n)$ 全态 Oracle（`1.0000`）。 |

- **历史归档核对**：全量 MNIST（60k/10k，3 seeds，[`full_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/full_metrics.json)）165d VQC `0.9421 ± 0.0008` vs `PCA-165 + MLP` `0.9554 ± 0.0028`（含 `130,144` 拟合 PCA 权重）；Round-8 审计（[`audit_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/audit_metrics.json)）确认 `qkernel_amp`（`0.9210`）与未标准化 `poly2(coef0=0)` 逐 seed 恒等，`PatchLinear`（`0.8910`）与 `PatchVQC`（`0.8800`）打平，Bonferroni 后仅 `2/20` 显著。

### 2. [`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)（数据：[`matched_task2_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/matched_task2_metrics.json) · [`quantum_native_seq_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_native_seq_metrics.json)）

边界声明：`quantum-llm` **不是**可对话大语言模型，而是受控量子序列建模与方法学审查基准。

| 赛道与任务 | 代表模型（范式 / 参量 / 规格） | 关键实测指标 | 严格对照结论 |
| --- | --- | --- | --- |
| **Regime II · 课题二字符 LM**（独立语料，13 项硬断言对齐） | `caus_q_rich30`（II · `12q / 3,646` = `0.33%` · 规范 30-Pauli / **3 QWC**） | Val Loss `2.4804 ± 0.0291`<br>Gap `0.3812 ± 0.0354`（`31.6 s`） | **证伪旧混杂**：恢复历史 `scale=5.0 + ni=True` 单独把间隙拉大 `+0.1249`（`p_holm=0.009705`）。对齐后与 12 参 `ClassicalTrig`（Val `2.4834`，`1.5 s`）无差异（`Δ=-0.0030`，`p_holm=0.895`）却慢 **20.7×**；与 `Tanh`（`2.4425`）亦无显著差（`p_holm=0.289`）。 |
| **Regime II · 课题一模 112 加法 & L1 诊断** | `train_addition --matched` v4（`12q / 10,230` = `0.12%`）<br>`state_channel_probe`（L1 $s=0,w=0$） | Val Acc `0.3251` vs `0.2767`（`p=0.311`）<br>`RZ+Y` `1.140e-01` vs `RZ+Z/X` `0.000e+00` | 模加法零重叠泛化无准确率优势（由共享循环先验驱动，`0.1835` vs `0.0062`；§12 架构归因已于 §13 撤回）。修复 L1 `RZ+Z` 死通道后，确定性延迟回忆（ID `pad=4`）由 `0.500` 升至 `0.750`（`RZ+Y`）/ `0.900`（`RY+Z`）。 |
| **Regime III 轨道 1 · MPS 玻恩机**（精确归一与祖先采样） | **Periodic-4**：复振幅 $\chi=2$（`24p`）<br>**Dyck-1**：有符号 $\chi=6$（`84p`） | OOD $T=14$ 精确 KL **`0.0949`**<br>Test-ID NLL **`0.4415`** / OOD 合法 **`0.856`** | **相消干涉表征优势**：Periodic-4 显著优于等 24 参非负 $\chi=3$ MPS（`0.3929`，`p_holm=0.0247`）与 RNN（`0.5144`，`p_holm=0.00573`）；Dyck-1 显著优于等 84 参非负 MPS（`0.4732`，`p_holm=0.000382`）与 RNN 合法率（`0.224`，`+63.2 pt`，`p_holm=0.0144`）。 |
| **Regime III 轨道 2 · 真正 L2 幺正 SSM** | **`l2_quantum_unitary_ssm`**（3q $d_H=8$，POVM 读出，0 MLP，PennyLane 校验 `<2.4e-7`） | $S_3$ OOD $T=16$：**`1.000 ± 0.000`**<br>延迟回忆 OOD `pad=8`：**`1.000 ± 0.000`** | 显著击败 RNN（`0.270`，`p_holm=0.000912`）、L1 `RZ+Y`（`0.291`，`p_holm=3.32e-5`）与 L0（`0.156`，`p_holm=2.06e-7`），与同维经典 $SO(9)$ `OrthogonalSSM`（`1.000` / `0.950`）持平。 |

---

## 跨仓库工程契约与诚实物理边界

1. **统一可观测量与 QWC 契约**：两个仓库的规范规格 `make_observables(n, "z+x+y+zzall+xxall+yyall")` 现已完全对齐（纯同基两体 `ZZ/XX/YY`，4q = 30 维 / 10q = 165 维），经 `count_qwc_groups()` 校验严格只需 **3 个 QWC 测量基**（`quantum-llm` 旧版含混合 `ZX/XY/YZ` 的 48 维读出改为显式 opt-in，实测需 **10 个 QWC 基**）。
2. **经典图像单拷贝二次型天花板**：在经典数据上，单拷贝 `AmplitudeEmbedding` 接任意 CPTP 量子通道（含带中间测量的 QCNN）期望值严格受限于归一化输入的齐次二次型，且真机需 $O(2^n)$ 态制备；QCNN 的优势只在原生纠缠量子态（Track B）上成立。
3. **张量网络与 FTQC / QRAM 边界**：1D MPS 玻恩机经典精确收缩为 $O(TV\chi^3)$，证明的是**复/负振幅相消干涉的表征优势**（Perron-Frobenius 分离），而非指数硬件加速；将 L2 `QuantumUnitarySSM` 扩至指数维 $2^n$ 或实现 **QSVT / 块编码量子自注意力**均需要容错量子硬件（FTQC）与 QRAM，不能在当前模拟器上宣称量子加速。
4. **方法学铁律**：固定种子在构造模型之前、严禁测试集选超参、显式分离 `量子 / 经典封装 / 预处理拟合` 参数、`assert_matched` 拦截多变量混杂、所有结论性比较报告多 seed 均值 ± 标准差与 Holm-Bonferroni 校正。

---

## 仓库与技术栈

| 仓库 | 内容 | 离线测试 | 许可 |
| --- | --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 三范式受控视觉基准（MNIST 10 类 + 1D Cluster-Ising SPT QCNN + 有限 shots）与完整方法学审查 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | `8 passed` (`pytest`) | MIT |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 严格对齐 L0/L1 序列基准、状态通道解析诊断、MPS 玻恩机与 L2 幺正 SSM，审查见 [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) | `15 passed` (`pytest`) | MIT |
| `.github` | 组织主页与共享说明（本仓库） | `py_compile` CI | MIT |

**技术栈**：Python 3.11+ · PennyLane 0.45（`default.qubit` 状态向量与真实 `qml.set_shots`）· PyTorch 2.x · scikit-learn / SciPy / NumPy / Matplotlib。全部实验纯 CPU 离线可复现，无需真机账号或 token。

---

## 快速复现

```bash
# 1) quantum-mnist：离线回归测试（8 passed）+ 三范式受控基准（MNIST + Cluster-Ising SPT）
git clone https://github.com/quantum-machine-learning-code/quantum-mnist.git && cd quantum-mnist
python3 -m venv .venv && .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -v tests/smoke_test.py
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --torch-threads 2 --archive docs/three_regimes_metrics.json

# 2) quantum-llm：离线契约测试（15 passed）+ 状态通道诊断 + 对齐课题二 + 量子原生序列基准
git clone https://github.com/quantum-machine-learning-code/quantum-llm.git && cd quantum-llm
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/state_channel_probe.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
```

> 所有主脚本均支持 `--quick` 秒级离线冒烟（产物写入 `results/`，不覆盖 `docs/` 归档）。欢迎提 issue 与 PR，新增实验请遵循各仓库 `CONTRIBUTING.md` 与受控对照契约。
