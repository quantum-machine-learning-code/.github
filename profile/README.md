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

## 正式实验报告索引（7 篇独立 IMRaD 报告 + QML 可复现核查表）

| 仓库 | 正式报告（含 300 dpi 配图与核查表） | 核心议题与归档实测结论 |
| --- | --- | --- |
| `quantum-mnist` | [**报告 01**：MNIST 三范式基准](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/01_mnist_three_regimes_report.md) | 经典图像三范式对照 + 有限 shots：`vqc_amp` 200 shots 暴跌 `−36.70 pt`；范式 II/III 均不超越等参/去量子化经典基线。 |
| `quantum-mnist` | [**报告 02**：Cluster-Ising QCNN](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/02_cluster_ising_qcnn_report.md) | 原生 1D Cluster-Ising $n=8$ SPT 相分类：QCNN（36q/0c/1 QWC）达 **`0.9967`**（200 shots `0.9633`），显著胜单比特局域期望。 |
| `quantum-mnist` | [**报告 03**：硬件拓扑与去极化噪声](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/03_hardware_topology_noise_report.md) | 4 拓扑 SABRE SWAP 路由 × `default.mixed` 噪声：`heavy_hex`/`linear_chain` 令 IQP 核方差塌缩至 `5.2%`/`3.4%`（`−11.40 pt`）。 |
| `quantum-llm` | [**报告 01**：Regime II L0 混杂清算](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/01_regime2_l0_vqc_confound_report.md) | 证伪历史 `+0.3263` 过拟合归因（`scale=5.0 + ni=True` 单独拉大 `+0.1249`）；对齐后与 12 参 `ClassicalTrig` 持平。 |
| `quantum-llm` | [**报告 02**：L1 Bloch 球通道诊断](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/02_l1_bloch_channel_probe_report.md) | 证明 $s=0, w=0$ 时 `RZ+Z/X` 导数恒为 `0.000e+00`，唯共轭 `RZ+Y`（`1.140e-01`）非零；修复后延迟回忆由 `0.500 → 0.750/0.900`。 |
| `quantum-llm` | [**报告 03**：MPS 玻恩机与 L2 幺正 SSM](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/03_regime3_born_and_unitary_ssm_report.md) | Periodic-4 / Dyck-1 验证复/负振幅干涉表征优势；真正 L2 `QuantumUnitarySSM`（0 MLP）在 $S_3$ OOD $T=16$ 达 **`1.000`**。 |
| `quantum-llm` | [**报告 04**：量子自注意力三范式基准](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/04_quantum_attention_benchmark_report.md) | KV 回忆（$T=6\to 10$）：SWAP-Test 等价经典且受 $O(T^2 S)$ 测量墙限制；Quixer LCU+QSVT（`946q/0c`）达 **`0.9910`**（$p_{\text{succ}}=0.395$）。 |

---

## 三范式 × 双模态统一研究框架

| 范式 | 定义 | 视觉 / 量子态与硬件赛道（[`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)） | 序列 / 注意力赛道（[`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)） |
| --- | --- | --- | --- |
| **Regime I · 经典在经典上** | 精确去量子化替身、同预处理与全参基线 | `poly2` 去量子化核、48 参经典滤波、双 BN 等宽 CNN、96 参三角网、局域期望 & 全态 Oracle | 12 参 `ClassicalTrig`、对齐 `Tanh`、非负实数 MPS、RNN、$SO(9)$ `OrthogonalSSM`、经典 Dot/Overlap/Linear 注意力 |
| **Regime II · 经典移植到量子** | 经典主干嵌入 VQC / 混合替换，单变量消融 | 10q 振幅 VQC（含 200/1000 shots）、8q `PatchVQC`、双 BN `HybridCNN+VQC` | L0 因果卷积 + 4q VQC（字符 LM / 模 112 加法）、L1 测量反馈循环、`QuantumSwapTestAttention`（5q `CSWAP`） |
| **Regime III · 量子原生在量子上** | 零经典 MLP 拐杖，以纠缠/干涉/相干演化为核心 | 1D Cluster-Ising SPT **Cong-et-al. QCNN**、6q IQP 核、4q 深度重上传 Born 分类器 + 4 拓扑 SABRE/`default.mixed` 噪声 | **自回归 MPS 玻恩机**（Periodic-4 / Dyck-1）、**真正 L2 幺正 SSM**（POVM）、**相干 LCU+QSVT Quixer 注意力** |

---

## 核心实测结论一览（5 独立 Seed，Holm-Bonferroni 校正）

### 1. [`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)（[`three_regimes_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/three_regimes_metrics.json) · [`hardware_topology_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/hardware_topology_metrics.json)）

| 赛道与任务 | 代表模型（范式 / 参量 / QWC） | 关键实测指标（Test Acc） | 严格对照结论 |
| --- | --- | --- | --- |
| **Track A · MNIST 10 类** (600/200/400) | `patch_vqc_matched`（II · `48q/20,316c` · 3 QWC）<br>`vqc_amp_matched`（II · `60q/1,642c` · 3 QWC） | `0.8705 ± 0.0262`<br>解析 `0.7760` / 200 shots `0.4090` / 1000 `0.6360` | `patch_vqc` 与 48 参滤波 `classical_patch_lowrank`（`0.8715`）持平（`p_holm=1.0`），低于去量子化 `poly2`（**`0.9155`**）；`vqc_amp` 200 shots 暴跌 **`−36.70 pt`**（`p_holm=0.000336`）；双 BN `hybrid_noq`（`0.8310`）胜 `hybrid_vqc`（`0.7600`）。 |
| **Track A · 零封装原生** | `qnative_reupload_fidelity`（III · `96q/0c` · 1 QWC）<br>`qnative_iqp_kernel`（III · 6q 2-rep · 1 QWC） | `0.6175 ± 0.0307`<br>`0.7380 ± 0.0339` | 零 MLP 封装可收敛，与 96 参 `classical_trig_net`（`0.6165`, `p_holm=1.0`）及同 PCA-6 `poly2/rbf`（`0.7415 / 0.7545`）持平；高 $\alpha$ 下核集中（`offdiag 0.875 → 0.019`）。 |
| **Track B · Cluster-Ising SPT** ($n=8$ PBC) | **`qcnn_spt`（III · Cong-et-al. QCNN · `36q/0c` · 1 QWC）** | 解析 **`0.9967 ± 0.0045`**<br>1000 shots **`0.9950`** / 200 **`0.9633`** | **Holm 显著胜单比特局域期望**（`1body_logreg 0.6917`, `+30.50 pt`, `p_holm=0.00678`; `1body_mlp 0.9167`, `+8.00 pt`, `p_holm=0.000235`），抗采样噪声（`p_holm=0.198`），36 参逼近 $O(2^n)$ 全态 Oracle（`1.0000`）。 |
| **硬件拓扑路由 & `default.mixed` 噪声** | 4 拓扑 SABRE 路由（`max_err ≤ 2.776e-16`）<br>`iqp_kernel` 6q & `qcnn_spt` 8q（$p_{2q}=0.010$） | IQP `0.6560 → 0.5560` (`heavy_hex`) / **`0.5420`** (`linear_chain`)<br>QCNN `0.9967` / **`0.9950`** | `heavy_hex`/`linear_chain` 令 IQP 物理 2Q 门增至 `2.90×`/`3.40×`（QFT-8 达 `3.03×`/`3.63×`），IQP 核方差保留比 `std_ratio` 暴跌至 `0.0525`/`0.0345`（`linear_chain −11.40 ± 4.72 pt`, `p_holm=0.0342` ✓）；QCNN 解析决策保持 `≥0.9950`（对称信道仅缩窄极化幅度）。 |

- **历史归档核对**：全量 MNIST（60k/10k, [`full_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/full_metrics.json)）165d VQC `0.9421 ± 0.0008` vs `PCA-165 + MLP` `0.9554 ± 0.0028`（含 `130,144` 拟合 PCA 参数）；Round-8 审计（[`audit_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/audit_metrics.json)）确认 `qkernel_amp == plain_poly2_c0`（`0.9210`）、`PatchLinear`（`0.8910`）持平 `PatchVQC`（`0.8800`）、Bonferroni 后 `2/20` 显著。

### 2. [`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)（[`matched_task2_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/matched_task2_metrics.json) · [`quantum_native_seq_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_native_seq_metrics.json) · [`quantum_attention_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_attention_metrics.json)）

边界声明：`quantum-llm` **不是**可对话大语言模型，而是受控量子序列建模与自注意力方法学基准。

| 赛道与任务 | 代表模型（范式 / 参量 / 规格） | 关键实测指标 | 严格对照结论 |
| --- | --- | --- | --- |
| **Regime II · 字符 LM / 模加法 / L1 诊断** | `caus_q_rich30`（`12q/3,646` = `0.33%` · 30-Pauli / **3 QWC**）<br>`train_addition --matched` v4（`12q/10,230`） | 字符 Val `2.4804` / Gap `0.3812` (`31.6 s`)<br>加法 `0.3251` vs `0.2767` (`p=0.311`) | 恢复 `scale=5.0 + ni=True` 单独拉大间隙 `+0.1249`（`p_holm=0.009705`）；对齐后持平 12 参 `ClassicalTrig`（`2.4834`, `p_holm=0.895`, 慢 **20.7×**）与 `Tanh`（`2.4425`, `p_holm=0.289`）。L1 $s=0,w=0$ `RZ+Y`（`1.140e-01`）修复 `RZ+Z/X`（`0.000e+00`）死通道，ID 延迟回忆 `0.500 → 0.750 / 0.900`。 |
| **Regime III 轨道 1 · MPS 玻恩机** | **Periodic-4** 复振幅 $\chi=2$（`24p`）<br>**Dyck-1** 有符号 $\chi=6$（`84p`） | OOD $T=14$ 精确 KL **`0.0949`**<br>Test-ID NLL **`0.4415`** / OOD 合法 **`0.856`** | **相消干涉表征优势**：Periodic-4 显著胜 24p 非负 $\chi=3$ MPS（`0.3929`, `p_holm=0.0247`）与 RNN（`0.5144`, `p_holm=0.00573`）；Dyck-1 胜 84p 非负（`0.4732`, `p_holm=0.000382`）与 RNN 合法率（`0.224`, `+63.2 pt`, `p_holm=0.0144`）。 |
| **Regime III 轨道 2 · L2 幺正 SSM** | **`l2_quantum_unitary_ssm`**（3q $d_H=8$, POVM, 0 MLP, `<2.4e-7`） | $S_3$ OOD $T=16$：**`1.000 ± 0.000`**<br>延迟回忆 OOD `pad=8`：**`1.000 ± 0.000`** | 击败 RNN（`0.270`, `p_holm=0.000912`）、L1 `RZ+Y`（`0.291`, `p_holm=3.32e-5`）、L0（`0.156`, `p_holm=2.06e-7`），与同维经典 $SO(9)$ `OrthogonalSSM`（`1.000` / `0.950`）持平。 |
| **轨道 3 · 量子自注意力**（KV 回忆 $T=6\to 10$，参 `924–946`） | Regime II `quantum_swap_test_attn`（5q `CSWAP`, `7.77e-16`）<br>**Regime III `quantum_quixer_lcu_qsvt`（`946q/0c`, `<3.7e-7`）** | SWAP 解析 `1.0000` / $S=100..2000$ `0.997..0.999`<br>**Quixer ID/OOD `0.9910`**, $p_{\text{succ}}=0.423/0.395$ | SWAP-Test 解析等价 `ClassicalOverlap`（`1.0000`），但 $S=100..2000$ 带来 `MAE 0.0737..0.0164` 与完整因果 `5,500..110,000` 次电路开销；Quixer 零 Softmax/MLP 持平 `LinearMixer`（`0.9950`）与 `DotAttn`（`0.9710`），胜均匀 LCU 消融（`0.8300`, `+0.1610`, `p=0.0209`）。 |

---

## 跨仓库工程契约与诚实物理边界

1. **规范可观测量与 QWC 对齐**：两仓 `make_observables(n, "z+x+y+zzall+xxall+yyall")` 已完全一致（同基两体 `ZZ/XX/YY`：4q=30 维 / 10q=165 维），`count_qwc_groups()` 校验严格为 **3 QWC 基**（`quantum-llm` 旧版混合 `ZX/XY/YZ` 48 维为 opt-in，需 **10 QWC 基**）。
2. **经典数据单拷贝二次型与硬件路由噪声墙**：经典图像上单拷贝 `AmplitudeEmbedding` 接任意 CPTP 通道严格受限于齐次二次型且需 $O(2^n)$ 态制备；稀疏硬件拓扑（`heavy_hex`/`linear_chain`）将全连接线路的 SWAP 开销与去极化噪声耦合，直接导致 IQP 核指数集中，而原生量子态 QCNN 仅局部耦合因而更具鲁棒性。
3. **张量网络与 FTQC / QRAM 边界**：1D MPS 为经典多项式 $O(TV\chi^3)$ 的干涉表征优势而非硬件加速；SWAP-Test 受 $O(T^2 S)$ 测量墙约束；将 `QuantumUnitarySSM` 或 **LCU+QSVT Quixer 注意力**扩至指数维希尔伯特空间与长序列依赖容错量子计算（FTQC）、振幅放大 $O(1/\sqrt{p_{\text{succ}}})$ 与 QRAM，不可在经典模拟器上宣称实用加速。
4. **方法学铁律**：构造前固定种子、零测试集泄露、显式拆解 `量子/经典/预处理` 参数、`assert_matched` 拦截混杂、结论性结果报告多 seed 均值 ± 标准差与 Holm-Bonferroni 校正。

---

## 仓库、测试与快速复现

| 仓库 | 内容与独立报告 | 离线测试 | 许可 |
| --- | --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | 三范式视觉/SPT 基准 + 4 拓扑 SABRE & `default.mixed` 噪声 · [3 篇正式报告](https://github.com/quantum-machine-learning-code/quantum-mnist/tree/main/docs/reports) · [`REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | `9 passed` (`pytest`) | MIT |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | 对齐 L0/L1、MPS 玻恩机、L2 幺正 SSM、SWAP-Test/Quixer 注意力 · [4 篇正式报告](https://github.com/quantum-machine-learning-code/quantum-llm/tree/main/docs/reports) · [`REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) | `16 passed` (`pytest`) | MIT |
| `.github` | 组织主页与共享说明（本仓库） | `py_compile` CI | MIT |

```bash
# 1) quantum-mnist（Python 3.11+, 纯 CPU 离线复现）
git clone https://github.com/quantum-machine-learning-code/quantum-mnist.git && cd quantum-mnist
python3 -m venv .venv && .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -v tests/smoke_test.py
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --archive docs/three_regimes_metrics.json
.venv/bin/python experiments/benchmark_hardware_topology.py --data-source idx --workers 20 --archive docs/hardware_topology_metrics.json

# 2) quantum-llm（纯 CPU 离线复现，建议 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1）
git clone https://github.com/quantum-machine-learning-code/quantum-llm.git && cd quantum-llm
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_attention.py --workers 4
```

> 所有主脚本均支持 `--quick` 秒级离线冒烟（产物写 `results/`，不覆盖 `docs/`）。欢迎提 issue 与 PR，新增实验请遵循各仓库 `CONTRIBUTING.md` 与受控对照契约。
