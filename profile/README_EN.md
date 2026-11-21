# Quantum Machine Learning Code

English | [简体中文](README.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar-dark.png">
    <img src="https://raw.githubusercontent.com/quantum-machine-learning-code/.github/main/assets/quantum-avatar.png" alt="Quantum Machine Learning Code mark" width="120">
  </picture>
</p>

> Open organization for strictly controlled quantum machine learning benchmarks.
> **Reproducible quantum–classical ML baselines — measured, not marketed.**

![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

---

## Formal Experimental Reports (7 Standalone IMRaD Reports + QML Checklists)

| Repo | Formal Report (300-dpi Figures & Reproducibility Checklist) | Scope & Verified Headline Finding |
| --- | --- | --- |
| `quantum-mnist` | [**Report 01**: MNIST Three-Regime Benchmark](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/01_mnist_three_regimes_report.md) | Classical-image three-regime benchmark + finite shots: `vqc_amp` drops `−36.70 pt` at 200 shots; Regime II/III do not beat matched/dequantized classical baselines. |
| `quantum-mnist` | [**Report 02**: Cluster-Ising SPT QCNN](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/02_cluster_ising_qcnn_report.md) | Native 1D Cluster-Ising $n=8$ SPT classification: QCNN (`36q/0c/1 QWC`) reaches **`0.9967`** (`0.9633` at 200 shots), Holm-beating 1-body local observables. |
| `quantum-mnist` | [**Report 03**: Hardware Topology & Noise](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/reports/03_hardware_topology_noise_report.md) | 4-topology SABRE SWAP routing × `default.mixed` noise: `heavy_hex`/`linear_chain` collapse IQP kernel spread to `5.2%`/`3.4%` (`−11.40 pt`). |
| `quantum-llm` | [**Report 01**: Regime II L0 Confound Audit](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/01_regime2_l0_vqc_confound_report.md) | Resolves legacy `+0.3263` overfitting claim (`scale=5.0 + ni=True` alone adds `+0.1249`); matched Q ties 12-param `ClassicalTrig`. |
| `quantum-llm` | [**Report 02**: L1 Bloch-Sphere Channel Probe](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/02_l1_bloch_channel_probe_report.md) | Proves `RZ+Z/X` gradients vanish (`0.000e+00`) at $s=0, w=0$ while conjugate `RZ+Y` (`1.140e-01`) stays $O(1)$; lifts delayed recall `0.500 → 0.750/0.900`. |
| `quantum-llm` | [**Report 03**: MPS Born Machine & L2 Unitary SSM](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/03_regime3_born_and_unitary_ssm_report.md) | Periodic-4 / Dyck-1 confirm interference representation gains; true L2 `QuantumUnitarySSM` (0 MLP) hits **`1.000`** on $S_3$ OOD $T=16$. |
| `quantum-llm` | [**Report 04**: Quantum Self-Attention Benchmark](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/reports/04_quantum_attention_benchmark_report.md) | KV Recall ($T=6\to 10$): SWAP-Test equals classical overlap + $O(T^2 S)$ shot wall; Quixer LCU+QSVT (`946q/0c`) hits **`0.9910`** ($p_{\text{succ}}=0.395$). |

---

## Unified Three-Regime × Two-Modality Framework

| Regime | Definition | Vision / Quantum-State & Hardware Track ([`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)) | Sequence / Attention Track ([`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)) |
| --- | --- | --- | --- |
| **Regime I · Classical-on-Classical** | Exact dequantized surrogates, matched preprocessing & full-param baselines | Dequantized `poly2` kernel, 48-param patch filter, dual-BN CNN, 96-param trig net, local & full-state oracles | 12-param `ClassicalTrig`, matched `Tanh`, non-negative MPS, RNN, $SO(9)$ `OrthogonalSSM`, Dot/Overlap/Linear attention |
| **Regime II · Classical-Algo-on-Quantum** | Classical backbones with VQC / hybrid swaps under single-variable ablations | 10q amplitude VQC (incl. 200/1000 shots), 8q `PatchVQC`, stabilized dual-BN `HybridCNN+VQC` | L0 causal Conv1d + 4q VQC (char LM / mod-112 addition), L1 measurement-feedback RNN, `QuantumSwapTestAttention` (5q `CSWAP`) |
| **Regime III · Quantum-Native-on-Quantum** | Zero classical MLP crutches; entanglement, interference & coherent evolution | 1D Cluster-Ising SPT **Cong-et-al. QCNN**, 6q IQP kernel, 4q deep re-uploading Born classifier + 4-topology SABRE/`default.mixed` noise | **Autoregressive MPS Born Machines** (Periodic-4 / Dyck-1), **True L2 Unitary SSM** (POVM), **Coherent LCU+QSVT Quixer Attention** |

---

## Verified Benchmark Summary (5 Independent Seeds, Holm-Bonferroni Corrected)

### 1. [`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist) ([`three_regimes_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/three_regimes_metrics.json) · [`hardware_topology_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/hardware_topology_metrics.json))

| Track & Task | Representative Model (Regime / Params / QWC) | Verified Metric (Test Acc) | Controlled Verdict |
| --- | --- | --- | --- |
| **Track A · MNIST 10-Class** (600/200/400) | `patch_vqc_matched` (II · `48q/20,316c` · 3 QWC)<br>`vqc_amp_matched` (II · `60q/1,642c` · 3 QWC) | `0.8705 ± 0.0262`<br>Analytic `0.7760` / 200 shots `0.4090` / 1000 `0.6360` | `patch_vqc` ties 48-param `classical_patch_lowrank` (`0.8715`, `p_holm=1.0`), trails dequantized `poly2` (**`0.9155`**); `vqc_amp` collapses **`−36.70 pt`** at 200 shots (`p_holm=0.000336`); dual-BN `hybrid_noq` (`0.8310`) beats `hybrid_vqc` (`0.7600`). |
| **Track A · Zero-Wrapper Native** | `qnative_reupload_fidelity` (III · `96q/0c` · 1 QWC)<br>`qnative_iqp_kernel` (III · 6q 2-rep · 1 QWC) | `0.6175 ± 0.0307`<br>`0.7380 ± 0.0339` | Trainable without MLP crutches, tying 96-param `classical_trig_net` (`0.6165`, `p_holm=1.0`) and same-PCA-6 `poly2/rbf` (`0.7415 / 0.7545`); exposes kernel concentration at high $\alpha$ (`offdiag 0.875 → 0.019`). |
| **Track B · Cluster-Ising SPT** ($n=8$ PBC) | **`qcnn_spt` (III · Cong-et-al. QCNN · `36q/0c` · 1 QWC)** | Analytic **`0.9967 ± 0.0045`**<br>1000 shots **`0.9950`** / 200 **`0.9633`** | **Holm-significantly beats 1-body local baselines** (`1body_logreg 0.6917`, `+30.50 pt`, `p_holm=0.00678`; `1body_mlp 0.9167`, `+8.00 pt`, `p_holm=0.000235`), shot-resilient (`p_holm=0.198`), matching the $O(2^n)$ full-state oracle (`1.0000`) with 36 params. |
| **Hardware Topology & `default.mixed` Noise** | 4-topology SABRE routing (`max_err ≤ 2.776e-16`)<br>`iqp_kernel` 6q & `qcnn_spt` 8q ($p_{2q}=0.010$) | IQP `0.6560 → 0.5560` (`heavy_hex`) / **`0.5420`** (`linear_chain`)<br>QCNN `0.9967` / **`0.9950`** | `heavy_hex`/`linear_chain` inflate IQP physical 2Q gates `2.90×`/`3.40×` (QFT-8 `3.03×`/`3.63×`), collapsing IQP spread `std_ratio` to `0.0525`/`0.0345` (`linear_chain −11.40 ± 4.72 pt`, `p_holm=0.0342` ✓); QCNN analytic boundary holds `≥0.9950` (symmetric noise shrinks margin without sign flip). |

- **Historical archive pointers**: Full MNIST (60k/10k, [`full_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/full_metrics.json)): 165d VQC `0.9421 ± 0.0008` vs `PCA-165 + MLP` `0.9554 ± 0.0028` (`130,144` fitted PCA weights); Round-8 audit ([`audit_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/audit_metrics.json)): `qkernel_amp == plain_poly2_c0` (`0.9210`), `PatchLinear` (`0.8910`) ties `PatchVQC` (`0.8800`), `2/20` survive Bonferroni.

### 2. [`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm) ([`matched_task2_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/matched_task2_metrics.json) · [`quantum_native_seq_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_native_seq_metrics.json) · [`quantum_attention_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_attention_metrics.json))

Scope disclaimer: `quantum-llm` is **not** a conversational LLM — it is a controlled quantum sequence modeling and self-attention methodological benchmark.

| Track & Task | Representative Model (Regime / Params / Spec) | Verified Metric | Controlled Verdict |
| --- | --- | --- | --- |
| **Regime II · Char LM / Mod-112 / L1 Probe** | `caus_q_rich30` (`12q/3,646` = `0.33%` · 30-Pauli / **3 QWC**)<br>`train_addition --matched` v4 (`12q/10,230`) | Char Val `2.4804` / Gap `0.3812` (`31.6 s`)<br>Mod-112 `0.3251` vs `0.2767` (`p=0.311`) | Restoring `scale=5.0 + ni=True` alone inflates gap `+0.1249` (`p_holm=0.009705`); matched Q ties 12-param `ClassicalTrig` (`2.4834`, `p_holm=0.895`, **20.7×** slower) and `Tanh` (`2.4425`, `p_holm=0.289`). L1 $s=0,w=0$ `RZ+Y` (`1.140e-01`) fixes `RZ+Z/X` (`0.000e+00`) dead channel, lifting ID delayed recall `0.500 → 0.750 / 0.900`. |
| **Regime III Track 1 · MPS Born Machine** | **Periodic-4** complex $\chi=2$ (`24p`)<br>**Dyck-1** signed $\chi=6$ (`84p`) | OOD $T=14$ exact KL **`0.0949`**<br>Test-ID NLL **`0.4415`** / OOD valid **`0.856`** | **Destructive-interference separation**: on Periodic-4, beats 24p non-negative $\chi=3$ MPS (`0.3929`, `p_holm=0.0247`) and RNN (`0.5144`, `p_holm=0.00573`); on Dyck-1, beats 84p non-negative (`0.4732`, `p_holm=0.000382`) and RNN validity (`0.224`, `+63.2 pt`, `p_holm=0.0144`). |
| **Regime III Track 2 · L2 Unitary SSM** | **`l2_quantum_unitary_ssm`** (3q $d_H=8$, POVM, 0 MLP, `<2.4e-7`) | $S_3$ OOD $T=16$: **`1.000 ± 0.000`**<br>Delayed Recall OOD `pad=8`: **`1.000 ± 0.000`** | Beats RNN (`0.270`, `p_holm=0.000912`), L1 `RZ+Y` (`0.291`, `p_holm=3.32e-5`), and L0 (`0.156`, `p_holm=2.06e-7`), matching classical $SO(9)$ `OrthogonalSSM` (`1.000` / `0.950`). |
| **Track 3 · Quantum Self-Attention** (KV Recall $T=6\to 10$, `924–946`p) | Regime II `quantum_swap_test_attn` (5q `CSWAP`, `7.77e-16`)<br>**Regime III `quantum_quixer_lcu_qsvt` (`946q/0c`, `<3.7e-7`)** | SWAP analytic `1.0000` / $S=100..2000$ `0.997..0.999`<br>**Quixer ID/OOD `0.9910`**, $p_{\text{succ}}=0.423/0.395$ | SWAP-Test equals `ClassicalOverlap` (`1.0000`), but $S=100..2000$ brings `MAE 0.0737..0.0164` & `5,500..110,000` causal circuits; Quixer (0 Softmax/MLP) matches `LinearMixer` (`0.9950`) & `DotAttn` (`0.9710`) while beating uniform LCU (`0.8300`, `+0.1610`, `p=0.0209`). |

---

## Cross-Repo Engineering Contracts & Honest Physical Boundaries

1. **Aligned Canonical Observables & QWC Contract**: Both repositories share identical semantics for `make_observables(n, "z+x+y+zzall+xxall+yyall")` (homogeneous `ZZ/XX/YY`: 30 ops on 4q, 165 ops on 10q), verified by `count_qwc_groups()` at **3 QWC bases** (legacy mixed `ZX/XY/YZ` 48-Pauli in `quantum-llm` is opt-in and takes **10 QWC bases**).
2. **Single-Copy Quadraticity Ceiling & Sparse-Topology Noise Wall**: On classical images, single-copy `AmplitudeEmbedding` + any CPTP channel remains strictly a homogeneous quadratic form with $O(2^n)$ state prep; sparse topologies (`heavy_hex`/`linear_chain`) couple SWAP overhead with depolarizing noise to collapse all-to-all IQP kernels, whereas local native-state QCNNs remain resilient.
3. **Tensor-Network & FTQC / QRAM Boundaries**: 1D MPS scales classically as $O(TV\chi^3)$ (interference representation, not hardware speedup); SWAP-Test hits an $O(T^2 S)$ measurement wall; scaling `QuantumUnitarySSM` or **LCU+QSVT Quixer Attention** to exponential Hilbert spaces and long contexts requires fault-tolerant quantum computing (FTQC), $O(1/\sqrt{p_{\text{succ}}})$ amplitude amplification, and QRAM.
4. **Methodological Guardrails**: Seed before model construction; zero test peeking; explicit `Quantum / Classical / Preprocessing` parameter breakdowns; hard `assert_matched` checks; multi-seed mean ± SD with Holm-Bonferroni correction.

---

## Repositories, Tests & Quick Reproduction

| Repository | Scope & Formal Reports | Offline Tests | License |
| --- | --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | Three-regime vision/SPT benchmark + 4-topology SABRE & `default.mixed` noise · [3 Formal Reports](https://github.com/quantum-machine-learning-code/quantum-mnist/tree/main/docs/reports) · [`REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | `9 passed` (`pytest`) | MIT |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | Matched L0/L1, MPS Born Machine, L2 Unitary SSM, SWAP-Test/Quixer Attention · [4 Formal Reports](https://github.com/quantum-machine-learning-code/quantum-llm/tree/main/docs/reports) · [`REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) | `16 passed` (`pytest`) | MIT |
| `.github` | Organization profile and shared documentation (this repo) | `py_compile` CI | MIT |

```bash
# 1) quantum-mnist (Python 3.11+, offline CPU reproduction)
git clone https://github.com/quantum-machine-learning-code/quantum-mnist.git && cd quantum-mnist
python3 -m venv .venv && .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -v tests/smoke_test.py
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --archive docs/three_regimes_metrics.json
.venv/bin/python experiments/benchmark_hardware_topology.py --data-source idx --workers 20 --archive docs/hardware_topology_metrics.json

# 2) quantum-llm (offline CPU reproduction; set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1)
git clone https://github.com/quantum-machine-learning-code/quantum-llm.git && cd quantum-llm
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_attention.py --workers 4
```

> All primary scripts support `--quick` offline smoke runs writing to `results/` without overwriting `docs/`. Issues and PRs are welcome; please follow each repository's `CONTRIBUTING.md` and controlled-comparison contracts.
