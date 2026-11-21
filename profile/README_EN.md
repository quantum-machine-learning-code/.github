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

## What We Do: Unified Three-Regime × Two-Modality Framework

This organization separates QML experiments cleanly by **algorithmic and physical mechanism** across two standalone, offline-reproducible repositories covering **Vision / Quantum-State Classification** ([`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist)) and **Sequence / Language Modeling** ([`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm)):

| Regime | Definition | Vision / Quantum-State Track (`quantum-mnist`) | Sequence / Language Track (`quantum-llm`) |
| --- | --- | --- | --- |
| **Regime I · Classical-on-Classical** | Exact dequantized surrogates, preprocessing-matched & full-parameter baselines | Exact dequantized `poly2` kernel, 48-param classical patch filter, dual-BN matched CNN, 96-param trig net, local-observable & full-state oracles | 12-param `ClassicalTrigBilinearLayer`, matched `Tanh`, non-negative MPS (no interference), classical RNN, and $SO(9)$ `OrthogonalSSM` |
| **Regime II · Classical-Algo-on-Quantum** | Small-parameter VQC bottlenecks embedded in classical backbones under single-variable ablations | 10q amplitude VQC (incl. 200/1000 finite shots), weight-shared 8q `PatchVQC`, stabilized dual-BN `HybridCNN+VQC` | L0 causal Conv1d + 4q VQC (independent char LM & zero-overlap mod-112 addition); L1 measurement-feedback quantum RNN (fixed `RZ+Y` / `RY+Z` channels) |
| **Regime III · Quantum-Native-on-Quantum** | Zero classical MLP crutches; coherent states, entanglement, and Born interference as the core mechanism | 1D Cluster-Ising $n=8$ SPT **Cong-et-al. QCNN** (36q / 0c), 6q entangling IQP kernel, 4q deep re-uploading Born classifier (96q / 0c) | **Autoregressive MPS Born Machines** (Periodic-4 complex & Dyck-1 signed interference), **True L2 Unitary SSM** (3q `QuantumUnitarySSM`, POVM readout) |

---

## Verified Benchmark Summary (5 Independent Seeds, Holm-Bonferroni Corrected)

### 1. [`quantum-mnist`](https://github.com/quantum-machine-learning-code/quantum-mnist) (Archive: [`three_regimes_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/three_regimes_metrics.json))

Protocol: independent disjoint stratified `Train / Val / Test` splits, zero test peeking, explicit `Quantum / Classical Wrapper / PCA Fitted` parameter accounting, programmatic QWC basis counts via `qml.pauli.group_observables`.

| Track & Task | Representative Model (Regime / Params / QWC) | Verified Metric (Test Acc) | Controlled Verdict |
| --- | --- | --- | --- |
| **Track A · MNIST 10-Class** (600/200/400) | `patch_vqc_matched` (II · `48q / 20,316c` · 3 QWC) | `0.8705 ± 0.0262` | Ties parameter-matched `classical_patch_lowrank` (`0.8715 ± 0.0157`, `−0.10 ± 1.82 pt`, `p_holm=1.0`; quantum share `0.24%`), well below exact dequantized `classical_poly2_unscaled` (**`0.9155 ± 0.0124`**). |
| **Track A · Hybrid & Finite Shots** | `vqc_amp_matched` (II · 10q 40d · `60q / 1,642c` · 3 QWC) | Analytic `0.7760 ± 0.0314`<br>200 shots `0.4090` / 1000 `0.6360` | Beats untrained `classical_random_quad` (`0.6515`, `p_holm=0.024`), but **collapses `−36.70 pt` at 200 shots** (`p_holm=0.000336`); with matched dual BN, `hybrid_noq` (`0.8310`) beats `hybrid_vqc` (`0.7600`). |
| **Track A · Zero-Wrapper Native Models** | `qnative_reupload_fidelity` (III · 4q L=4 · `96q / 0c` · 1 QWC)<br>`qnative_iqp_kernel` (III · 6q 2-rep · 1 QWC) | `0.6175 ± 0.0307`<br>`0.7380 ± 0.0339` | Trainable without MLP crutches, tying matched `classical_trig_net` (`0.6165`, `p_holm=1.0`) and same-PCA-6 `poly2/rbf` (`0.7415 / 0.7545`); exposes kernel concentration at high $\alpha$ (`offdiag 0.875 → 0.019`). |
| **Track B · 1D Cluster-Ising SPT Phases** ($n=8$ PBC, 120/60/120) | **`qcnn_spt` (III · Cong-et-al. QCNN · `36q / 0c` · 1 QWC)** | Analytic **`0.9967 ± 0.0045`**<br>1000 shots **`0.9950`** / 200 **`0.9633`** | **Holm-significantly beats local 1-body baselines** (`1body_logreg 0.6917`, `+30.50 pt`, `p_holm=0.00678`; `1body_mlp 0.9167`, `+8.00 pt`, `p_holm=0.000235`), shot-resilient (`p_holm=0.198`), matching the $O(2^n)$ full-state oracle (`1.0000`) with 36 params. |

- **Historical archive pointers**: Full MNIST (60k/10k, 3 seeds, [`full_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/full_metrics.json)): 165d VQC `0.9421 ± 0.0008` vs `PCA-165 + MLP` `0.9554 ± 0.0028` (including `130,144` fitted PCA weights); Round-8 audit ([`audit_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/audit_metrics.json)): `qkernel_amp` (`0.9210`) equals unscaled `poly2(coef0=0)` seed-for-seed, `PatchLinear` (`0.8910`) ties `PatchVQC` (`0.8800`), and `2/20` tests survive Bonferroni.

### 2. [`quantum-llm`](https://github.com/quantum-machine-learning-code/quantum-llm) (Archives: [`matched_task2_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/matched_task2_metrics.json) · [`quantum_native_seq_metrics.json`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/quantum_native_seq_metrics.json))

Scope disclaimer: `quantum-llm` is **not** a conversational LLM — it is a controlled quantum sequence modeling and methodological audit benchmark.

| Track & Task | Representative Model (Regime / Params / Spec) | Verified Metric | Controlled Verdict |
| --- | --- | --- | --- |
| **Regime II · Task 2 Char LM** (independent corpus, 13 asserted matched fields) | `caus_q_rich30` (II · `12q / 3,646` = `0.33%` · canonical 30-Pauli / **3 QWC**) | Val Loss `2.4804 ± 0.0291`<br>Gap `0.3812 ± 0.0354` (`31.6 s`) | **Legacy confound resolved**: restoring `scale=5.0 + ni=True` alone inflates the gap by `+0.1249` (`p_holm=0.009705`). Once matched, Quantum ties 12-param `ClassicalTrig` (Val `2.4834`, `1.5 s`, `Δ=-0.0030`, `p_holm=0.895`) at **20.7×** runtime cost, and does not significantly differ from `Tanh` (`2.4425`, `p_holm=0.289`). |
| **Regime II · Task 1 Mod-112 & L1 Probe** | `train_addition --matched` v4 (`12q / 10,230` = `0.12%`)<br>`state_channel_probe` (L1 at $s=0, w=0$) | Val Acc `0.3251` vs `0.2767` (`p=0.311`)<br>`RZ+Y` `1.140e-01` vs `RZ+Z/X` `0.000e+00` | No accuracy advantage on zero-overlap addition (driven by shared circular prior `0.1835` vs `0.0062`; §12 architecture claim withdrawn in §13). Fixing the L1 `RZ+Z` dead channel lifts deterministic delayed recall (ID `pad=4`) from `0.500` to `0.750` (`RZ+Y`) / `0.900` (`RY+Z`). |
| **Regime III Track 1 · MPS Born Machine** (exact normalization & ancestral sampling) | **Periodic-4**: complex $\chi=2$ (`24p`)<br>**Dyck-1**: signed $\chi=6$ (`84p`) | OOD $T=14$ exact KL **`0.0949`**<br>Test-ID NLL **`0.4415`** / OOD valid **`0.856`** | **Destructive-interference separation**: on Periodic-4, significantly beats 24-param non-negative $\chi=3$ MPS (`0.3929`, `p_holm=0.0247`) and RNN (`0.5144`, `p_holm=0.00573`); on Dyck-1, beats 84-param non-negative MPS (`0.4732`, `p_holm=0.000382`) and RNN OOD validity (`0.224`, `+63.2 pt`, `p_holm=0.0144`). |
| **Regime III Track 2 · True L2 Unitary SSM** | **`l2_quantum_unitary_ssm`** (3q $d_H=8$, POVM readout, 0 MLP, PennyLane `<2.4e-7`) | $S_3$ OOD $T=16$: **`1.000 ± 0.000`**<br>Delayed Recall OOD `pad=8`: **`1.000 ± 0.000`** | Decisively beats RNN (`0.270`, `p_holm=0.000912`), L1 `RZ+Y` (`0.291`, `p_holm=3.32e-5`), and L0 (`0.156`, `p_holm=2.06e-7`), while matching classical $SO(9)$ `OrthogonalSSM` (`1.000` / `0.950`). |

---

## Cross-Repo Engineering Contracts & Honest Physical Boundaries

1. **Aligned Canonical Observables & QWC Contract**: Both repositories now share identical semantics for `make_observables(n, "z+x+y+zzall+xxall+yyall")` (homogeneous `ZZ/XX/YY` pairs: 30 ops on 4q, 165 ops on 10q), programmatically verified via `count_qwc_groups()` to require **3 QWC bases** (legacy mixed `ZX/XY/YZ` 48-Pauli readout in `quantum-llm` is opt-in only and requires **10 QWC bases**).
2. **Single-Copy CPTP Quadraticity Ceiling on Classical Data**: On classical images, single-copy `AmplitudeEmbedding` followed by any CPTP channel (including QCNNs with mid-circuit pooling) yields expectations strictly restricted to homogeneous quadratic forms in $\hat{x}$ and requires $O(2^n)$ state preparation; QCNN gains apply only on native entangled quantum states (Track B).
3. **Tensor-Network & FTQC / QRAM Boundaries**: 1D MPS contraction scales classically as $O(TV\chi^3)$, demonstrating **representational destructive-interference separation** (Perron-Frobenius theorem), not exponential hardware speedup. Scaling L2 `QuantumUnitarySSM` to $2^n$ Hilbert spaces — or realizing **QSVT / block-encoded Quantum Attention** — requires fault-tolerant quantum computing (FTQC) and QRAM and cannot be claimed from classical simulators.
4. **Methodological Guardrails**: Seed before model construction; zero test-set tuning; explicit `Quantum / Classical Wrapper / Preprocessing Fitted` parameter breakdowns; hard `assert_matched` checks; multi-seed mean ± SD with Holm-Bonferroni correction on all conclusive claims.

---

## Repositories & Tech Stack

| Repository | Scope | Offline Tests | License |
| --- | --- | --- | --- |
| [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) | Controlled three-regime vision benchmark (MNIST 10-class + 1D Cluster-Ising SPT QCNN + finite shots) & methodology review [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-mnist/blob/main/docs/REVIEW.md) | `8 passed` (`pytest`) | MIT |
| [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) | Matched L0/L1 sequence benchmarks, state-channel probe, MPS Born Machine & L2 Unitary SSM; review in [`docs/REVIEW.md`](https://github.com/quantum-machine-learning-code/quantum-llm/blob/main/docs/REVIEW.md) | `15 passed` (`pytest`) | MIT |
| `.github` | Organization profile and shared documentation (this repo) | `py_compile` CI | MIT |

**Tech Stack**: Python 3.11+ · PennyLane 0.45 (`default.qubit` statevector & real `qml.set_shots`) · PyTorch 2.x · scikit-learn / SciPy / NumPy / Matplotlib. Everything runs offline on CPU with no quantum hardware accounts or tokens.

---

## Quick Reproduction

```bash
# 1) quantum-mnist: offline test suite (8 passed) + three-regime benchmark (MNIST + Cluster-Ising SPT)
git clone https://github.com/quantum-machine-learning-code/quantum-mnist.git && cd quantum-mnist
python3 -m venv .venv && .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -v tests/smoke_test.py
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --torch-threads 2 --archive docs/three_regimes_metrics.json

# 2) quantum-llm: offline contract suite (15 passed) + state probe + matched Task 2 + quantum-native seq
git clone https://github.com/quantum-machine-learning-code/quantum-llm.git && cd quantum-llm
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/state_channel_probe.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
```

> All primary scripts support `--quick` offline smoke runs writing to `results/` without overwriting `docs/`. Issues and PRs are welcome; please follow each repository's `CONTRIBUTING.md` and controlled-comparison contracts.
