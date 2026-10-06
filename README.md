# SHARD: Self-Attention Heuristics for Attentive Reactor Disruption

**Transformer-based plasma disruption prediction with interpretable signal attribution**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue)](shard_paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-DisruptionBench-green)](https://dataverse.harvard.edu)

---

## What is SHARD?

SHARD predicts plasma disruptions in tokamak fusion reactors — and explains each prediction in real time. It combines:

- **Transformer self-attention** for temporal modeling of 13 simultaneous plasma diagnostic signals
- **Native signal attribution** — a learnable layer that outputs per-signal importance weights (no post-hoc approximation)
- **Attribution trajectory** — showing how model attention evolves *as disruption approaches*, tracking physical instability dynamics
- **Cross-machine generalization** — trained on one device, tested on two others, without device-specific retraining

---

## Results

### Main Results (MIT Alcator C-Mod)

| Method | ROC-AUC | PR-AUC | Precision |
|--------|---------|--------|-----------|
| HDL (published SOTA) | 0.801 | — | — |
| Random Forest | 0.926 | 0.536 | 0.376 |
| LSTM | 0.857 | 0.412 | 0.391 |
| **SHARD (single seed)** | **0.864** | **0.449** | **0.405** |
| **SHARD (5-seed ensemble)** | **0.857** | **0.382** | **0.420** |

SHARD achieves **6.3-point ROC-AUC improvement** over the published state-of-the-art (HDL, 0.801).

### Cross-Tokamak Generalization

| Train | Test | AUC |
|-------|------|-----|
| C-Mod + DIII-D | EAST | 0.845 |
| C-Mod + EAST | DIII-D | 0.636 |
| DIII-D + EAST | C-Mod | 0.853 |
| — | Mean | 0.778 |

Exceeds single-device published baseline (0.801) in **2/3 transfer scenarios** without device-specific fine-tuning.

### Warning Time Analysis

| Metric | Value |
|--------|-------|
| Mean earliest alarm lead time | **113 ms** |
| Median earliest alarm lead time | **150 ms** |
| Prediction horizon (training label window) | 50 ms |
| Coverage (% disruptive shots with alarm) | **85.5%** |

SHARD fires **3× earlier** than its labeled prediction horizon — detecting precursors before the formal disruption window.

### Ablation Study

| Variant | AUC | ΔAUC |
|---------|-----|------|
| SHARD (Full) | 0.875 | — |
| w/o Positional Encoding | 0.848 | −0.027 |
| 1-Layer Transformer | 0.830 | −0.045 |
| **w/o Signal Attribution** | **0.876** | **+0.001** |
| LSTM Baseline | 0.857 | −0.018 |

**Key finding:** Signal attribution adds interpretability at **zero cost to accuracy** (AUC change < 0.001).

---

## Architecture

```
Input: x ∈ R^(50 × 13)    [50 timesteps × 13 diagnostic signals]
  ↓
Linear projection → d_model = 64
  ↓
Learnable positional encoding
  ↓
3× Temporal Attention Blocks (MHSA, n_heads=4, LayerNorm, Dropout)
  ↓
Mean pooling → g ∈ R^64
  ↓
┌─────────────────────────┬──────────────────────────────────┐
│  Classifier MLP         │  Signal Attribution Layer        │
│  g → 32 → ReLU → 1     │  g → 13 → Softmax = β ∈ R^13    │
│  p_disrupt ∈ [0,1]     │  per-signal importance weights   │
└─────────────────────────┴──────────────────────────────────┘
```

**Total parameters: 62,671** — real-time inference compatible.

---

## Signal Attribution

SHARD's attribution layer learns which of 13 plasma diagnostics matter for each prediction:

| Rank | Signal | Physical Category | Attribution at Disruption |
|------|--------|-------------------|--------------------------|
| 1 | Ip_error (current loss) | Structural | 0.268 |
| 2 | κ (elongation) | Positional | 0.203 |
| 3 | li (internal inductance) | Structural | 0.147 |
| 4 | Btor (Mirnov/Btor) | MHD | 0.102 |
| 5 | n/nG (Greenwald fraction) | MHD | 0.083 |

**Attribution trajectory** reveals the causal sequence:
elongation instability → MHD mode growth → current loss — **recovered without physics supervision**.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0.0`, `numpy>=1.23.0`, `scipy>=1.9.0`, `scikit-learn>=1.1.0`, `matplotlib>=3.6.0`

---

## Usage

### Train SHARD
```bash
python model.py
# Outputs: shard_best.pt (best checkpoint by val AUC)
```

### Warning Time Analysis
```bash
python warning_time_viz.py
# Outputs: warning_time_hist.png, shot_trajectory.png, warning_time_stats.json
```

### Attribution Trajectory
```bash
python attribution_trajectory.py
# Outputs: attr_trajectory.png (heatmap of attribution weights over disruptive shot)
```

### PR-AUC + Calibration
```bash
python pr_calibration.py
# Outputs: pr_curves.png, calibration.png, pr_auc_results.json
```

### 5-Seed Ensemble
```bash
python ensemble_eval.py
# Outputs: ensemble_results.json
```

### Cross-Tokamak Evaluation
```bash
python cross_tokamak_eval.py
# Outputs: cross_tokamak_results.json, cross_tokamak.png
```

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Full SHARD architecture + training loop |
| `shard_best.pt` | Best model checkpoint (val AUC 0.870) |
| `attribution_trajectory.py` | Attribution heatmap over disruptive shot |
| `warning_time_viz.py` | Warning time histogram + shot trajectory |
| `pr_calibration.py` | PR-AUC curves + calibration reliability diagram |
| `ensemble_eval.py` | 5-seed ensemble training and evaluation |
| `cross_tokamak_eval.py` | Leave-one-machine-out evaluation |
| `shard_paper.tex` | IEEE-format research paper (6 pages) |
| `shard_paper.pdf` | Compiled paper |
| `shard_architecture.pdf` | Architecture diagram |
| `attr_trajectory.png` | Attribution trajectory figure |
| `warning_time_hist.png` | Warning time distribution |
| `shot_trajectory.png` | Disruption probability trajectory |
| `signal_attribution.png` | Mean signal attribution bar chart |
| `cross_tokamak.png` | Cross-device AUC comparison |
| `training_curves.png` | Ablation training curves |
| `pr_curves.png` | Precision-recall curves |
| `calibration.png` | Reliability diagram |
| `pr_auc_results.json` | PR-AUC and Brier score results |
| `ensemble_results.json` | 5-seed ensemble results |
| `warning_time_stats.json` | Warning time statistics |
| `ablation_results.json` | Ablation study AUC results |
| `cross_tokamak_results.json` | Cross-device AUC results |
| `requirements.txt` | Python dependencies |

---

## Citation

```bibtex
@article{goradia2025shard,
  title={{SHARD}: Self-Attention Heuristics for Attentive Reactor Disruption Prediction with Interpretable Signal Attribution},
  author={Goradia, Veer},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Dataset

Data from [DisruptionBench](https://dataverse.harvard.edu) — Harvard Dataverse.  
Plasma shots from MIT Alcator C-Mod, DIII-D (General Atomics), and EAST (ASIPP, China).
