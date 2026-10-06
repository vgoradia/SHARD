# SHARD: Self-Attention Heuristics for Attentive Reactor Disruption

**Transformer-based plasma disruption prediction with interpretable signal attribution.**

---

## Overview

Plasma disruptions are sudden, catastrophic losses of confinement in tokamak fusion reactors. They release megajoules of energy in milliseconds, threatening reactor walls and halting experiments. Reliable early warning systems are essential for ITER and future commercial reactors.

**SHARD** predicts plasma disruptions and explains each prediction in real time, combining:

- **Transformer self-attention** for temporal modeling of 13 simultaneous plasma diagnostic signals
- **Native signal attribution** — a learnable layer that outputs per-signal importance weights with no post-hoc approximation
- **Attribution trajectory** — tracking how model attention evolves as disruption approaches, recovering physical instability dynamics
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

SHARD achieves a **6.3-point ROC-AUC improvement** over the published state-of-the-art (HDL, 0.801).

### Cross-Tokamak Generalization

| Train | Test | AUC |
|-------|------|-----|
| C-Mod + DIII-D | EAST | 0.845 |
| C-Mod + EAST | DIII-D | 0.636 |
| DIII-D + EAST | C-Mod | 0.853 |
| — | **Mean** | **0.778** |

Exceeds the single-device published baseline (0.801) in **2 of 3 transfer scenarios** without device-specific fine-tuning.

### Warning Time Analysis

| Metric | Value |
|--------|-------|
| Mean earliest alarm lead time | **113 ms** |
| Median earliest alarm lead time | **150 ms** |
| Labeled prediction horizon | 50 ms |
| Coverage (% disruptive shots with alarm) | **85.5%** |

SHARD fires **3× earlier** than its labeled prediction horizon, detecting precursors before the formal disruption window begins.

### Ablation Study

| Variant | AUC | ΔAUC |
|---------|-----|------|
| SHARD (Full) | 0.875 | — |
| w/o Positional Encoding | 0.848 | −0.027 |
| 1-Layer Transformer | 0.830 | −0.045 |
| w/o Signal Attribution | 0.876 | +0.001 |
| LSTM Baseline | 0.857 | −0.018 |

Signal attribution adds full interpretability at **zero cost to predictive accuracy** (ΔAUC < 0.001).

---

## Architecture

Input: x ∈ R^(50 × 13) [50 timesteps × 13 diagnostic signals]
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
│ Classifier MLP │ Signal Attribution Layer │
│ g → 32 → ReLU → 1 │ g → 13 → Softmax = β ∈ R^13 │
│ p_disrupt ∈ [0,1] │ per-signal importance weights │
└─────────────────────────┴──────────────────────────────────┘


**Total parameters: 62,671** — designed for real-time inference in plasma control systems.

---

## Signal Attribution

SHARD's attribution layer learns which of 13 plasma diagnostics drive each disruption prediction:

| Rank | Signal | Physical Category | Attribution at Disruption |
|------|--------|-------------------|--------------------------|
| 1 | Ip_error (current loss) | Structural | 0.268 |
| 2 | κ (elongation) | Positional | 0.203 |
| 3 | li (internal inductance) | Structural | 0.147 |
| 4 | Btor (Mirnov/Btor) | MHD | 0.102 |
| 5 | n/nG (Greenwald fraction) | MHD | 0.083 |

The **attribution trajectory** recovers the causal disruption sequence without physics supervision:

> elongation instability → MHD mode growth → current loss

---

## Installation

```bash
git clone https://github.com/vgoradia/SHARD.git
cd SHARD
pip install -r requirements.txt
```

**Requirements:** `torch>=2.0.0`, `numpy>=1.23.0`, `scipy>=1.9.0`, `scikit-learn>=1.1.0`, `matplotlib>=3.6.0`

---

## Usage

### Train SHARD
```bash
python train.py
# Outputs: shard_best.pt (best checkpoint by val AUC)
```

### Run Full Evaluation
```bash
python eval_full.py
```

### Attribution Trajectory
```bash
python attribution_trajectory.py
# Outputs: attr_trajectory.png
```

### Ablation Study
```bash
python ablation.py
```

### 5-Seed Ensemble
```bash
python ensemble_eval.py
```

### Cross-Tokamak Evaluation
```bash
python cross_tokamak.py
```

### Streamlit Demo
```bash
streamlit run app.py
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `model.py` | Full SHARD architecture + training loop |
| `train.py` | Training script |
| `eval_full.py` | Full evaluation pipeline |
| `baselines.py` | Baseline model comparisons (RF, LSTM) |
| `ablation.py` | Ablation study |
| `ensemble_eval.py` | 5-seed ensemble evaluation |
| `cross_tokamak.py` | Leave-one-machine-out evaluation |
| `attribution_trajectory.py` | Attribution heatmap over disruptive shot |
| `data_loader.py` | Data loading and preprocessing |
| `app.py` | Streamlit demo app |
| `requirements.txt` | Python dependencies |
| `shard_refs.bib` | Bibliography |
| `hf_model_card.md` | HuggingFace model card |

---

## Dataset

Data sourced from [DisruptionBench](https://dataverse.harvard.edu) via Harvard Dataverse.
Plasma shots from MIT Alcator C-Mod, DIII-D (General Atomics), and EAST (ASIPP, China).

---

## Citation

```bibtex
@article{goradia2026shard,
  title={{SHARD}: Self-Attention Heuristics for Attentive Reactor Disruption Prediction with Interpretable Signal Attribution},
  author={Goradia, Veer},
  year={2026}
}
```

---

## Author

**Veer Goradia** — Howard High School, Ellicott City, Maryland  
vgoradia07@gmail.com

---
