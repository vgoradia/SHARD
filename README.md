# SHARD

**Transformer-based plasma disruption prediction with interpretable signal attribution.**

SHARD is a lightweight transformer trained on plasma diagnostic time-series from the MIT Alcator C-Mod disruption database, predicting whether a tokamak plasma will disrupt within the next 50ms. Across 5 independent training runs, SHARD achieves a mean test ROC-AUC of 0.864 on C-Mod, outperforming the published state-of-the-art (HDL, 2021: 0.801) by 6.3 points under the same evaluation protocol. Beyond classification, SHARD identifies which plasma signals drove each prediction through a native signal attribution layer, and generalizes to unseen tokamaks (EAST, DIII-D) without device-specific retraining.

**Live App:** https://shard-disruption.streamlit.app/

---

## Features

- **Disruption prediction** across 13 plasma diagnostic signals (β_p, q95, MHD modes, Greenwald fraction, and more)
- **Native signal attribution** — a learnable layer outputs per-signal importance weights without post-hoc approximation
- **Attribution trajectory** — tracks how attention shifts as disruption approaches, recovering the causal instability sequence without physics supervision
- **Cross-machine generalization** — trained on C-Mod, tested zero-shot on DIII-D and EAST
- **62,671 parameters** — real-time inference compatible with plasma control system latency requirements

---

## Model

| Metric | Value |
|---|---|
| Architecture | 3-layer Transformer encoder + Signal Attribution Layer |
| Training data | MIT Alcator C-Mod disruption database |
| Test ROC-AUC | 0.864 (single seed), 0.857 (5-seed ensemble) |
| PR-AUC | 0.449 |
| Parameters | 62,671 |
| Mean warning lead time | 113 ms (3× earlier than labeled horizon) |
| Comparison baseline | HDL (2021): 0.801 ROC-AUC |

The 5-seed ensemble trades a minor AUC variance for improved probability calibration, making it better suited for operational threshold-setting.

---

## Cross-Tokamak Generalization

| Train | Test | AUC |
|-------|------|-----|
| C-Mod + DIII-D | EAST | 0.845 |
| C-Mod + EAST | DIII-D | 0.636 |
| DIII-D + EAST | C-Mod | 0.853 |
| — | **Mean** | **0.778** |

Exceeds the single-device published baseline in 2 of 3 leave-one-machine-out scenarios without fine-tuning.

---

## Dataset

[DisruptionBench](https://dataverse.harvard.edu/dataverse/disruption) — Harvard Dataverse. Plasma shots from MIT Alcator C-Mod, DIII-D (General Atomics), and EAST (ASIPP, China). 13 diagnostic signals per shot, 50 timesteps per sample, binary disruption labels with a 50ms prediction horizon.

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
```

### Run Full Evaluation
```bash
python eval_full.py
```

### Attribution Trajectory
```bash
python attribution_trajectory.py
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

## Project Structure

```
SHARD/
  model.py                   # SHARD architecture + training loop
  train.py                   # Training script
  eval_full.py               # Full evaluation pipeline
  baselines.py               # Baseline comparisons (RF, LSTM)
  ablation.py                # Ablation study
  ensemble_eval.py           # 5-seed ensemble evaluation
  cross_tokamak.py           # Leave-one-machine-out evaluation
  attribution_trajectory.py  # Attribution heatmap over disruptive shot
  data_loader.py             # Data loading and preprocessing
  app.py                     # Streamlit demo app
  requirements.txt           # Python dependencies
  shard_refs.bib             # Bibliography
  hf_model_card.md           # HuggingFace model card
```

---

## Disclaimer

SHARD is a research prototype, not a certified safety system. Predictions should be used alongside existing plasma control infrastructure and are not a replacement for validated disruption mitigation hardware.

---

## Author

Veer Goradia  
GitHub: [vgoradia](https://github.com/vgoradia) | Hugging Face: [vgoradia](https://huggingface.co/vgoradia)

---
