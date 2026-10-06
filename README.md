
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
# Outputs: attr_trajectory.png (heatmap of attribution weights over a disruptive shot)
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

## Repository Structure

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

**Veer Goradia**
vgoradia07@gmail.com

---
