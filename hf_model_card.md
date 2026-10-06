# SHARD: Signal Harmonics for Anomaly and Reactor Disruption

**SHARD** is a Transformer-based deep learning model for predicting plasma disruptions in tokamak fusion reactors. It was developed by Veer Goradia and submitted to the Regeneron Science Talent Search 2027.

## Model Description

Plasma disruptions — sudden losses of plasma confinement — are the single greatest threat to sustained fusion energy. They can release megajoules of energy in milliseconds, damaging reactor walls and quenching the plasma. SHARD predicts disruptions far enough in advance to trigger mitigation systems, and explains *which* plasma signals drove each prediction.

**Architecture:**
- Linear projection of 13 plasma diagnostic signals into a 64-dimensional embedding space
- Learnable positional encoding for temporal awareness
- 3-layer Transformer encoder with multi-head self-attention (4 heads)
- Novel **Signal Attribution Layer**: cross-signal attention that identifies which plasma parameters (β_p, q₉₅, MHD modes, etc.) most influenced each prediction
- Dual output heads: disruption probability (binary classification) + time-to-disruption (regression)

**Total parameters:** 62,671

## Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.864 |
| PR-AUC | 0.449 |
| Improvement over HDL baseline | +6.3 points ROC-AUC |
| 5-seed ensemble ROC-AUC | 0.857 ± 0.008 |

Cross-tokamak generalization: model trained on C-Mod and evaluated zero-shot on DIII-D, demonstrating transfer without retraining.

## Training Data

- **Primary dataset:** MIT Alcator C-Mod disruption database (DisruptionBench)
- **Signals used (13):** β_p, lᵢ, q₉₅, n₁ MHD mode, n/nG (Greenwald fraction), lower gap, κ (elongation), Ip error, loop voltage, radiated power, Wmhd, dW/dt, v_loop
- **Sequence length:** 100 timesteps per shot
- **Split:** 80/10/10 train/val/test, stratified by disruption label

## Intended Use

- Research use for plasma disruption prediction and tokamak safety
- Real-time deployment on tokamak diagnostic systems (latency ~2ms per shot)
- Interpretability analysis via Signal Attribution Layer

## How to Use

```python
import torch
from model import SHARDEncoder, SIGNAL_NAMES

# Load model
model = SHARDEncoder(n_signals=13, seq_len=100, d_model=64, n_heads=4, n_layers=3)
model.load_state_dict(torch.load('shard_best.pt', map_location='cpu'))
model.eval()

# Input: (batch, seq_len, n_signals) tensor of normalized plasma signals
x = torch.randn(1, 100, 13)  # replace with real diagnostic data
disruption_prob, time_to_disruption, signal_weights = model(x)

print(f"Disruption probability: {disruption_prob.item():.3f}")
print(f"Time to disruption: {time_to_disruption.item():.1f} ms")

# Get signal attribution (which signals drove the prediction)
attribution = model.get_signal_attribution(x, SIGNAL_NAMES)
for signal, weight in sorted(attribution.items(), key=lambda x: -x[1]):
    print(f"  {signal}: {weight:.4f}")
```

## Live Demo

Try the interactive Streamlit app: [SHARD Demo](https://huggingface.co/spaces/vgoradia/SHARD)

## Citation

If you use SHARD in your research, please cite:

```
@misc{goradia2026shard,
  title={SHARD: Signal Harmonics for Anomaly and Reactor Disruption — 
         A Transformer Architecture for Interpretable Plasma Disruption Prediction},
  author={Goradia, Veer},
  year={2026},
  note={Submitted to Regeneron Science Talent Search 2027}
}
```

## License

MIT License — see LICENSE file.

## Contact

Veer Goradia · vgoradia07@gmail.com
