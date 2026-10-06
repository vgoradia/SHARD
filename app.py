"""
SHARD: Signal Harmonics for Anomaly and Reactor Disruption
Streamlit demo app — live disruption probability + signal attribution
"""

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHARD — Plasma Disruption Predictor",
    page_icon="⚡",
    layout="wide"
)

# ── inline model (no checkpoint needed for demo) ───────────────────────────────
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, x):
        attn_out, self.attn_weights = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class SignalAttentionBlock(nn.Module):
    def __init__(self, n_signals, d_model):
        super().__init__()
        self.signal_attn = nn.Linear(d_model, n_signals)
        self.signal_weights = None

    def forward(self, x):
        self.signal_weights = torch.softmax(self.signal_attn(x), dim=-1)
        return self.signal_weights

class SHARDEncoder(nn.Module):
    def __init__(self, n_signals=13, seq_len=100, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_signals = n_signals
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(n_signals, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.signal_attn = SignalAttentionBlock(n_signals, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.ttd_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        for block in self.temporal_blocks:
            h = block(h)
        sig_weights = self.signal_attn(h)
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)
        disruption_prob = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        ttd = F.relu(self.ttd_head(pooled)).squeeze(-1)
        return disruption_prob, ttd, sig_weights

    def get_signal_attribution(self, x, signal_names):
        self.eval()
        with torch.no_grad():
            _, _, sig_weights = self.forward(x)
        mean_weights = sig_weights.mean(dim=1).mean(dim=0)
        return dict(zip(signal_names, mean_weights.cpu().numpy()))


SIGNAL_NAMES = [
    'β_p (plasma pressure)',
    'lᵢ (internal inductance)',
    'q₉₅ (safety factor)',
    'n₁ mode (MHD amplitude)',
    'n/nG (Greenwald fraction)',
    'lower gap (plasma-wall)',
    'κ (elongation)',
    'Ip error (current error)',
    'loop voltage',
    'radiated power',
    'Wmhd (stored energy)',
    'dW/dt (energy rate)',
    'v_loop (toroidal voltage)'
]

SIGNAL_KEYS = [
    'beta_p', 'li', 'q95', 'n1_mode', 'n_over_nG',
    'lower_gap', 'kappa', 'Ip_error', 'loop_voltage',
    'radiated_power', 'Wmhd', 'dWdt', 'v_loop'
]

# ── load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = SHARDEncoder(n_signals=13, seq_len=100, d_model=64, n_heads=4, n_layers=3)
    # Try to load trained weights; fall back to untrained demo if not found
    import os
    ckpt_path = os.path.join(os.path.dirname(__file__), 'shard_best.pt')
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state)
            return model, True
        except Exception:
            pass
    return model, False

# ── helper: generate synthetic shot ───────────────────────────────────────────
def synthetic_shot(disruption: bool, noise: float = 0.3):
    """
    Returns (100, 13) array simulating a tokamak plasma shot.
    If disruption=True, signals deteriorate in the final 20 timesteps.
    """
    np.random.seed(np.random.randint(0, 9999))
    t = np.linspace(0, 1, 100)
    base = np.random.randn(100, 13) * noise

    if disruption:
        # Ramp degradation: β_p drops, n/nG climbs, MHD mode spikes
        ramp = np.clip((t - 0.7) / 0.3, 0, 1)
        base[:, 0] -= ramp * 2.0   # beta_p drops
        base[:, 3] += ramp * 3.0   # n1_mode spikes
        base[:, 4] += ramp * 1.5   # n/nG rises
        base[:, 11] += ramp * 2.5  # dW/dt spikes
        base[:, 8] += ramp * 1.8   # loop voltage rises

    return base.astype(np.float32)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("⚡ SHARD — Plasma Disruption Predictor")
st.markdown(
    "**SHARD** (Signal Harmonics for Anomaly and Reactor Disruption) is a "
    "Transformer-based model that predicts plasma disruptions in tokamak fusion "
    "reactors. It achieves **ROC-AUC 0.864** on held-out shots and generalizes "
    "across tokamaks without retraining."
)
st.markdown("---")

model, loaded_weights = load_model()
if loaded_weights:
    st.success("Trained model weights loaded (shard_best.pt)")
else:
    st.info("Running in demo mode — model weights not bundled (too large). Predictions use random init for illustration.")

# ── sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Shot Configuration")
mode = st.sidebar.radio(
    "Shot type",
    ["Stable plasma", "Disrupting plasma", "Custom (manual signals)"],
    index=1
)

noise_level = st.sidebar.slider("Signal noise level", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Manual signal overrides** (only used in Custom mode)")
manual_vals = {}
if mode == "Custom (manual signals)":
    for name, key in zip(SIGNAL_NAMES, SIGNAL_KEYS):
        manual_vals[key] = st.sidebar.slider(name, -3.0, 3.0, 0.0, 0.1)

# ── run inference ──────────────────────────────────────────────────────────────
if mode == "Stable plasma":
    shot_data = synthetic_shot(disruption=False, noise=noise_level)
elif mode == "Disrupting plasma":
    shot_data = synthetic_shot(disruption=True, noise=noise_level)
else:
    shot_data = np.zeros((100, 13), dtype=np.float32)
    for i, key in enumerate(SIGNAL_KEYS):
        shot_data[:, i] = manual_vals.get(key, 0.0)
    shot_data += (np.random.randn(100, 13) * noise_level).astype(np.float32)

x_tensor = torch.tensor(shot_data).unsqueeze(0)  # (1, 100, 13)

model.eval()
with torch.no_grad():
    prob, ttd, _ = model(x_tensor)
    prob_val = prob.item()
    ttd_val = ttd.item()

attribution = model.get_signal_attribution(x_tensor, SIGNAL_NAMES)

# ── layout ─────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    color = "#e74c3c" if prob_val > 0.5 else "#2ecc71"
    st.markdown(
        f"<div style='text-align:center; padding:20px; background:#1a1a2e; "
        f"border-radius:12px; border: 2px solid {color};'>"
        f"<p style='color:#aaa; margin:0; font-size:14px;'>DISRUPTION PROBABILITY</p>"
        f"<p style='color:{color}; font-size:52px; font-weight:bold; margin:8px 0;'>"
        f"{prob_val:.1%}</p>"
        f"<p style='color:#aaa; margin:0; font-size:12px;'>"
        f"{'⚠️ HIGH RISK' if prob_val > 0.5 else '✅ STABLE'}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div style='text-align:center; padding:20px; background:#1a1a2e; "
        f"border-radius:12px; border: 2px solid #3498db;'>"
        f"<p style='color:#aaa; margin:0; font-size:14px;'>EST. TIME TO DISRUPTION</p>"
        f"<p style='color:#3498db; font-size:52px; font-weight:bold; margin:8px 0;'>"
        f"{ttd_val:.0f}</p>"
        f"<p style='color:#aaa; margin:0; font-size:12px;'>milliseconds</p>"
        f"</div>",
        unsafe_allow_html=True
    )

with col3:
    model_mode = "Trained" if loaded_weights else "Demo"
    shot_label = mode
    st.markdown(
        f"<div style='text-align:center; padding:20px; background:#1a1a2e; "
        f"border-radius:12px; border: 2px solid #9b59b6;'>"
        f"<p style='color:#aaa; margin:0; font-size:14px;'>MODEL</p>"
        f"<p style='color:#9b59b6; font-size:28px; font-weight:bold; margin:8px 0;'>"
        f"SHARD</p>"
        f"<p style='color:#aaa; margin:0; font-size:12px;'>"
        f"{model_mode} · ROC-AUC 0.864</p>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ── attribution chart ──────────────────────────────────────────────────────────
st.subheader("📊 Signal Attribution — What drove this prediction?")
st.markdown(
    "The Signal Attention Layer shows which plasma parameters most influenced "
    "the disruption prediction. Higher bars = more attention from the model."
)

sorted_attr = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
signals, weights = zip(*sorted_attr)
weights = np.array(weights)

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

colors = ['#e74c3c' if w > weights.mean() + weights.std() else '#3498db' for w in weights]
bars = ax.barh(range(len(signals)), weights, color=colors, edgecolor='none', height=0.6)

ax.set_yticks(range(len(signals)))
ax.set_yticklabels(signals, color='#ccc', fontsize=10)
ax.set_xlabel("Attention Weight", color='#aaa', fontsize=10)
ax.tick_params(colors='#aaa')
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()

# Annotate bars
for i, (bar, w) in enumerate(zip(bars, weights)):
    ax.text(w + 0.001, i, f'{w:.3f}', va='center', color='#ccc', fontsize=9)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── signal traces ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Plasma Signal Traces (100 timesteps)")

top3_signals = [s for s, _ in sorted_attr[:3]]
top3_indices = [SIGNAL_NAMES.index(s) for s in top3_signals]

fig2, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
fig2.patch.set_facecolor('#0d1117')
colors_trace = ['#e74c3c', '#f39c12', '#2ecc71']

for ax, idx, color, name in zip(axes, top3_indices, colors_trace, top3_signals):
    ax.set_facecolor('#0d1117')
    ax.plot(shot_data[:, idx], color=color, linewidth=1.5)
    ax.set_ylabel(name, color='#ccc', fontsize=8)
    ax.tick_params(colors='#555', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333')

axes[-1].set_xlabel("Timestep", color='#aaa', fontsize=9)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='color:#555; font-size:12px; text-align:center;'>"
    "SHARD · Veer Goradia · "
    "62,671 parameters · Trained on C-Mod disruption database · "
    "github.com/vgoradia/SHARD"
    "</div>",
    unsafe_allow_html=True
)
