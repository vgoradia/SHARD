"""
Attribution trajectory: show how per-signal weights evolve over a disruptive shot.
The KEY scientific finding: model tracks physical instability evolution in real time.
Generates: attr_trajectory.png
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---- Model (exact copy from model.py) ----
class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class SignalAttentionBlock(nn.Module):
    def __init__(self, n_signals, d_model):
        super().__init__()
        self.signal_attn = nn.Linear(d_model, n_signals)
    def forward(self, x):
        return torch.softmax(self.signal_attn(x), dim=-1)

class SHARDEncoder(nn.Module):
    def __init__(self, n_signals=13, seq_len=100, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj   = nn.Linear(n_signals, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.signal_attn = SignalAttentionBlock(n_signals, d_model)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Sequential(nn.Linear(d_model,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.ttd_head    = nn.Sequential(nn.Linear(d_model,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        for blk in self.temporal_blocks: h = blk(h)
        sig_w   = self.signal_attn(h)           # (batch, seq_len, n_signals)
        pooled  = self.pool(h.transpose(1,2)).squeeze(-1)
        p_disr  = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        ttd     = F.relu(self.ttd_head(pooled)).squeeze(-1)
        return p_disr, ttd, sig_w

SIGNAL_KEYS = [
    'Greenwald_fraction', 'Mirnov_norm_btor', 'Te_peaking_ECE',
    'beta_p', 'ip_error_normalized', 'kappa', 'li', 'lower_gap',
    'n_equal_1_normalized', 'q95', 'radiated_fraction', 'v_loop', 'z_error'
]

SIGNAL_LABELS = [
    r'$n/n_G$', r'$B_{tor}$', r'$T_{e,pk}$',
    r'$\beta_p$', r'$I_p$ err', r'$\kappa$', r'$l_i$', r'$\delta_{low}$',
    r'$n_1$', r'$q_{95}$', r'$P_{rad}$', r'$v_{loop}$', r'$z_{err}$'
]

# Categories for coloring
SIGNAL_CATEGORIES = {
    'MHD': [0, 1, 8, 9],        # n/nG, Btor, n1, q95
    'Thermal': [2, 3, 10],       # Te_pk, beta_p, Prad
    'Structural': [4, 6, 11],    # Ip_err, li, vloop
    'Positional': [5, 7, 12],    # kappa, lower_gap, z_err
}

CATEGORY_COLORS = {
    'MHD': '#e74c3c',
    'Thermal': '#e67e22',
    'Structural': '#3498db',
    'Positional': '#27ae60',
}

MAT_PATH = "/home/claude/SHARD/fig1_data.mat"
SEQ_LEN  = 50
PRED_HOR = 0.05
STRIDE   = SEQ_LEN // 2
FINE_STRIDE = 5
TRAJ_SHOT = 1150818014   # longest disruptive test shot — richest trajectory (241 timesteps)

print("Loading data...")
data = sio.loadmat(MAT_PATH)
shots = data['shot'].flatten()
tud   = data['time_until_disrupt'].flatten()
sig_matrix = np.column_stack([data[k].flatten() for k in SIGNAL_KEYS])

# Build sequences for scaler fitting
X_seqs, y_labels, sids_arr = [], [], []
shot_raw = {}

for shot_id in np.unique(shots):
    mask = shots == shot_id
    sig  = sig_matrix[mask].copy()
    d_arr = tud[mask]
    if len(sig) < SEQ_LEN + 1:
        continue
    for col in range(sig.shape[1]):
        c = sig[:, col]; nans = np.isnan(c)
        if nans.all():   sig[:, col] = 0.0
        elif nans.any():
            idx = np.where(~nans)[0]
            c[nans] = np.interp(np.where(nans)[0], idx, c[idx])
            sig[:, col] = c
    shot_raw[shot_id] = (sig, d_arr)
    for i in range(0, len(sig) - SEQ_LEN, STRIDE):
        win = sig[i:i+SEQ_LEN]
        if np.isnan(win).any(): continue
        e_tud = d_arr[i + SEQ_LEN - 1]
        y_labels.append(1.0 if (e_tud > 0 and e_tud <= PRED_HOR) else 0.0)
        X_seqs.append(win)
        sids_arr.append(shot_id)

X = np.array(X_seqs, dtype=np.float32)
sids_arr = np.array(sids_arr)
unique_shots = np.unique(sids_arr)
train_s, test_s = train_test_split(unique_shots, test_size=0.2, random_state=42)
train_s, val_s  = train_test_split(train_s, test_size=0.15, random_state=42)
train_mask = np.isin(sids_arr, train_s)

scaler = StandardScaler()
scaler.fit(X[train_mask].reshape(-1, X.shape[-1]))

# Load model
model = SHARDEncoder(n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3)
model.load_state_dict(torch.load("/home/claude/SHARD/shard_best.pt", map_location='cpu'))
model.eval()
print(f"Model loaded. Running trajectory on shot {TRAJ_SHOT}...")

# Generate trajectory
sig_raw, tud_raw = shot_raw[TRAJ_SHOT]
traj_probs, traj_ttds, traj_attr = [], [], []
for i in range(0, len(sig_raw) - SEQ_LEN, FINE_STRIDE):
    win = sig_raw[i:i+SEQ_LEN].copy()
    if np.isnan(win).any(): continue
    win_scaled = scaler.transform(win.reshape(-1, 13)).reshape(1, SEQ_LEN, 13)
    x_t = torch.tensor(win_scaled, dtype=torch.float32)
    with torch.no_grad():
        p, _, sig_w = model(x_t)
    # Average attribution over seq_len positions
    attr_mean = sig_w.squeeze(0).mean(0).cpu().numpy()  # (13,)
    traj_probs.append(p.item())
    traj_ttds.append(tud_raw[i + SEQ_LEN - 1])
    traj_attr.append(attr_mean)

traj_probs = np.array(traj_probs)
traj_ttds  = np.array(traj_ttds)
traj_attr  = np.array(traj_attr)   # (T_windows, 13)
traj_time  = np.arange(len(traj_probs)) * FINE_STRIDE * 0.005   # seconds

print(f"Trajectory: {len(traj_probs)} windows, prob range [{traj_probs.min():.3f}, {traj_probs.max():.3f}]")
print(f"Attribution shape: {traj_attr.shape}")
print(f"Signal weights at final window: {dict(zip(SIGNAL_LABELS, traj_attr[-1].round(3)))}")

# ---- FIGURE: Attribution trajectory ----
# Two panels stacked: top = attribution heatmap, bottom = disruption probability
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1], hspace=0.08)

ax_heat = fig.add_subplot(gs[0])
ax_prob = fig.add_subplot(gs[1])

# ---- Top panel: attribution heatmap ----
# We want signals sorted by category and by final-window weight
# Sort signals: put highest final-window weight at top
final_weights = traj_attr[-1]
sort_idx = np.argsort(final_weights)[::-1]   # descending by final weight

sorted_attr  = traj_attr[:, sort_idx].T   # (13, T_windows)
sorted_labels = [SIGNAL_LABELS[i] for i in sort_idx]

# Build category color list for y-axis labels
def get_cat(sig_idx):
    for cat, idxs in SIGNAL_CATEGORIES.items():
        if sig_idx in idxs:
            return cat
    return 'MHD'

label_colors = [CATEGORY_COLORS[get_cat(i)] for i in sort_idx]

im = ax_heat.imshow(
    sorted_attr,
    aspect='auto',
    cmap='YlOrRd',
    interpolation='nearest',
    vmin=0, vmax=sorted_attr.max() * 0.85,
    extent=[traj_time[0], traj_time[-1], -0.5, 12.5],
    origin='lower'
)

# Disruption zone shading
disrupt_mask = (traj_ttds >= 0) & (traj_ttds <= PRED_HOR)
if disrupt_mask.any():
    dz = traj_time[disrupt_mask]
    ax_heat.axvspan(dz.min(), dz.max(), alpha=0.25, color='#e74c3c', zorder=3,
                    label='Disruption window')
    ax_heat.axvline(dz.min(), color='#c0392b', lw=1.5, ls='--', alpha=0.7)

ax_heat.set_yticks(range(13))
ax_heat.set_yticklabels(sorted_labels, fontsize=9)
for ticklabel, color in zip(ax_heat.get_yticklabels(), label_colors):
    ticklabel.set_color(color)
    ticklabel.set_fontweight('bold')

cbar = plt.colorbar(im, ax=ax_heat, pad=0.01, shrink=0.9)
cbar.set_label('Attribution Weight', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax_heat.set_ylabel('Plasma Diagnostic', fontsize=11)
ax_heat.set_xticklabels([])
ax_heat.set_title(
    'SHARD Signal Attribution Over Disruptive Shot\n'
    '(how the model\'s attention shifts as disruption approaches)',
    fontsize=12, fontweight='bold', pad=8
)

# Legend for category colors
from matplotlib.patches import Patch
cat_legend = [Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
ax_heat.legend(handles=cat_legend, fontsize=8, loc='upper left',
               ncol=4, framealpha=0.85, columnspacing=0.8)

# ---- Bottom panel: disruption probability ----
ax_prob.plot(traj_time, traj_probs, color='#2c3e50', lw=2.0,
             label='P(disruption)')
ax_prob.fill_between(traj_time, traj_probs, alpha=0.15, color='#2c3e50')
ax_prob.axhline(0.3, color='#e74c3c', lw=1.3, ls='--', alpha=0.7,
                label='Alarm threshold')
if disrupt_mask.any():
    dz = traj_time[disrupt_mask]
    ax_prob.axvspan(dz.min(), dz.max(), alpha=0.2, color='#e74c3c')
ax_prob.set_ylim([-0.05, 1.1])
ax_prob.set_ylabel('P(disruption)', fontsize=10)
ax_prob.set_xlabel('Time in Plasma Shot (s)', fontsize=11)
ax_prob.legend(fontsize=9, loc='upper left')
ax_prob.grid(alpha=0.3)
ax_prob.spines['top'].set_visible(False)
ax_prob.spines['right'].set_visible(False)

plt.savefig('/home/claude/SHARD/attr_trajectory.png', dpi=150, bbox_inches='tight')
print("Saved attr_trajectory.png")

# ---- Print summary stats for paper ----
print("\n--- Attribution summary at disruption window entry ---")
if disrupt_mask.any():
    first_disrupt_idx = np.where(disrupt_mask)[0][0]
    attr_at_alarm = traj_attr[first_disrupt_idx]
    top3 = np.argsort(attr_at_alarm)[::-1][:3]
    print("Top signals at disruption:")
    for rank, idx in enumerate(top3):
        print(f"  {rank+1}. {SIGNAL_KEYS[idx]}: {attr_at_alarm[idx]:.3f}")
