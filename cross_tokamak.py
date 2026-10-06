"""
Cross-tokamak generalization experiments for SHARD.

Shared signals across C-Mod, DIII-D, EAST:
  Greenwald_fraction, beta_p, ip_error_normalized, kappa, li,
  lower_gap, n_equal_1_normalized, q95, v_loop, radiated_fraction/rad_input_frac

We use the 10 signals common to all three devices.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 10 signals shared across all three tokamaks
SHARED_SIGNALS = [
    'Greenwald_fraction',
    'beta_p',
    'ip_error_normalized',
    'kappa',
    'li',
    'lower_gap',
    'n_equal_1_normalized',
    'q95',
    'v_loop',
    # radiated fraction has different key names — handle separately
]

SIGNAL_MAP = {
    'C_Mod':  {
        'path': '/tmp/C_Mod_for_nn.mat',
        'rad': 'radiated_fraction',
        'z': 'z_error',
    },
    'DIII-D': {
        'path': '/tmp/d3d_for_nn.mat',
        'rad': 'radiated_fraction',
        'z': 'zcur',
    },
    'EAST':   {
        'path': '/tmp/EAST_for_nn.mat',
        'rad': 'rad_input_frac',
        'z': 'z_error',
    },
}

FINAL_SIGNALS = SHARED_SIGNALS + ['radiated_fraction']  # we rename to unified name
N_SIGNALS = 10  # 9 shared + radiated

SEQ_LEN = 50
STRIDE = 25
DISRUPT_WINDOW = 0.05  # 50ms


def load_device(device_name):
    """Load a single tokamak's data, return (X, y, shots) arrays."""
    cfg = SIGNAL_MAP[device_name]
    data = sio.loadmat(cfg['path'])

    # Build signal matrix: (N_timesteps, N_SIGNALS)
    cols = []
    for sig in SHARED_SIGNALS:
        cols.append(data[sig].flatten())
    # Add radiated fraction under unified key
    cols.append(data[cfg['rad']].flatten())

    X_raw = np.stack(cols, axis=1).astype(np.float32)  # (T, 10)
    shots = data['shot'].flatten()
    ttd = data['time_until_disrupt'].flatten()

    return X_raw, ttd, shots


def build_sequences(X_raw, ttd, shots, seq_len=SEQ_LEN, stride=STRIDE, disrupt_win=DISRUPT_WINDOW):
    """Build sliding window sequences with shot-level NaN handling."""
    unique_shots = np.unique(shots)
    seqs, labels = [], []

    for shot_id in unique_shots:
        mask = shots == shot_id
        X_shot = X_raw[mask]
        ttd_shot = ttd[mask]

        # Forward fill NaNs per signal
        for s in range(X_shot.shape[1]):
            col = X_shot[:, s]
            nans = np.isnan(col)
            if nans.any():
                # forward fill
                not_nan = np.where(~nans)[0]
                if len(not_nan) == 0:
                    X_shot[:, s] = 0.0
                    continue
                for i in range(len(col)):
                    if nans[i]:
                        prev = not_nan[not_nan < i]
                        X_shot[i, s] = col[prev[-1]] if len(prev) else col[not_nan[0]]

        T = len(X_shot)
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            window = X_shot[start:end]
            # Label: is disruption imminent at end of window?
            ttd_end = ttd_shot[end - 1]
            label = 1 if (ttd_end >= 0 and ttd_end <= disrupt_win) else 0
            if not np.isnan(window).any():
                seqs.append(window)
                labels.append(label)

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)


def make_loader(seqs, labels, scaler=None, fit_scaler=False, batch_size=512, shuffle=True):
    N, L, S = seqs.shape
    flat = seqs.reshape(-1, S)
    if fit_scaler:
        scaler = StandardScaler()
        flat = scaler.fit_transform(flat)
    else:
        flat = scaler.transform(flat)
    seqs_norm = flat.reshape(N, L, S)
    X_t = torch.tensor(seqs_norm, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), scaler


# ---- Compact SHARD for cross-device (10 signals) ----
class SHARDCross(nn.Module):
    def __init__(self, n_signals=10, seq_len=50, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_signals, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.signal_proj = nn.Linear(d_model, n_signals)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc
        h = self.transformer(h)
        g = h.mean(dim=1)
        sig_w = torch.softmax(self.signal_proj(g), dim=-1)
        prob = torch.sigmoid(self.cls_head(g)).squeeze(-1)
        return prob, sig_w


def focal_loss(pred, target, gamma=2.0, pos_weight=10.0):
    target = target.float()
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    weight = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
    return (weight * (1 - pt) ** gamma * bce).mean()


def train_eval(train_loader, val_loader, test_loader, n_signals=10, n_epochs=12, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cpu')
    model = SHARDCross(n_signals=n_signals).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val_auc, best_state = 0, None
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            prob, _ = model(xb)
            loss = focal_loss(prob, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p, _ = model(xb.to(device))
                vp.extend(p.cpu().numpy()); vy.extend(yb.numpy())
        if len(set(vy)) > 1:
            va = roc_auc_score(vy, vp)
            if va > best_val_auc:
                best_val_auc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 3 == 0:
            print(f"    epoch {epoch:2d}: val AUC {best_val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    tp, ty = [], []
    sig_weights_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            p, sw = model(xb.to(device))
            tp.extend(p.cpu().numpy()); ty.extend(yb.numpy())
            # collect signal weights for disruptive predictions
            mask = yb == 1
            if mask.any():
                sig_weights_all.append(sw[mask].cpu().numpy())

    test_auc = roc_auc_score(ty, tp) if len(set(ty)) > 1 else 0.0
    mean_sig_w = np.concatenate(sig_weights_all, 0).mean(0) if sig_weights_all else np.zeros(n_signals)
    return test_auc, mean_sig_w


# ---- Main experiments ----
print("Loading all tokamak data...")
data_cmod = load_device('C_Mod')
data_d3d = load_device('DIII-D')
data_east = load_device('EAST')

print("Building sequences...")
seqs_cmod, labs_cmod = build_sequences(*data_cmod)
seqs_d3d, labs_d3d = build_sequences(*data_d3d)
seqs_east, labs_east = build_sequences(*data_east)

print(f"C-Mod:  {len(seqs_cmod):6d} seqs, {labs_cmod.sum():4d} disruptive ({100*labs_cmod.mean():.1f}%)")
print(f"DIII-D: {len(seqs_d3d):6d} seqs, {labs_d3d.sum():4d} disruptive ({100*labs_d3d.mean():.1f}%)")
print(f"EAST:   {len(seqs_east):6d} seqs, {labs_east.sum():4d} disruptive ({100*labs_east.mean():.1f}%)")

results = {}

# ---- Experiment 1: Train C-Mod+DIII-D → Test EAST ----
print("\n=== EXP 1: Train C-Mod+DIII-D, Test EAST ===")
# Combine C-Mod + DIII-D, split off 10% as val
X_train_all = np.concatenate([seqs_cmod, seqs_d3d], axis=0)
y_train_all = np.concatenate([labs_cmod, labs_d3d], axis=0)
idx = np.random.RandomState(42).permutation(len(X_train_all))
n_val = int(0.12 * len(idx))
val_idx, train_idx = idx[:n_val], idx[n_val:]
X_tr, y_tr = X_train_all[train_idx], y_train_all[train_idx]
X_vl, y_vl = X_train_all[val_idx], y_train_all[val_idx]

tr_loader, scaler1 = make_loader(X_tr, y_tr, fit_scaler=True)
vl_loader, _ = make_loader(X_vl, y_vl, scaler=scaler1, shuffle=False)
te_loader, _ = make_loader(seqs_east, labs_east, scaler=scaler1, shuffle=False)

auc1, sw1 = train_eval(tr_loader, vl_loader, te_loader, n_epochs=12)
print(f"  → Test on EAST AUC: {auc1:.4f}")
results['train_cmod_d3d_test_east'] = round(auc1, 4)

# ---- Experiment 2: Train C-Mod+EAST → Test DIII-D ----
print("\n=== EXP 2: Train C-Mod+EAST, Test DIII-D ===")
X_train_all2 = np.concatenate([seqs_cmod, seqs_east], axis=0)
y_train_all2 = np.concatenate([labs_cmod, labs_east], axis=0)
idx2 = np.random.RandomState(42).permutation(len(X_train_all2))
n_val2 = int(0.12 * len(idx2))
val_idx2, train_idx2 = idx2[:n_val2], idx2[n_val2:]
X_tr2, y_tr2 = X_train_all2[train_idx2], y_train_all2[train_idx2]
X_vl2, y_vl2 = X_train_all2[val_idx2], y_train_all2[val_idx2]

tr_loader2, scaler2 = make_loader(X_tr2, y_tr2, fit_scaler=True)
vl_loader2, _ = make_loader(X_vl2, y_vl2, scaler=scaler2, shuffle=False)
te_loader2, _ = make_loader(seqs_d3d, labs_d3d, scaler=scaler2, shuffle=False)

auc2, sw2 = train_eval(tr_loader2, vl_loader2, te_loader2, n_epochs=12)
print(f"  → Test on DIII-D AUC: {auc2:.4f}")
results['train_cmod_east_test_d3d'] = round(auc2, 4)

# ---- Experiment 3: Train DIII-D+EAST → Test C-Mod ----
print("\n=== EXP 3: Train DIII-D+EAST, Test C-Mod ===")
X_train_all3 = np.concatenate([seqs_d3d, seqs_east], axis=0)
y_train_all3 = np.concatenate([labs_d3d, labs_east], axis=0)
idx3 = np.random.RandomState(42).permutation(len(X_train_all3))
n_val3 = int(0.12 * len(idx3))
val_idx3, train_idx3 = idx3[:n_val3], idx3[n_val3:]

tr_loader3, scaler3 = make_loader(X_train_all3[train_idx3], y_train_all3[train_idx3], fit_scaler=True)
vl_loader3, _ = make_loader(X_train_all3[val_idx3], y_train_all3[val_idx3], scaler=scaler3, shuffle=False)
te_loader3, _ = make_loader(seqs_cmod, labs_cmod, scaler=scaler3, shuffle=False)

auc3, sw3 = train_eval(tr_loader3, vl_loader3, te_loader3, n_epochs=12)
print(f"  → Test on C-Mod AUC: {auc3:.4f}")
results['train_d3d_east_test_cmod'] = round(auc3, 4)

print(f"\n{'='*50}")
print(f"CROSS-TOKAMAK RESULTS SUMMARY:")
print(f"  C-Mod+DIII-D → EAST:  AUC = {auc1:.4f}")
print(f"  C-Mod+EAST → DIII-D:  AUC = {auc2:.4f}")
print(f"  DIII-D+EAST → C-Mod:  AUC = {auc3:.4f}")
mean_cross = np.mean([auc1, auc2, auc3])
print(f"  Mean cross-device AUC: {mean_cross:.4f}")
results['mean_cross_device'] = round(mean_cross, 4)

with open('cross_tokamak_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ---- Plot: cross-device comparison ----
fig, ax = plt.subplots(figsize=(8, 5))
scenarios = ['C-Mod+DIII-D\n→ EAST', 'C-Mod+EAST\n→ DIII-D', 'DIII-D+EAST\n→ C-Mod']
aucs = [auc1, auc2, auc3]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(scenarios, aucs, color=colors, width=0.5, zorder=3, edgecolor='#333', linewidth=1.2)
ax.axhline(0.801, color='gray', linestyle='--', linewidth=1.5, label='Single-device baseline (HDL, AUC=0.801)')
ax.axhline(mean_cross, color='#e67e22', linestyle=':', linewidth=2, label=f'Mean cross-device AUC={mean_cross:.3f}')
ax.set_ylim(0.6, 1.0)
ax.set_ylabel('AUC-ROC (Test Device)', fontsize=13)
ax.set_title('SHARD Cross-Tokamak Generalization\n(Train on 2 devices, Test on held-out device)', fontsize=13, fontweight='bold')
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('cross_tokamak.png', dpi=150, bbox_inches='tight')
print("Saved cross_tokamak.png")
