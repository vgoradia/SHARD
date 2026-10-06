"""
Ablation study for SHARD — fast version (12 epochs, 2 key variants).
  1. Full SHARD
  2. w/o Signal Attribution
  3. w/o Positional Encoding
Also generates training curve figure.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import get_dataloaders

def focal_loss(pred, target, gamma=2.0, pos_weight=10.0):
    target = target.float()
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    w = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
    return (w * (1 - pt)**gamma * bce).mean()

class FullSHARD(nn.Module):
    def __init__(self, n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_signals, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.sig_proj = nn.Linear(d_model, n_signals)  # signal attribution
        self.cls = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        g = h.mean(1)
        return torch.sigmoid(self.cls(g)).squeeze(-1)

class NoSignalAttr(nn.Module):
    def __init__(self, n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_signals, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        # No sig_proj — no attribution layer
        self.cls = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        g = h.mean(1)
        return torch.sigmoid(self.cls(g)).squeeze(-1)

class NoPosEnc(nn.Module):
    def __init__(self, n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_signals, d_model)
        # No positional encoding parameter
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(self.proj(x))  # no pos enc
        g = h.mean(1)
        return torch.sigmoid(self.cls(g)).squeeze(-1)

class SingleLayer(nn.Module):
    def __init__(self, n_signals=13, seq_len=50, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_signals, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, 1)
        self.cls = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        g = h.mean(1)
        return torch.sigmoid(self.cls(g)).squeeze(-1)


def train_variant(model, train_loader, val_loader, test_loader, n_epochs=12, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best_val, best_state = 0, None
    val_curve = []

    for epoch in range(n_epochs):
        model.train()
        for xb, yb, _ in train_loader:
            opt.zero_grad()
            pred = model(xb)
            focal_loss(pred, yb.float()).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                vp.extend(model(xb).cpu().numpy())
                vy.extend(yb.numpy())
        va = roc_auc_score(vy, vp)
        val_curve.append(va)
        if va > best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep{epoch:2d}: val AUC {va:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    tp, ty = [], []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            tp.extend(model(xb).cpu().numpy())
            ty.extend(yb.numpy())
    test_auc = roc_auc_score(ty, tp)
    return test_auc, val_curve


print("Loading data...")
train_loader, val_loader, test_loader, scaler = get_dataloaders(
    "/home/claude/SHARD/fig1_data.mat", batch_size=512
)

variants = {
    'SHARD (Full)': FullSHARD(),
    'w/o Signal Attribution': NoSignalAttr(),
    'w/o Positional Encoding': NoPosEnc(),
    '1-Layer Transformer': SingleLayer(),
}

results = {}
curves = {}
for name, model in variants.items():
    print(f"\nTraining: {name}")
    auc, curve = train_variant(model, train_loader, val_loader, test_loader, n_epochs=12)
    print(f"  → Test AUC: {auc:.4f}")
    results[name] = round(auc, 4)
    curves[name] = curve

results['LSTM Baseline'] = 0.8573

with open('ablation_results.json', 'w') as f:
    json.dump({'ablation': results, 'curves': {k: v for k, v in curves.items()}}, f, indent=2)

print("\n=== ABLATION RESULTS ===")
full_auc = results['SHARD (Full)']
for k, v in results.items():
    delta = v - full_auc
    marker = "" if k == 'SHARD (Full)' else f"  ({delta:+.4f})"
    print(f"  {k}: {v:.4f}{marker}")

# ---- Figure: Training curves ----
fig, ax = plt.subplots(figsize=(7, 4.5))
palette = {
    'SHARD (Full)': '#2ecc71',
    'w/o Signal Attribution': '#e74c3c',
    'w/o Positional Encoding': '#e67e22',
    '1-Layer Transformer': '#3498db',
}
for name, curve in curves.items():
    ax.plot(range(1, len(curve)+1), curve,
            label=name, color=palette.get(name, '#999'),
            lw=2.5 if name == 'SHARD (Full)' else 1.5,
            linestyle='-' if name == 'SHARD (Full)' else '--')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation AUC-ROC', fontsize=12)
ax.set_title('Training Curves — Ablation Variants', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Saved training_curves.png")

# ---- Figure: Ablation bar chart ----
fig, ax = plt.subplots(figsize=(8, 4))
names = [k for k in results.keys() if k != 'LSTM Baseline']
aucs = [results[k] for k in names]
colors = ['#2ecc71' if n == 'SHARD (Full)' else '#e74c3c' for n in names]
bars = ax.barh(names, aucs, color=colors, height=0.45, zorder=3)
ax.axvline(full_auc, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7,
           label=f'Full SHARD (AUC={full_auc:.4f})')
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.001, bar.get_y() + bar.get_height()/2,
            f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0.78, 0.91)
ax.set_xlabel('AUC-ROC (Test)', fontsize=12)
ax.set_title('SHARD Ablation Study (MIT C-Mod)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('ablation_chart.png', dpi=150, bbox_inches='tight')
print("Saved ablation_chart.png")
