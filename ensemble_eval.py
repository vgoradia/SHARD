"""
Train 5 SHARD seeds (15 epochs each) and compute ensemble predictions.
Outputs: ensemble_results.json
"""
import numpy as np, scipy.io as sio, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import json

class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_m, n_h, dr=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_m, n_h, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(d_m)
        self.drop = nn.Dropout(dr)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(a))

class SHARDEncoder(nn.Module):
    def __init__(self, n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj   = nn.Linear(n_signals, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.temporal_blocks = nn.ModuleList([TemporalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.signal_attn = nn.Sequential(nn.Linear(d_model, n_signals), nn.Softmax(dim=-1))
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Sequential(nn.Linear(d_model,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.ttd_head    = nn.Sequential(nn.Linear(d_model,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        for blk in self.temporal_blocks: h = blk(h)
        pooled = self.pool(h.transpose(1,2)).squeeze(-1)
        return torch.sigmoid(self.classifier(pooled)).squeeze(-1)

def focal_loss(pred, target, gamma=2.0, pos_weight=10.0):
    bce = nn.functional.binary_cross_entropy(pred, target.float(), reduction='none')
    pt  = torch.where(target == 1, pred, 1 - pred)
    w   = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
    return (w * (1-pt)**gamma * bce).mean()

SIGNAL_KEYS = ['Greenwald_fraction','Mirnov_norm_btor','Te_peaking_ECE','beta_p','ip_error_normalized',
               'kappa','li','lower_gap','n_equal_1_normalized','q95','radiated_fraction','v_loop','z_error']
MAT_PATH = "/home/claude/SHARD/fig1_data.mat"
SEQ_LEN = 50; PRED_HOR = 0.05; STRIDE = SEQ_LEN // 2

print("Loading data...")
data = sio.loadmat(MAT_PATH)
shots = data['shot'].flatten()
tud   = data['time_until_disrupt'].flatten()
sig_matrix = np.column_stack([data[k].flatten() for k in SIGNAL_KEYS])

X_seqs, y_labels, sids_arr = [], [], []
for shot_id in np.unique(shots):
    mask = shots == shot_id
    sig  = sig_matrix[mask].copy()
    d_arr = tud[mask]
    if len(sig) < SEQ_LEN + 1: continue
    for col in range(sig.shape[1]):
        c = sig[:, col]; nans = np.isnan(c)
        if nans.all():   sig[:, col] = 0.0
        elif nans.any():
            idx = np.where(~nans)[0]
            c[nans] = np.interp(np.where(nans)[0], idx, c[idx])
            sig[:, col] = c
    for i in range(0, len(sig) - SEQ_LEN, STRIDE):
        win = sig[i:i+SEQ_LEN]
        if np.isnan(win).any(): continue
        e_tud = d_arr[i + SEQ_LEN - 1]
        y_labels.append(1.0 if (e_tud > 0 and e_tud <= PRED_HOR) else 0.0)
        X_seqs.append(win)
        sids_arr.append(shot_id)

X = np.array(X_seqs, dtype=np.float32)
y = np.array(y_labels, dtype=np.float32)
sids_arr = np.array(sids_arr)
unique_shots = np.unique(sids_arr)
train_s, test_s = train_test_split(unique_shots, test_size=0.2, random_state=42)
train_s, val_s  = train_test_split(train_s, test_size=0.15, random_state=42)

train_m = np.isin(sids_arr, train_s)
val_m   = np.isin(sids_arr, val_s)
test_m  = np.isin(sids_arr, test_s)

scaler = StandardScaler()
scaler.fit(X[train_m].reshape(-1, 13))
X_tr = scaler.transform(X[train_m].reshape(-1,13)).reshape(X[train_m].shape)
X_va = scaler.transform(X[val_m].reshape(-1,13)).reshape(X[val_m].shape)
X_te = scaler.transform(X[test_m].reshape(-1,13)).reshape(X[test_m].shape)
y_tr, y_va, y_te = y[train_m], y[val_m], y[test_m]

print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}, Test positives: {int(y_te.sum())}")

ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

SEEDS = [42, 7, 123, 256, 999]
EPOCHS = 15
seed_preds = []
seed_aucs  = []
seed_praucs = []

for seed in SEEDS:
    print(f"\n--- Training seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SHARDEncoder()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best_val_auc = 0.0
    best_preds   = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            focal_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(X_va, dtype=torch.float32)).cpu().numpy()
            tp = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy()
        val_auc = roc_auc_score(y_va, vp)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_preds   = tp.copy()

    test_auc   = roc_auc_score(y_te, best_preds)
    test_prauc = average_precision_score(y_te, best_preds)
    print(f"  Best val AUC={best_val_auc:.4f} | Test AUC={test_auc:.4f}, PR-AUC={test_prauc:.4f}")
    seed_preds.append(best_preds)
    seed_aucs.append(test_auc)
    seed_praucs.append(test_prauc)

# Ensemble: average predictions across seeds
ens_preds = np.mean(seed_preds, axis=0)
ens_auc   = roc_auc_score(y_te, ens_preds)
ens_prauc = average_precision_score(y_te, ens_preds)

results = {
    'individual_seeds': {str(s): {'roc_auc': round(a,4), 'pr_auc': round(p,4)}
                         for s, a, p in zip(SEEDS, seed_aucs, seed_praucs)},
    'mean_single_seed': round(float(np.mean(seed_aucs)),4),
    'std_single_seed':  round(float(np.std(seed_aucs)),4),
    'mean_pr_auc':      round(float(np.mean(seed_praucs)),4),
    'std_pr_auc':       round(float(np.std(seed_praucs)),4),
    'ensemble_roc_auc': round(ens_auc,4),
    'ensemble_pr_auc':  round(ens_prauc,4),
}
with open('/home/claude/SHARD/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== ENSEMBLE RESULTS ===")
print(f"Mean single-seed AUC: {results['mean_single_seed']:.4f} ± {results['std_single_seed']:.4f}")
print(f"Mean PR-AUC:          {results['mean_pr_auc']:.4f} ± {results['std_pr_auc']:.4f}")
print(f"Ensemble  ROC-AUC:    {results['ensemble_roc_auc']:.4f}")
print(f"Ensemble  PR-AUC:     {results['ensemble_pr_auc']:.4f}")
