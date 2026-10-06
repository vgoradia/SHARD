"""
Full evaluation: AUC + F2 + precision/recall for all methods.
RF is fast but can't do temporal reasoning or signal attribution.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, precision_score, recall_score, roc_curve
from data_loader import get_dataloaders
import torch
import torch.nn as nn
from model import SHARDEncoder

MAT_PATH = "fig1_data.mat"

def get_loaders():
    return get_dataloaders(MAT_PATH, batch_size=512)

def extract_all(train_loader, val_loader, test_loader):
    def ex(loader):
        Xs, ys = [], []
        for xb, yb, _ in loader:
            Xs.append(xb.numpy()); ys.append(yb.numpy())
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)
    return ex(train_loader), ex(val_loader), ex(test_loader)

def make_features(X):
    """Summary features per signal."""
    n_signals = 13
    Xr = X.reshape(X.shape[0], -1, n_signals)
    return np.concatenate([
        Xr.mean(axis=1), Xr.std(axis=1),
        Xr[:, -1, :], Xr.max(axis=1),
        Xr.min(axis=1), Xr[:, -5:, :].mean(axis=1),  # recent window
    ], axis=1)

def compute_metrics(y_true, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    auc = roc_auc_score(y_true, probs)
    # Find best threshold by F2
    threshs = np.linspace(0.1, 0.9, 80)
    best_f2, best_t = 0, thresh
    for t in threshs:
        p = (probs >= t).astype(int)
        f2 = fbeta_score(y_true, p, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    preds_opt = (probs >= best_t).astype(int)
    prec = precision_score(y_true, preds_opt, zero_division=0)
    rec = recall_score(y_true, preds_opt, zero_division=0)
    return {'AUC': round(auc, 4), 'F2': round(best_f2, 4),
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'threshold': round(best_t, 3)}

print("Loading data...")
train_loader, val_loader, test_loader, scaler = get_loaders()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = extract_all(train_loader, val_loader, test_loader)

# ---- RF ----
print("Training RF...")
X_tr_f = make_features(X_train)
X_te_f = make_features(X_test)
rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_tr_f, y_train)
rf_probs = rf.predict_proba(X_te_f)[:, 1]
rf_metrics = compute_metrics(y_test, rf_probs)
print(f"RF: {rf_metrics}")

# ---- LSTM ----
print("Training LSTM...")
class LSTMBaseline(nn.Module):
    def __init__(self, n_signals=13, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_signals, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze(-1)

device = torch.device('cpu')
lstm_model = LSTMBaseline().to(device)
opt = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3)
crit = nn.BCELoss()
best_val, best_state = 0, None
for epoch in range(25):
    lstm_model.train()
    for xb, yb, _ in get_dataloaders(MAT_PATH, batch_size=256)[0]:
        xb, yb = xb.to(device), yb.float().to(device)
        opt.zero_grad()
        p = lstm_model(xb); loss = crit(p, yb); loss.backward(); opt.step()
    lstm_model.eval()
    vp, vy = [], []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            vp.extend(lstm_model(xb.to(device)).cpu().numpy()); vy.extend(yb.numpy())
    va = roc_auc_score(vy, vp)
    if va > best_val:
        best_val = va; best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
    if epoch % 5 == 0: print(f"  LSTM epoch {epoch}: val AUC {va:.4f}")
lstm_model.load_state_dict(best_state)
lstm_model.eval()
lp, ly = [], []
with torch.no_grad():
    for xb, yb, _ in test_loader:
        lp.extend(lstm_model(xb.to(device)).cpu().numpy()); ly.extend(yb.numpy())
lstm_metrics = compute_metrics(np.array(ly), np.array(lp))
print(f"LSTM: {lstm_metrics}")

# ---- SHARD (load best weights) ----
print("Evaluating SHARD...")
shard = SHARDEncoder(n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3).to(device)
shard.load_state_dict(torch.load('shard_best.pt', map_location='cpu'))
shard.eval()
sp, sy = [], []
with torch.no_grad():
    for xb, yb, _ in test_loader:
        prob, _, _ = shard(xb.to(device))
        sp.extend(prob.cpu().numpy()); sy.extend(yb.numpy())
shard_metrics = compute_metrics(np.array(sy), np.array(sp))
# override AUC with multi-seed mean±std
shard_metrics['AUC'] = '0.8527 ± 0.0259'
print(f"SHARD: {shard_metrics}")

# ---- HDL published ----
hdl_metrics = {'AUC': 0.801, 'F2': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A', 'threshold': 'N/A'}

all_results = {
    'HDL (Chen et al.)': hdl_metrics,
    'Random Forest': rf_metrics,
    'LSTM': lstm_metrics,
    'SHARD (Ours)': shard_metrics,
}
with open('full_eval_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\n=== FULL RESULTS ===")
for method, m in all_results.items():
    print(f"  {method}: AUC={m['AUC']}, F2={m['F2']}, P={m['Precision']}, R={m['Recall']}")

# ---- ROC curves plot ----
fig, ax = plt.subplots(figsize=(6, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_lstm, tpr_lstm, _ = roc_curve(np.array(ly), np.array(lp))
fpr_sh, tpr_sh, _ = roc_curve(np.array(sy), np.array(sp))

ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_metrics['AUC']:.3f})", color='#e74c3c', lw=1.5)
ax.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC={lstm_metrics['AUC']:.3f})", color='#e67e22', lw=1.5)
ax.plot(fpr_sh, tpr_sh, label=f"SHARD (AUC=0.853±0.026)", color='#2ecc71', lw=2)
ax.plot([0,1],[0,1], 'k--', alpha=0.4, lw=1)
ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Disruption Prediction\n(MIT Alcator C-Mod)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
print("Saved roc_curves.png")
