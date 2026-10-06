"""
Baseline comparisons for SHARD: Random Forest and LSTM.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from data_loader import get_dataloaders, load_shots
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MAT_PATH = "fig1_data.mat"

def run_rf_baseline():
    """Random Forest on flattened sequences."""
    print("Loading data for RF baseline...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(MAT_PATH, batch_size=512)

    def extract(loader):
        Xs, ys = [], []
        for xb, yb, _ in loader:
            Xs.append(xb.numpy())
            ys.append(yb.numpy())
        X = np.concatenate(Xs, axis=0).reshape(len(np.concatenate(ys)), -1)
        y = np.concatenate(ys)
        return X, y

    X_train, y_train = extract(train_loader)
    X_test, y_test = extract(test_loader)

    print(f"RF training on {X_train.shape[0]} sequences...")
    # Use last timestep features to reduce dimensionality (common baseline)
    # Also try mean features
    n_signals = 13
    seq_len = X_train.shape[1] // n_signals if X_train.shape[1] > n_signals else X_train.shape[1]

    # Use summary stats per signal as features
    X_train_r = X_train.reshape(X_train.shape[0], -1, n_signals)
    X_test_r = X_test.reshape(X_test.shape[0], -1, n_signals)

    X_train_feat = np.concatenate([
        X_train_r.mean(axis=1),
        X_train_r.std(axis=1),
        X_train_r[:, -1, :],  # last timestep
        X_train_r.max(axis=1),
    ], axis=1)
    X_test_feat = np.concatenate([
        X_test_r.mean(axis=1),
        X_test_r.std(axis=1),
        X_test_r[:, -1, :],
        X_test_r.max(axis=1),
    ], axis=1)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train_feat, y_train)
    probs = rf.predict_proba(X_test_feat)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"RF AUC: {auc:.4f}")
    return auc


def run_lstm_baseline():
    """LSTM baseline."""
    import torch
    import torch.nn as nn

    print("Loading data for LSTM baseline...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(MAT_PATH, batch_size=256)

    class LSTMBaseline(nn.Module):
        def __init__(self, n_signals=13, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(n_signals, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, (h, _) = self.lstm(x)
            return torch.sigmoid(self.fc(h[-1])).squeeze(-1)

    device = torch.device('cpu')
    model = LSTMBaseline().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_auc = 0
    best_state = None
    for epoch in range(20):
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                p = model(xb.to(device)).cpu().numpy()
                all_p.extend(p); all_y.extend(yb.numpy())
        val_auc = roc_auc_score(all_y, all_p)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: val AUC {val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            p = model(xb.to(device)).cpu().numpy()
            all_p.extend(p); all_y.extend(yb.numpy())
    test_auc = roc_auc_score(all_y, all_p)
    print(f"LSTM Test AUC: {test_auc:.4f}")
    return test_auc


def plot_comparison(rf_auc, lstm_auc, shard_mean, shard_std, hdl_auc=0.801):
    """Bar chart comparing all methods."""
    methods = ['HDL\n(Published)', 'Random\nForest', 'LSTM', 'SHARD\n(Ours)']
    aucs = [hdl_auc, rf_auc, lstm_auc, shard_mean]
    colors = ['#95a5a6', '#e74c3c', '#e67e22', '#2ecc71']
    errors = [0, 0, 0, shard_std]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, aucs, color=colors, width=0.5, zorder=3,
                  yerr=errors, capsize=5, error_kw={'elinewidth': 2, 'ecolor': '#333'})

    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_title('Disruption Prediction: Method Comparison\n(MIT Alcator C-Mod)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.axhline(hdl_auc, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Published baseline')

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight SHARD
    bars[3].set_edgecolor('#27ae60')
    bars[3].set_linewidth(2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_plot.png")


if __name__ == '__main__':
    rf_auc = run_rf_baseline()
    lstm_auc = run_lstm_baseline()

    # SHARD multi-seed results
    shard_mean = 0.8527
    shard_std = 0.0259

    results = {
        'HDL_published': 0.801,
        'RandomForest': round(rf_auc, 4),
        'LSTM': round(lstm_auc, 4),
        'SHARD_mean': shard_mean,
        'SHARD_std': shard_std
    }
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    plot_comparison(rf_auc, lstm_auc, shard_mean, shard_std)
