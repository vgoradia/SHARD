import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from data_loader import get_dataloaders, SIGNAL_NAMES
from model import SHARDEncoder
import json, os

def focal_loss(pred, target, gamma=2.0, pos_weight=10.0):
    """Focal loss to handle class imbalance (few disruptions vs many normal)."""
    bce = nn.BCELoss(reduction='none')(pred, target)
    pt = torch.where(target == 1, pred, 1 - pred)
    focal = ((1 - pt) ** gamma) * bce
    weights = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
    return (focal * weights).mean()

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y, ttd in loader:
            X, y = X.to(device), y.to(device)
            prob, _, _ = model(X)
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f2 = f1_score(labels, preds, zero_division=0)
    return auc, f2, probs, labels

def train(mat_path='/home/claude/SHARD/fig1_data.mat', epochs=30, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    train_loader, val_loader, test_loader, scaler = get_dataloaders(mat_path, seed=seed)

    model = SHARDEncoder(n_signals=13, seq_len=50, d_model=64, n_heads=4, n_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y, ttd in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            prob, pred_ttd, _ = model(X)
            loss = focal_loss(prob, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_auc, val_f2, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F2: {val_f2:.4f}")
        history.append({'epoch': epoch+1, 'val_auc': val_auc, 'val_f2': val_f2})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), '/home/claude/SHARD/shard_best.pt')

    # Load best and evaluate on test
    model.load_state_dict(torch.load('/home/claude/SHARD/shard_best.pt'))
    test_auc, test_f2, test_probs, test_labels = evaluate(model, test_loader, device)
    print(f"\nFinal Test AUC: {test_auc:.4f} | Test F2: {test_f2:.4f}")

    # Signal attribution on test set
    model.eval()
    all_weights = []
    with torch.no_grad():
        for X, y, ttd in test_loader:
            X = X.to(device)
            _, _, sig_w = model(X)
            all_weights.append(sig_w.mean(dim=1).cpu().numpy())
    mean_weights = np.concatenate(all_weights).mean(axis=0)
    attribution = dict(zip(SIGNAL_NAMES, mean_weights.tolist()))
    print("\nSignal Attribution (mean attention weight):")
    for sig, w in sorted(attribution.items(), key=lambda x: -x[1]):
        print(f"  {sig}: {w:.4f}")

    # Save results
    results = {
        'test_auc': test_auc,
        'test_f2': test_f2,
        'best_val_auc': best_val_auc,
        'signal_attribution': attribution,
        'history': history
    }
    with open('/home/claude/SHARD/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")
    return results

if __name__ == '__main__':
    os.chdir('/home/claude/SHARD')
    train()
