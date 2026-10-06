import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SIGNAL_KEYS = [
    'Greenwald_fraction', 'Mirnov_norm_btor', 'Te_peaking_ECE',
    'beta_p', 'ip_error_normalized', 'kappa', 'li', 'lower_gap',
    'n_equal_1_normalized', 'q95', 'radiated_fraction', 'v_loop', 'z_error'
]

SIGNAL_NAMES = [
    'n/nG (Greenwald)', 'Mirnov/Btor', 'Te peaking (ECE)',
    'βp', 'Ip error', 'κ (elongation)', 'li (inductance)', 'lower gap',
    'n=1 mode', 'q95', 'radiated fraction', 'v_loop', 'z error'
]

def load_shots(mat_path, seq_len=50, prediction_horizon=0.05):
    """
    Load shot data, build sequences for each shot.
    Label: 1 if disruption occurs within prediction_horizon seconds.
    seq_len: number of 5ms timesteps per window (~250ms window)
    """
    data = sio.loadmat(mat_path)
    shots = data['shot'].flatten()
    times = data['time'].flatten()
    tud = data['time_until_disrupt'].flatten()

    # Stack signals
    sig_matrix = np.column_stack([data[k].flatten() for k in SIGNAL_KEYS])

    X_seqs, y_labels, y_ttd, shot_ids = [], [], [], []

    for shot_id in np.unique(shots):
        mask = shots == shot_id
        sig = sig_matrix[mask]
        t = times[mask]
        d = tud[mask]

        # Skip very short shots
        if len(sig) < seq_len + 1:
            continue

        # Impute NaNs per shot with forward fill then mean
        for col in range(sig.shape[1]):
            col_data = sig[:, col]
            nans = np.isnan(col_data)
            if nans.all():
                sig[:, col] = 0.0
            elif nans.any():
                # forward fill
                idx = np.where(~nans)[0]
                col_data[nans] = np.interp(np.where(nans)[0], idx, col_data[idx])
                sig[:, col] = col_data

        # Slide window across shot
        for i in range(0, len(sig) - seq_len, seq_len // 2):
            window = sig[i:i+seq_len]
            if np.isnan(window).any():
                continue
            # Label: is disruption within prediction_horizon of end of window?
            end_tud = d[i + seq_len - 1]
            label = 1.0 if (end_tud > 0 and end_tud <= prediction_horizon) else 0.0
            ttd_val = end_tud if end_tud > 0 else 0.0

            X_seqs.append(window)
            y_labels.append(label)
            y_ttd.append(ttd_val)
            shot_ids.append(shot_id)

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    ttd = np.array(y_ttd, dtype=np.float32)
    shot_ids = np.array(shot_ids)

    print(f"Total sequences: {len(X)}")
    print(f"Disruption sequences: {y.sum():.0f} ({100*y.mean():.1f}%)")
    return X, y, ttd, shot_ids


class DisruptionDataset(Dataset):
    def __init__(self, X, y, ttd):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.ttd = torch.tensor(ttd)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ttd[idx]


def get_dataloaders(mat_path, seq_len=50, batch_size=128, prediction_horizon=0.05, seed=42):
    X, y, ttd, shot_ids = load_shots(mat_path, seq_len, prediction_horizon)

    # Shot-level split (no data leakage across shots)
    unique_shots = np.unique(shot_ids)
    train_shots, test_shots = train_test_split(unique_shots, test_size=0.2, random_state=seed)
    train_shots, val_shots = train_test_split(train_shots, test_size=0.15, random_state=seed)

    train_mask = np.isin(shot_ids, train_shots)
    val_mask = np.isin(shot_ids, val_shots)
    test_mask = np.isin(shot_ids, test_shots)

    # Normalize using training set only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask].reshape(-1, X.shape[-1])).reshape(X[train_mask].shape)
    X_val = scaler.transform(X[val_mask].reshape(-1, X.shape[-1])).reshape(X[val_mask].shape)
    X_test = scaler.transform(X[test_mask].reshape(-1, X.shape[-1])).reshape(X[test_mask].shape)

    print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    train_ds = DisruptionDataset(X_train, y[train_mask], ttd[train_mask])
    val_ds = DisruptionDataset(X_val, y[val_mask], ttd[val_mask])
    test_ds = DisruptionDataset(X_test, y[test_mask], ttd[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler

if __name__ == '__main__':
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        '/home/claude/SHARD/fig1_data.mat'
    )
    print("Data loaders ready.")
