import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention over time steps with interpretability hooks."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # stored for interpretability

    def forward(self, x):
        attn_out, self.attn_weights = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class SignalAttentionBlock(nn.Module):
    """Cross-signal attention: which plasma parameters matter most at each timestep."""
    def __init__(self, n_signals, d_model):
        super().__init__()
        self.signal_attn = nn.Linear(d_model, n_signals)
        self.signal_weights = None  # stored for interpretability

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        self.signal_weights = torch.softmax(self.signal_attn(x), dim=-1)
        return self.signal_weights


class SHARDEncoder(nn.Module):
    """
    SHARD: Signal Harmonics for Anomaly and Reactor Disruption
    
    Architecture:
    - Linear projection of input signals into d_model space
    - Positional encoding for temporal awareness
    - Stack of Temporal Attention Blocks (Transformer encoder)
    - Signal Attention Block for interpretability (which signals drove prediction)
    - Classification head: disruption probability + time-to-disruption
    """
    def __init__(self, n_signals=13, seq_len=100, d_model=64, n_heads=4, 
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.n_signals = n_signals
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_signals, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

        # Transformer encoder layers
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Signal attribution layer (the novel interpretability piece)
        self.signal_attn = SignalAttentionBlock(n_signals, d_model)

        # Global pooling + classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # disruption probability
        )

        # Time-to-disruption regression head
        self.ttd_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # milliseconds until disruption
        )

    def forward(self, x):
        # x: (batch, seq_len, n_signals)
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]

        # Temporal attention stack
        for block in self.temporal_blocks:
            h = block(h)

        # Signal attribution weights (for interpretability)
        sig_weights = self.signal_attn(h)  # (batch, seq_len, n_signals)

        # Pool over time
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # (batch, d_model)

        # Predictions
        disruption_prob = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        ttd = F.relu(self.ttd_head(pooled)).squeeze(-1)

        return disruption_prob, ttd, sig_weights

    def get_signal_attribution(self, x, signal_names):
        """
        Returns mean attention weight per signal across time.
        This is SHARD's interpretability output — which plasma parameter
        drove the disruption prediction.
        """
        self.eval()
        with torch.no_grad():
            _, _, sig_weights = self.forward(x)
        mean_weights = sig_weights.mean(dim=1).mean(dim=0)  # avg over batch and time
        return dict(zip(signal_names, mean_weights.cpu().numpy()))


# Standard 13-signal feature set from DisruptionBench literature
SIGNAL_NAMES = [
    'beta_p',        # normalized plasma pressure
    'li',            # internal inductance  
    'q95',           # safety factor at 95% flux surface
    'n1_mode',       # n=1 MHD mode amplitude
    'n_over_nG',     # Greenwald density fraction
    'lower_gap',     # plasma-wall gap (lower)
    'kappa',         # plasma elongation
    'Ip_error',      # plasma current error
    'loop_voltage',  # loop voltage
    'radiated_power',# total radiated power
    'Wmhd',          # MHD stored energy
    'dWdt',          # rate of change of stored energy
    'v_loop'         # toroidal loop voltage
]

if __name__ == '__main__':
    # Quick sanity check
    model = SHARDEncoder(n_signals=13, seq_len=100, d_model=64, n_heads=4, n_layers=3)
    x = torch.randn(8, 100, 13)  # batch=8, 100 timesteps, 13 signals
    prob, ttd, sig_attn = model(x)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Disruption prob shape: {prob.shape}")
    print(f"Time-to-disruption shape: {ttd.shape}")
    print(f"Signal attention shape: {sig_attn.shape}")
    print("Architecture check passed.")
