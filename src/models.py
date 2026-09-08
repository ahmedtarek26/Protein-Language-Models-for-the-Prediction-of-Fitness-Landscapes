"""
src/models.py
Regression architectures: Linear (Ridge), Tree (XGBoost), Neural (MLP), and GeoEpiNet.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


# ------------------------------------------------------------------------------
# 1. Baseline Neural MLP
# ------------------------------------------------------------------------------
class ProteinMLP(nn.Module):
    """Deep regularized MLP with LayerNorm and GELU activations."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_mlp(X_tr, y_tr, X_te, device, epochs=45, lr=1e-3, batch_size=64):
    """Trains ProteinMLP and returns predictions on the test partition."""
    model = ProteinMLP(input_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    return preds


# ------------------------------------------------------------------------------
# 2. Advanced Architecture: GeoEpiNet (Geometric Epistasis Network)
# ------------------------------------------------------------------------------
class GaussianRBF(nn.Module):
    """Encodes pairwise 3D physical distances into continuous RBF kernels."""
    def __init__(self, num_kernels=16, d_min=0.0, d_max=25.0):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(d_min, d_max, num_kernels), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(1.0 / ((d_max - d_min) / num_kernels) ** 2), requires_grad=False)

    def forward(self, dists):
        diff = dists.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * (diff ** 2))


class GeoEpiNet(nn.Module):
    """
    Processes multi-mutation sets with 3D structural distance attention bias.
    """
    def __init__(self, input_dim=1536, hidden_dim=128, num_heads=4, num_rbf=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rbf = GaussianRBF(num_kernels=num_rbf)
        self.rbf_proj = nn.Linear(num_rbf, num_heads)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.pool_attn = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_tokens, dist_matrix):
        # x_tokens: (B, k, input_dim), dist_matrix: (B, k, k)
        B, k, _ = x_tokens.shape
        h = self.input_proj(x_tokens)

        # 3D Structural attention bias
        rbf_feat = self.rbf(dist_matrix)
        geom_bias = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)  # (B, heads, k, k)

        q = self.q_proj(h).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_proj(h).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k_t.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores + geom_bias, dim=-1)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, k, -1)

        h = self.norm1(h + self.out_proj(attn_out))
        h = self.norm2(h + self.ffn(h))

        # Dynamic attentive pooling
        pool_w = F.softmax(self.pool_attn(h), dim=1)
        pooled = torch.sum(pool_w * h, dim=1)
        return self.regressor(pooled).squeeze(-1)