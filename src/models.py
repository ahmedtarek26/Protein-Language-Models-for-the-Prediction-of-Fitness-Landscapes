"""
src/models.py
Regression heads: Ridge, XGBoost, Deep MLP, and the 3D-contact-aware GeoEpiNet.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ProteinMLP(nn.Module):
    """Deep regularized MLP with LayerNorm, GELU activations, and Dropout."""
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


def train_torch_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, device: torch.device, epochs: int = 45):
    """Trains ProteinMLP and returns out-of-fold test predictions."""
    model = ProteinMLP(input_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=64, shuffle=True
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


class GeoEpiNet(nn.Module):
    """
    Geometric Epistasis Network (GeoEpiNet).
    Combines a high-capacity residual backbone with pairwise 3D-distance-weighted
    Hadamard interaction terms to model higher-order epistasis.
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        # 1. Main residual pathway across full ESM3 delta representation
        self.base_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 2. Epistatic pairwise projection module
        self.epistatic_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        self.epistatic_head = nn.Linear(hidden_dim // 4, hidden_dim // 2)

        # 3. Readout head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.epistatic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_tokens: torch.Tensor, dist_matrix: torch.Tensor = None):
        """
        x_tokens: (B, k, input_dim) or (B, input_dim)
        dist_matrix: (B, k, k) pairwise C-alpha Euclidean distances
        """
        if x_tokens.dim() == 2:
            x_tokens = x_tokens.unsqueeze(1)

        B, k, d = x_tokens.shape
        x_mean = x_tokens.mean(dim=1)
        base_h = self.base_stream(x_mean)

        # Calculate contact-weighted epistasis for multi-point variants (k >= 2)
        if k >= 2 and dist_matrix is not None:
            z_epi = self.epistatic_proj(x_tokens)
            # Physical spatial decay kernel: exp(-D_ij / 8.0 Angstroms)
            contact_weight = torch.exp(-dist_matrix / 8.0).unsqueeze(-1)
            pairs = torch.einsum('bik,bjk->bijk', z_epi, z_epi) * contact_weight

            # Zero out diagonal self-interactions (i == j)
            diag_mask = torch.ones(k, k, device=x_tokens.device) - torch.eye(k, device=x_tokens.device)
            pairs = pairs * diag_mask.unsqueeze(0).unsqueeze(-1)

            epi_h = pairs.sum(dim=(1, 2)) / max(1, k * (k - 1))
            epi_out = self.epistatic_head(epi_h)
            total_h = base_h + self.epistatic_scale * epi_out
        else:
            total_h = base_h

        return self.head(total_h).squeeze(-1)


def train_geoepinet(
    df, X_esm3: np.ndarray, y: np.ndarray,
    idx_tr: np.ndarray, idx_te: np.ndarray,
    coords: np.ndarray, max_seq_len: int,
    device: torch.device, epochs: int = 50
):
    """Batches variants and trains GeoEpiNet with 3D structural distance matrices."""
    def prepare_tensors(indices):
        sub_df = df.iloc[indices]
        k_max = max(1, max(sub_df["num_mutations"]))
        tokens_list, dists_list = [], []

        for i_row, row in sub_df.iterrows():
            positions = row["extracted_positions"][:k_max]
            k_curr = len(positions)
            vec = X_esm3[i_row]

            t_pad = np.zeros((k_max, 1536), dtype=np.float32)
            d_mat = np.zeros((k_max, k_max), dtype=np.float32)

            for ki in range(k_curr):
                t_pad[ki] = vec
                p_i = min(positions[ki] - 1, len(coords) - 1)
                for kj in range(k_curr):
                    p_j = min(positions[kj] - 1, len(coords) - 1)
                    d_mat[ki, kj] = np.linalg.norm(coords[p_i] - coords[p_j])

            tokens_list.append(t_pad)
            dists_list.append(d_mat)

        return (
            torch.tensor(np.array(tokens_list), dtype=torch.float32, device=device),
            torch.tensor(np.array(dists_list), dtype=torch.float32, device=device)
        )

    bx_tr, bd_tr = prepare_tensors(idx_tr)
    by_tr = torch.tensor(y[idx_tr], dtype=torch.float32, device=device)
    bx_te, bd_te = prepare_tensors(idx_te)

    model = GeoEpiNet(input_dim=1536).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(bx_tr, bd_tr)
        loss = criterion(pred, by_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(bx_te, bd_te).cpu().numpy()
    return preds