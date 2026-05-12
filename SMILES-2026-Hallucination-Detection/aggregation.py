from __future__ import annotations

import torch


_N_LAYERS = 8


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.
    """
    n_total, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device
    real_mask = attention_mask.to(device).bool()  # (seq_len,)
    n_real = int(real_mask.sum().item())

    n_select = min(_N_LAYERS, n_total - 1)
    selected = hidden_states[n_total - n_select:]  # (n_select, seq_len, hidden_dim)

    real_hidden = selected[:, real_mask, :]   # (n_select, n_real, hidden_dim)
    layer_means = real_hidden.mean(dim=1)     # (n_select, hidden_dim)

    multi_layer_feat = layer_means.reshape(-1)

    norms = layer_means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = layer_means / norms                              # (n_select, hidden_dim)
    cos_sims = (normalized[:-1] * normalized[1:]).sum(dim=1)      # (n_select - 1,)

    layer_norms = layer_means.norm(dim=1)   # (n_select,)

    log_seq_len = torch.tensor([float(n_real)], dtype=torch.float32, device=device).log1p()

    last_real = hidden_states[-1][real_mask, :]    # (n_real, hidden_dim)
    token_var = last_real.var(dim=0).mean().unsqueeze(0)  # (1,)

    last_pos = int(real_mask.nonzero(as_tuple=False)[-1].item())
    last_token = hidden_states[-1][last_pos]       # (hidden_dim,)

    return torch.cat([
        multi_layer_feat,   # n_select * hidden_dim
        last_token,         # hidden_dim
        cos_sims,           # n_select - 1
        layer_norms,        # n_select
        log_seq_len,        # 1
        token_var,          # 1
    ])


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.
    """
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
