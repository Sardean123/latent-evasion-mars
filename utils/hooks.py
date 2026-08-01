from typing import Dict, List, Optional

import torch


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def projection_hook(
    w: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    margin: float,
    eps: float = 1e-12,
):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))

        return replace_hidden(output, h_mod)

    return hook


def gated_projection_hook(
    w: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    margin: float,
    gate_c: float,
    stats: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 1e-12,
):
    """CLE-P*: project only the positions whose probe score is above the gate threshold c.

    Plain CLE-P (projection_hook) moves EVERY position to score exactly -m. For an activation
    that already reads harmless -- raw score below -m -- `score = raw + m` is negative and the
    update pushes it back UP toward the boundary, i.e. CLE-P actively steers *toward* refusal
    activations that never needed steering. The gate makes the intervention one-sided: any
    position with raw score <= c passes through untouched.

    gate_c is in raw probe-score units (w·h + b), so:
        c = 0     -- steer only what the probe reads as harmful (the decision boundary)
        c = -m    -- steer only what has not already reached the target (a plain ReLU on the
                     projection amount; never moves an activation backwards)
        c = -inf  -- recovers projection_hook exactly

    Optional `stats` accumulates on-device counters ("fired", "total") so a caller can report
    what fraction of positions the gate actually let through; no host sync inside the hook.
    """
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        c_local = torch.as_tensor(gate_c, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        gate = (raw_score > c_local).to(h.dtype)
        h_mod = h - (gate * (beta * (score / w_norm_sq)) * w_local.view(1, 1, -1))

        if stats is not None:
            stats["fired"] = stats.get("fired", torch.zeros((), device=h.device)) + gate.sum()
            stats["total"] = stats.get("total", torch.zeros((), device=h.device)) + gate.numel()

        return replace_hidden(output, h_mod)

    return hook


def pipeline_delta_hook(
    w: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    margin: float,
    layer_idx: int,
    delta_store: Dict[int, torch.Tensor],
    eps: float = 1e-12,
):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))
        delta_store[layer_idx] = (h_mod[:, -1, :] - h[:, -1, :]).detach().to(dtype=torch.float32)

        return replace_hidden(output, h_mod)

    return hook


def add_hook(delta: torch.Tensor):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        if delta.ndim == 1:
            v = delta.view(1, 1, -1)
        elif delta.ndim == 2:
            if delta.shape[0] != h.shape[0]:
                raise ValueError(f"Batch delta size mismatch: delta batch={delta.shape[0]} hidden batch={h.shape[0]}")
            v = delta.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected delta shape: {tuple(delta.shape)}")

        h_mod = h + v.to(device=h.device, dtype=h.dtype)
        return replace_hidden(output, h_mod)

    return hook


def remove_hooks(handles: List[object]) -> None:
    for handle in handles:
        handle.remove()
