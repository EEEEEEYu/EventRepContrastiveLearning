import torch
import torch.nn.functional as F

def _ensure_bkC(x, expect_last_dim_C=True):
    """
    Utility:
      - If 'x' is global embeddings with shape (B, C, K) or (B, K, C),
        return as (B, K, C).
      - If 'x' is dense maps with shape (B, K, C, H, W) or (B, C, H, W, K),
        return as (B, K, C, H, W).

    Args:
        x: torch.Tensor
        expect_last_dim_C: if True, assumes C is last for globals (B, K, C) else tries (B, C, K)
    """
    if x.dim() == 3:
        B, A, Bdim = x.shape
        # try to guess whether it's (B, C, K) or (B, K, C)
        # Heuristic: treat the shorter of A/Bdim as K (often small: #views 2..8)
        if A <= Bdim:
            # could be (B, K, C) already
            return x
        else:
            # swap to (B, K, C)
            return x.transpose(1, 2).contiguous()
    elif x.dim() == 5:
        # Either (B, K, C, H, W) or (B, C, H, W, K)
        if x.shape[1] <= x.shape[-1]:
            # Likely (B, K, C, H, W)
            return x
        else:
            # (B, C, H, W, K) -> (B, K, C, H, W)
            return x.permute(0, 4, 1, 2, 3).contiguous()
    else:
        raise ValueError(f"Unexpected tensor dims: {x.shape}")


def global_multipos_info_nce(
    z,
    temperature: float = 0.07,
    eps: float = 1e-8,
):
    """
    Multi-positive global InfoNCE for pooled embeddings.

    Args:
        z: Tensor of shape (B, K, C) or (B, C, K).
           B = batch size (segments), K = #views/representations, C = embed dim.
           We'll align to (B, K, C) internally.
        temperature: softmax temperature.
        eps: numerical stability for norms.

    Returns:
        loss: scalar tensor
        stats: dict with 'pos_sim_mean', 'neg_sim_mean'
    """
    z = _ensure_bkC(z)                  # (B, K, C)
    B, K, C = z.shape
    z = F.normalize(z, dim=-1, eps=eps) # L2-normalize

    # Build anchors and candidates
    # Anchors: every (b, k) pair -> B*K anchors, shape (B*K, C)
    anchors = z.reshape(B * K, C)       # (BK, C)

    # Candidates: all (b', k') pairs
    candidates = anchors                # (BK, C)

    # Similarity matrix (BK x BK)
    logits = (anchors @ candidates.T) / temperature  # cosine since normalized

    # Build multi-positive mask: positives are same sample (same b), different view (k' != k)
    # idx mapping: flat_idx = b*K + k
    arange = torch.arange(B*K, device=z.device)
    b_idx = arange // K
    k_idx = arange % K

    same_sample = (b_idx.unsqueeze(1) == b_idx.unsqueeze(0))   # (BK, BK)
    same_view   = (k_idx.unsqueeze(1) == k_idx.unsqueeze(0))
    pos_mask    = same_sample & (~same_view)                   # true for positives
    neg_mask    = ~same_sample                                 # negatives from other samples

    # For InfoNCE with multi-positives, we compute:
    #  - numerator: sum over all positives' exp(sim)
    #  - denominator: sum over all candidates except self (include pos + neg)
    #  (exclude self-pair along the diagonal)
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # stability
    exp_logits = torch.exp(logits)

    # zero-out self matches
    eye = torch.eye(B*K, device=z.device, dtype=torch.bool)
    exp_logits = exp_logits.masked_fill(eye, 0.0)

    pos_sum = (exp_logits * pos_mask).sum(dim=1)  # (BK,)
    denom   = exp_logits.sum(dim=1)               # (BK,)

    # avoid log(0) by clipping
    pos_sum = torch.clamp(pos_sum, min=eps)
    denom   = torch.clamp(denom,   min=eps)

    loss = -torch.log(pos_sum / denom).mean()

    with torch.no_grad():
        # Track average positive/negative cosine (pre-temperature)
        sim = anchors @ candidates.T
        pos_sim_mean = sim[pos_mask].mean() if pos_mask.any() else torch.tensor(0., device=z.device)
        neg_sim_mean = sim[neg_mask].mean() if neg_mask.any() else torch.tensor(0., device=z.device)

    return {
        'loss': loss,
        'pos_sim_mean': pos_sim_mean.item(), 
        'neg_sim_mean': neg_sim_mean.item(),
    }


def dense_info_nce(
    maps,
    temperature: float = 0.07,
    include_spatial_negatives: bool = True,
    neighborhood: int = 0,
    eps: float = 1e-8,
):
    """
    Dense (pixel/patch) multi-positive InfoNCE across views.

    Args:
        maps: Tensor of shape (B, K, C, H, W) or (B, C, H, W, K).
              We'll align to (B, K, C, H, W) internally.
              For each sample b and location (u,v), all views k=0..K-1 form positives.
        temperature: softmax temperature for dense head.
        include_spatial_negatives:
            - If True, negatives include *all spatial tokens from other samples*
              and *other locations in the batch* (stronger, but heavier).
            - If False, negatives include only *same spatial index* from other samples.
        neighborhood: allow positives to match within a (2r+1)x(2r+1) window across views
            (robust to slight misalignment). 0 = exact (u,v) only.
        eps: numerical stability.

    Returns:
        loss: scalar tensor
        stats: dict with 'pos_sim_mean', 'neg_sim_mean'
    """
    maps = _ensure_bkC(maps)                  # (B, K, C, H, W)
    B, K, C, H, W = maps.shape
    maps = F.normalize(maps, dim=2, eps=eps)  # normalize on channel dim

    # Flatten spatial to tokens per view
    HW = H * W
    feats = maps.reshape(B, K, C, HW)         # (B, K, C, HW)

    # We'll treat each (b, k, l) as an anchor, where l in [0..HW-1]
    # To keep memory manageable, we compute by location in chunks if needed.
    # Here we do a single pass; consider chunking l if HW is large.

    # Build candidate bank
    # If include_spatial_negatives=True: candidates are all (b', k', l')
    # Else: candidates are (b', k', l) same spatial index only (lighter)
    if include_spatial_negatives:
        cand = feats.permute(0, 1, 3, 2).reshape(B*K*HW, C)  # (B*K*HW, C)
    else:
        # Restrict to same spatial index later by selecting rows
        cand = feats.permute(0, 1, 3, 2).reshape(B*K*HW, C)  # still same shape; we'll mask by index

    # Anchors: all (b, k, l)
    anch = feats.permute(0, 1, 3, 2).reshape(B*K*HW, C)      # (B*K*HW, C)

    # Similarity matrix (can be huge). We'll compute row-wise to save memory.
    # For clarity and completeness, below is the full matmul; if OOM, switch to chunked sim.
    logits = (anch @ cand.T) / temperature                   # (BKHW, BKHW)
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # stability
    exp_logits = torch.exp(logits)

    # Build index helpers
    arange = torch.arange(B*K*HW, device=maps.device)
    b_idx = (arange // (K*HW))
    k_idx = (arange // HW) % K
    l_idx = arange % HW

    # Candidate indices for each column
    arange_c = torch.arange(B*K*HW, device=maps.device)
    b_idx_c = (arange_c // (K*HW))
    k_idx_c = (arange_c // HW) % K
    l_idx_c = arange_c % HW

    # Positive mask: same sample (b), different view (k'!=k),
    # and location match within neighborhood window (on the 2D grid).
    # Build (u,v) from l index.
    u = l_idx // W
    v = l_idx % W
    u_c = l_idx_c // W
    v_c = l_idx_c % W

    same_b  = (b_idx.unsqueeze(1) == b_idx_c.unsqueeze(0))
    diff_k  = (k_idx.unsqueeze(1) != k_idx_c.unsqueeze(0))

    if neighborhood == 0:
        loc_match = (u.unsqueeze(1) == u_c.unsqueeze(0)) & (v.unsqueeze(1) == v_c.unsqueeze(0))
    else:
        du = (u.unsqueeze(1) - u_c.unsqueeze(0)).abs()
        dv = (v.unsqueeze(1) - v_c.unsqueeze(0)).abs()
        loc_match = (du <= neighborhood) & (dv <= neighborhood)

    pos_mask = same_b & diff_k & loc_match

    # Negatives: everything else except self
    eye = torch.eye(B*K*HW, device=maps.device, dtype=torch.bool)
    exp_logits = exp_logits.masked_fill(eye, 0.0)

    # If include_spatial_negatives=False, constrain negatives to same l only (other samples/views)
    if not include_spatial_negatives:
        same_l = (l_idx.unsqueeze(1) == l_idx_c.unsqueeze(0))
        exp_logits = exp_logits * same_l  # zero out other positions entirely

    pos_sum = (exp_logits * pos_mask).sum(dim=1)
    denom   = exp_logits.sum(dim=1)

    pos_sum = torch.clamp(pos_sum, min=eps)
    denom   = torch.clamp(denom,   min=eps)

    loss = -torch.log(pos_sum / denom).mean()

    with torch.no_grad():
        # track raw cosine (pre-temperature) for diagnostics
        # compute on a random subset to avoid OOM
        sample_idx = torch.randperm(B*K*HW, device=maps.device)[:min(8192, B*K*HW)]
        anch_s = anch[sample_idx]                # (S, C)
        # for stats, compute sim against *matching* positives and random negatives
        sim_full = anch_s @ cand.T               # (S, BKHW)
        pos_rows = pos_mask[sample_idx]          # (S, BKHW)
        neg_rows = (~pos_rows) & (~torch.eye(cand.shape[0], device=maps.device, dtype=torch.bool)[sample_idx][:, :])
        pos_sim_mean = sim_full[pos_rows].mean() if pos_rows.any() else torch.tensor(0., device=maps.device)
        neg_sim_mean = sim_full[neg_rows].mean() if neg_rows.any() else torch.tensor(0., device=maps.device)

    return {
        'loss': loss,
        'pos_sim_mean': pos_sim_mean.item(), 
        'neg_sim_mean': neg_sim_mean.item()
    }
