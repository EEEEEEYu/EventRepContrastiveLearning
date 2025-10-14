import torch

def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR between two batches of images/tensors.

    Args:
        x, y: tensors of shape (B, C, H, W) or (B, ...), values in [0, data_range].
        data_range: The maximum possible value (1.0 if normalized, 255 for 8-bit).

    Returns:
        Scalar PSNR averaged over the batch (torch.Tensor).
    """
    assert x.shape == y.shape, "Input shapes must match"
    mse = torch.mean((x - y) ** 2, dim=list(range(1, x.ndim)))  # per-sample MSE
    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))   # avoid log(0)
    return psnr.mean()  # average over batch
