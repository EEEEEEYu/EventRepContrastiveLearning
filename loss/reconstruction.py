import torch.nn.functional as F

def recon_loss(pred, gt):
    return F.mse_loss(pred, gt, reduction='mean')