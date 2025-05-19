import torch
import torch.nn.functional as F

def diversity_loss(masks):
  B, N, T = masks.shape
  loss = 0.0

  for b in range(B):
    norm_masks = F.normalize(masks[b], p=2, dim=-1)
    sim = torch.matmul(norm_masks, norm_masks.T)

    off_diag = sim - torch.eye(N, device = masks.device)
    loss += off_diag.abs().sum()
  
  return loss    # batch norm 할지 고민중
  