import torch
import torch.nn.functional as F

def diversity_loss(masks):
  B, N, T = masks.shape
  loss = 0.0

  for b in range(B):
    x = F.normalize(masks[b], p=2, dim=-1)      # (N,T)
    sim = (x @ x.T).abs()                       # (N,N)
    loss += (sim.sum() - sim.diag().sum()) / (N * (N-1))
  return loss / B
  