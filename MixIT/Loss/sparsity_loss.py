import torch

def sparsity_loss(est_sources, mixture, eps=1e-8):
  # calculating RMS
  sources_rms = torch.sqrt(torch.mean(est_sources ** 2, dim=-1) + eps)
  mixture_rms = torch.sqrt(torch.mean(mixture ** 2, dim=-1, keepdim=True) + eps)

  # normalize
  norm_sources_rms = sources_rms / (mixture_rms + eps)
  loss = norm_sources_rms.sum(dim=-1).mean()

  return loss