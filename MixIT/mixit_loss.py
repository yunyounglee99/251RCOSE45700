import torch

def si_snr(ref, est, eps=1e-8):
  ref_energy = torch.sum(ref ** 2, dim = -1, keepdim = True) + eps
  proj = torch.sum(ref * est, dim = -1, keepdim = True) * ref / ref_energy
  noise = est -proj
  ratio = torch.sum(proj ** 2, -1) / (torch.sum(noise ** 2, -1) + eps)

  return 10 * torch.log10(ratio + eps)

def mixit_loss(source_mix_pairs, est_sources):
  """
  source_mix_pairs : (B, 2, T) - 원본 믹스
  est_sources : (B, M, T) - output masks
  """
  B, M, T = est_sources.shape
  device = est_sources.device

  