import torch

def si_snr(ref, est, eps=1e-8):
  ref_energy = torch.sum(ref ** 2, dim = -1, keepdim = True) + eps
  proj = torch.sum(ref * est, dim = -1, keepdim = True) * ref / ref_energy
  noise = est -proj
  ratio = torch.sum(proj ** 2, -1) / (torch.sum(noise ** 2, -1) + eps)

  return 10 * torch.log10(ratio + eps)

def mixit_loss(source_mix_pairs, est_sources, threshold=30):
  """
  source_mix_pairs : (B, 2, T) - 원본 믹스
  est_sources : (B, M, T) - output masks
  """
  B, M, T = est_sources.shape
  device = est_sources.device

  # 마스크 탐색 코드
  best_loss = torch.full((B, ), float('inf'), device=device)

  for mask in range(1, 2**M - 1):
    mask_bits = ((mask >> torch.arange(M, device=device)) & 1).float()
    mask_bits = mask_bits.view(1, M, 1)

    group1 = torch.sum(est_sources * mask_bits, dim = 1)
    group2 = torch.sum(est_sources * (1-mask_bits), dim = 1)

    loss1 = -si_snr(source_mix_pairs[:,0], group1)
    loss2 = -si_snr(source_mix_pairs[:,1], group2)

    loss = loss1 + loss2

    best_loss = torch.minimum(best_loss, loss)
  
  return best_loss.mean()