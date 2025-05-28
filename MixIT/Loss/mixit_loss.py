import torch

def si_snr(ref, est, eps=1e-8):
  ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
  proj = ((ref * est).sum(dim=-1, keepdim=True) * ref) / ref_energy
  noise = est - proj
  ratio = (proj**2).sum(-1) / ((noise**2).sum(-1) + eps)

  return 10 * torch.log10(ratio + eps)

def mixit_loss(source_mix_pairs, est_sources, threshold=50):
  """
  source_mix_pairs : (B, 2, T) - 원본 믹스
  est_sources : (B, M, T) - output masks
  """
  B, M, T = est_sources.shape
  device  = est_sources.device

  # 모든 2ᴹ-2 조합 bitmask 한꺼번에 생성 (C, M, 1)
  bits = torch.arange(1, 2**M - 1, device=device)
  mask_mat = ((bits[:, None] >> torch.arange(M, device=device)) & 1) \
              .float().unsqueeze(-1)
  C = mask_mat.size(0)

  # 그룹 합산 (broadcast)
  est = est_sources.unsqueeze(1) 
  g1  = (est *  mask_mat.unsqueeze(0)).sum(dim=2)
  g2  = (est * (1-mask_mat).unsqueeze(0)).sum(dim=2)          # (B,C,T)

  # 두 레퍼런스 vs 모든 조합의 si-SNR 동시 계산
  ref1 = source_mix_pairs[:, 0].unsqueeze(1)
  ref2 = source_mix_pairs[:, 1].unsqueeze(1)

  loss = -si_snr(ref1, g1) - si_snr(ref2, g2) + threshold 

  # 가장 작은 조합 선택 → batch 평균
  best = loss.min(dim=1).values
  return best.mean()