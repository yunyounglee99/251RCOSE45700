import torch

def mixit_loss(source_mix_pairs, est_sources, threshold=10, eps=1e-8):
    """
    source_mix_pairs : (B, 2, T) — 원본 믹스 신호
    est_sources      : (B, M, T) — 분리된 M개의 소스
    threshold        : 최대 SNR (dB) -> tau = 10^(-threshold/10)
    eps              : 수치 안정성을 위한 작은 값
    """
    # print(f'est_sources shape : {est_sources.shape}')
    B, M, T = est_sources.shape
    device = est_sources.device

    # tau 계산 (threshold를 snr_max로 사용)
    tau = 10 ** (-threshold / 10)

    # ALL-ZERO, ALL-ONE 제외한 2^M-2개의 bitmask 생성 (shape (C, M))
    bits = torch.arange(1, 2**M - 1, device=device)
    mask = ((bits[:, None] >> torch.arange(M, device=device)) & 1) \
              .float()  # (C, M)
    C = mask.size(0)

    # est_sources를 (B,1,M,T)로 확장, mask를 (1,C,M,1)로 확장
    est = est_sources.unsqueeze(1)               # (B, 1, M, T)
    mask = mask.unsqueeze(0).unsqueeze(-1)       # (1, C, M, 1)

    # 두 그룹으로 합산
    g1 = (est * mask).sum(dim=2)                 # (B, C, T)
    g2 = (est * (1 - mask)).sum(dim=2)           # (B, C, T)

    # 레퍼런스 신호
    x1 = source_mix_pairs[:, 0].unsqueeze(1)     # (B, 1, T)
    x2 = source_mix_pairs[:, 1].unsqueeze(1)     # (B, 1, T)

    # 레퍼런스 에너지
    E_ref1 = (x1**2).sum(dim=-1)                 # (B, 1)
    E_ref2 = (x2**2).sum(dim=-1)                 # (B, 1)

    # 재구성 오차 에너지
    E_err1 = ((x1 - g1)**2).sum(dim=-1)          # (B, C)
    E_err2 = ((x2 - g2)**2).sum(dim=-1)          # (B, C)

    # 식 (2): 10·log10( E_err + tau·E_ref )
    loss1 = 10 * torch.log10(E_err1 + tau * E_ref1 + eps)
    loss2 = 10 * torch.log10(E_err2 + tau * E_ref2 + eps)

    # 두 믹스 손실 합산 후 최솟값 선택
    loss = loss1 + loss2                         # (B, C)
    best, _ = loss.min(dim=1)                    # (B,)

    return best.mean()

"""
def si_snr(ref, est, eps=1e-8):
  ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
  proj = ((ref * est).sum(dim=-1, keepdim=True) * ref) / ref_energy
  noise = est - proj
  ratio = (proj**2).sum(-1) / ((noise**2).sum(-1) + eps)

  return 10 * torch.log10(ratio + eps)

def mixit_loss(source_mix_pairs, est_sources, threshold=50):
  '''
  source_mix_pairs : (B, 2, T) - 원본 믹스
  est_sources : (B, M, T) - output masks
  '''
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
"""
