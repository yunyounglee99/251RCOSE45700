import torch
import torch.nn as nn
from performer_pytorch import Performer

class PerformerSeperator(nn.Module):
  def __init__(
      self,
      freq_bins : int,     # Mel 대역 수
      n_masks : int,      # 출력 마스크 개수 M
      dim : int = 256,     # 모델 차원
      depth : int = 6,     #레이어 수
      heads : int = 8,     #어텐션 헤드 수
      nb_features : int = 128,     #random feature 수
      max_seq_len : int = 512     # 최대 시퀀스 길이
  ):
    super().__init__()

    self.performer = Performer(
      dim = dim,
      depth = depth,
      heads = heads,
      causal = False,
      nb_features = nb_features,
      generalized_attention = False,
      kernel_fn = None,
      max_seq_len = max_seq_len
    )
    self.to_mask = nn.Linear(dim, n_masks)

    def forward(self, mel: torch.Tensor):
      """
      mel : (B, F, T)
      returns masks : (B, M, T)
      """
      