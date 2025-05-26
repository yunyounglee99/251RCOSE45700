import torch
import torch.nn as nn
import torchaudio
from torchaudio.models import ConvTasNet

class ConvtasnetSeperator(nn.Module):
  def __init__(
      self,
      num_sources: int = 8,
      enc_kernel_size: int = 16,
      enc_num_feats: int = 512,
      msk_kernel_size: int = 3,
      msk_num_feats: int = 128,
      msk_num_hidden_feats: int = 512,
      msk_num_layers: int = 8,
      msk_num_stacks: int = 3,
      msk_activate: str = 'sigmoid'

  ):
    super().__init__()

    self.convtasnet = ConvTasNet(
      num_sources=num_sources,
      enc_kernel_size=enc_kernel_size,
      enc_num_feats=enc_num_feats,
      msk_kernel_size=msk_kernel_size,
      msk_num_feats=msk_num_feats,
      msk_num_hidden_feats=msk_num_hidden_feats,
      msk_num_layers=msk_num_layers,
      msk_num_stacks=msk_num_stacks,
      msk_activate=msk_activate
    )

  def forward(self, wav):
    """
    mel 변환없이 바로 waveform으로 받음
    wav : (B, T) -> batch * waveform 길이
    return : (B, num_sources, T) -> 분리된 source waveforms
    """
    if wav.dim() == 1:
      wav = wav.unsqueeze(0)    # (1,T)

    est_sources = self.convtasnet(wav)

    return est_sources
