import torch
import torch.nn as nn
import torch.nn.functional as F
from performer.performer import PerformerSeperator
from utils import UpsampleBlock

class MixITModel(nn.Module):
  def __init__(self, 
              freq_bins, 
              n_masks,
              performer_dim = 256,
              performer_depth = 6,
              performer_heads = 8,
              performer_nb_features = 128,
              performer_max_seq_len = 512):
    super().__init__()
    self.seperater = PerformerSeperator(
      freq_bins = freq_bins,
      n_masks = n_masks,
      dim = performer_dim,
      depth = performer_depth,
      heads = performer_heads,
      nb_features = performer_nb_features,
      max_seq_len = performer_max_seq_len
    )
    self.upsample_block = None

  def forward(self, mel, mixture_waveform, device):
    masks = self.seperater(mel)
    B, M, T_mel = masks.shape
    T = mixture_waveform[-1]

    if (self.upsample_block is None) or (self.upsample_block.upsample.weight.shape[-1] != (T // T_mel) * 2):
      self.upsample_block = UpsampleBlock(T_mel, T, M).to(device)

    masks = self.upsample_block(masks)

    masks_4d = masks.unsqueeze(2)     # (B, M, 1, T)
    mel_up = F.interpolate(mel, size=T, mode='linear', align_corners=False)     # (B, F, T)
    mel_sources = masks_4d * mel_up.unsqueeze(1)     # (B, M, F, T)

    return masks, mel_sources
