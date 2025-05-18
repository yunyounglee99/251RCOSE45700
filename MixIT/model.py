import torch
import torch.nn as nn
import torch.nn.functional as F
from performer.performer import PerformerSeperator

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

  def forward(self, mel, mixture_waveform):
    masks = self.seperater(mel)
    masks_4d = masks.unsqueeze(2)     # (B, M, 1, T)
    mel_sources = masks_4d * mel.unsqueeze(1)     # (B, M, F, T)

    return masks, mel_sources
