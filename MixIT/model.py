import torch
import torch.nn as nn
import torch.nn.functional as F
from models.performer import PerformerSeperator
from models.convtasnet import ConvtasnetSeperator
# from utils import UpsampleBlock

class MixITModel(nn.Module):
  def __init__(
      self,
      model_type: str,
      # <performer> 
      freq_bins, 
      n_masks,
      performer_dim = 256,
      performer_depth = 6,
      performer_heads = 8,
      performer_nb_features = 128,
      performer_max_seq_len = 512,
      # <convtasnet>
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
    self.model_type = model_type.lower()
    if self.model_type == 'performer':
      self.seperator = PerformerSeperator(
        freq_bins = freq_bins,
        n_masks = n_masks,
        dim = performer_dim,
        depth = performer_depth,
        heads = performer_heads,
        nb_features = performer_nb_features,
        max_seq_len = performer_max_seq_len
      )

    elif self.model_type == 'convtasnet':
      self.seperator = ConvtasnetSeperator(
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
    else:
      raise ValueError(f"Unknown model type : {self.model_type} model type should be performer or convtasnet")

  def forward(self, mel, mixture_waveform, device):
    if self.model_type == 'performer':
      masks = self.seperator(mel)
      B, M, T_mel = masks.shape
      T = mixture_waveform.shape[-1]

      masks = F.interpolate(
        masks,
        size = T,
        mode = 'linear',
        align_corners=False
      )

      return masks
    if self.model_type == 'convtasnet':
      est_sources = self.seperator(mixture_waveform)
      return est_sources
