import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import Performer
import timm

def resize_mel_for_audiomae(mel):
  """mel을 AudioMAE 입력 크기로 조정"""
  B, C, H, W = mel.shape  # (4, 1, 80, 126)
  target_H, target_W = 1024, 128
  print(f'B : {B} C : {C} H : {H}, W : {W}')
  
  # Width만 먼저 조정 (시간 축)
  if W != target_W:
      mel = F.interpolate(mel, size=(target_H, target_W), mode='bilinear', align_corners=False)
      print(f'after interpolation : {mel.shape}')

      B, C, H, W = mel.shape
      # (4, 1, 80, 126) → (4, 1, 80, 128)
  
  # Height를 패딩으로 확장 (주파수 축)
  if H < target_H:
      padding = target_H - H
      mel = F.pad(mel, (0, 0, 0, padding), mode='constant', value=0)
      print(f'after padding : {mel.shape}')
      # (4, 1, 80, 128) → (4, 1, 1024, 128)
  
  return mel  # (4, 1, 1024, 128)


class PerformerSeperator(nn.Module):
  def __init__(
      self,
      freq_bins : int,     # Mel 대역 수
      n_masks : int,      # 출력 마스크 개수 M
      dim : int = 768,     # 모델 차원
      depth : int = 6,     #레이어 수
      heads : int = 8,     #어텐션 헤드 수
      nb_features : int = 128,     #random feature 수
      max_seq_len : int = 1024,     # 최대 시퀀스 길이      
  ):
    super().__init__()

    # Audio MAE encoder
    self.encoder = timm.create_model(
      'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
      pretrained = True,
      num_classes = 0
    )
    self.encoder.eval()

    self.performer = Performer(
      dim = dim,
      depth = depth,
      heads = heads,
      dim_head = dim//heads
    )

    self.to_mask = nn.Linear(dim, n_masks)

  def forward(self, mel: torch.Tensor):
    """
    mel : (B, F, T)
    returns masks : (B, M, T)
    """
    mel = resize_mel_for_audiomae(mel)

    with torch.no_grad():
      x = self.encoder.patch_embed(mel)

      if hasattr(self.encoder, 'pos_embed'):
         x = x + self.encoder.pos_embed[:, 1:]
         print('successfully add pos_embed!')

      x = self.encoder.pos_drop(x)

    print(f'after mae shape : {x.shape}')

    x = self.performer(x)
    print(f'after performer shape : {x.shape}')

    mask_logits = self.to_mask(x)
    print(f'after mask shape : {mask_logits.shape}')
    masks = torch.sigmoid(mask_logits)
    
    return masks.permute(0, 2, 1)