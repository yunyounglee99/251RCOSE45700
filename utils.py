import torch
import librosa
import torchaudio
import torch.nn as nn

def pitch_shift(waveform, sample_rate, n_steps):
  waveform_np = waveform.squeeze().cpu().numpy()
  shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)
  return torch.from_numpy(shifted).unsqueeze(0)

def time_stretch(waveform, stretch_factor):
  waveform_np = waveform.squeeze().cpu().numpy()
  stretched = librosa.effects.time_stretch(waveform_np, rate=stretch_factor)
  return torch.from_numpy(stretched).unsqueeze(0)

def wav_to_mel(wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
  single = False
  if len(wav.shape) == 1:
    wav = wav.unsqueeze(0)
    single = True
  
  if wav.dim() == 3 and wav.shape[1] == 1:
        wav = wav.squeeze(1)

  mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels = n_mels, n_fft=n_fft, hop_length=hop_length
  ).to(wav.device)

  mel = mel_transform(wav)

  if mel.dim() == 3:
     mel = mel.unsqueeze(1)

  if single:
    mel = mel.squeeze(0)

  return mel

def mel_to_wav(mel, sr=16000, n_fft=1024, hop_length=256, n_iter=32):
  single = False
  if len(mel.shape) == 2:
    mel = mel.unsqueeze(0)
    single = True

  if mel.dim() == 4 and mel.shape[1] == 1:
        mel = mel.squeeze(1)

  mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels=mel.shape[1], n_fft=n_fft, hop_length=hop_length
  )
  inv_mel_basis = mel_transform.mel_basis.pinverse()
  spec = torch.matmul(inv_mel_basis, mel)
  spec = torch.clamp(spec, min=1e-10)
  griffin_lim = torchaudio.transforms.GriffinLim(
    n_fft=n_fft, hop_length=hop_length, n_iter=n_iter
  )
  wav = griffin_lim(spec)
  if single:
    wav = wav.squeeze(0)
  return wav

class UpsampleBlock(nn.Module):
   def __init__(self, in_len: int, out_len: int, channels: int, max_scale_per_layer: int = 8):
      super().__init__()
      if out_len %in_len != 0:
         raise ValueError("out_len must be integer multiple of in_len")
      
      total_scale = out_len // in_len
      factors = []
      s = total_scale
      while s > 1:
         f = min(max_scale_per_layer, s if s < max else max_scale_per_layer)
         if s % f != 0:
            f = 2
         factors.append(f)
         s //= f

      layers = []
      cur_len = in_len
      for f in factors:
         next_len = cur_len * f
         k, st, pad = f * 2, f, f //2
         layers.append(
            nn.ConvTranspose1d(
               channels, channels, kernel_size=k, stride = st, padding = pad, bias = False
            )
         )
         cur_len = next_len
      self.net = nn.Sequential(*layers)

      for m in self.net:
         nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

   def forward(self, x):
      return self.net(x)


"""
class UpsampleBlock(nn.Module):
    def __init__(self, in_len, out_len, channels):
        super().__init__()
        scale = out_len // in_len
        kernel_size = scale * 2
        stride = scale
        padding = scale // 2
        expected_len = (in_len - 1) * stride - 2 * padding + kernel_size
        output_padding = out_len - expected_len
        assert output_padding >= 0 and output_padding < stride, \
            f"Invalid output_padding: {output_padding}"
        
        self.upsample = nn.ConvTranspose1d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

    def forward(self, x):
        # x: (B, N, T_mel)
        return self.upsample(x)

"""

import torch

def si_snr(ref, est, eps=1e-8):
    """
    Scale-Invariant SNR  (B, T) or (B, 1, T) 텐서 지원
    """
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    # projektion
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref ** 2).sum(-1, keepdim=True).clamp_min(eps)
    noise = est - proj
    ratio = (proj ** 2).sum(-1) / (noise ** 2).sum(-1).clamp_min(eps)
    return 10 * torch.log10(ratio.clamp_min(eps))

def si_snr_improvement(mixture, est, ref):
    """
    mixture : (B, T) (2소스 혼합 or 원 믹스)
    est     : (B, T) (모델이 예측한 단일/복수 그룹)
    ref     : (B, T) (GT 단일 소스) ─ MixIT 학습에선 실제로는 없음.
    반환     : SI-SNR(est, ref) − SI-SNR(mixture, ref)
    """
    return si_snr(ref, est) - si_snr(ref, mixture)