import torch
import torchaudio

def wav_to_mel(wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
  single = False
  if len(wav.shape) == 1:
    wav = wav.unsqueeze(0)
    single = True
  mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels = n_mels, n_fft=n_fft, hop_length=hop_length
  )
  mel = mel_transform(wav)
  if single:
    mel = mel.squeeze(0)

  return mel

def mel_to_wav(mel, sr=16000, n_fft=1024, hop_length=256, n_iter=32):
  single = False
  if len(mel.shape) == 2:
    mel = mel.unsqueeze(0)
    single = True
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