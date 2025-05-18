import torch
import torchaudio
import random
import glob
from torch.utils.data import Dataset

class MoMDataset(Dataset):
  def __init__(self, root, segment_sec=10.0, sr=16000):
    self.files = glob.glob(f"{root}/**/*.wav", recursive = True)
    self.seg_len = int(segment_sec * sr)
    self.sr = sr

  def __len__(self):
    return len(self.files) ** 2
  
  def _load_random_seg(self, path):
    wav, sr = torchaudio.load(path)
    if sr != self.sr:
      wav = torchaudio.functional.resample(wav, sr, self.sr)
    if wav.size(1) < self.seg_len:
      wav = torch.nn.functional.pad(wav, (0, self.seg_len - wav.size(1)))
    start = random.randint(0, max(0, wav.size(1) - self.seg_len))

    return wav[:, start:start+self.seg_len]
  
  def __getitem__(self, idx):
    idx1 = random.randint(0, len(self.files) - 1)
    idx2 = random.randint(0, len(self.files) - 1)
    x1 = self._load_random_seg(self.files[idx1])
    x2 = self._load_random_seg(self.files[idx2])
    mom = x1 + x2


    return mom.squeeze(0), torch.stack([x1.squeeze(0), x2.squeeze(0)])