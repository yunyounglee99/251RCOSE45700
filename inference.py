import torch
import torchaudio
from models.performer import Performer
from MixIT.model import PerformerSeperator
from utils import wav_to_mel, mel_to_wav

SR = 16000
FREQ_BINS = 80
N_MASKS = 8
SEQ_SEC = 10.0
STRIDE_SEC = 2.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sliding_window(wav, segment_len, stride):
  total_len = wav.shape[-1]
  seg_len = int(segment_len * SR)
  stride_len = int(stride * SR)
  starts = list(range(0, total_len - seg_len + 1, stride_len))
  if (total_len - seg_len) % stride_len != 0:
    starts.append(total_len - seg_len)
  return [wav[s : s+seg_len] for s in starts]

model = PerformerSeperator(FREQ_BINS, N_MASKS).to(DEVICE)
model.load_state_dict(torch.load("mixit_performer.pth"))
model.eval()

wav, sr = torchaudio.load("full_song.wav")
wav = wav.mean(0)
wav = torchaudio.functional.resample(wav, sr, SR)
segments = sliding_window(wav, SEQ_SEC, STRIDE_SEC)

all_masks = []
for seg in segments:
  seg = seg.unsqueeze(0).to(DEVICE)
  mel = wav_to_mel(seg, sr = SR)
  with torch.no_grad():
    masks = model(mel)
  all_masks.append(masks.cpu())

masks_full = [torch.nn.functional.interpolate(m.cpu(), size = wav.shape[-1], mode = 'linear') for m in all_masks]

for n in range(N_MASKS):
  est_source = masks_full[0][0, n] * wav
  torchaudio.save(f'track_{n+1}.wav', est_source.unsqueeze(0), SR)