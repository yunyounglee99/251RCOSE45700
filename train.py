import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from performer.performer import PerformerSeperator
from MixIT.mixit_loss import mixit_loss
from MixIT.model import MixITModel
from dataset import MoMDataset
from utils import wav_to_mel
from tqdm import tqdm

SR = 16000
FREQ_BINS = 80
N_MASKS = 8
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MoMDataset(root="./audio", segment_sec = 10.0, sr=SR)
loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

model = MixITModel(
  freq_bins=FREQ_BINS,
  n_masks=N_MASKS,
  performer_dim=256,
  performer_depth=6,
  performer_heads=8,
  performer_nb_features=128,
  performer_max_seq_len=512
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(EPOCHS)):
  model.train()
  total_loss = 0
  for mom_wav, pair_wav in loader:
    mom_wav = mom_wav.to(DEVICE)
    pair_wav = pair_wav.to(DEVICE)
    mom_mel = wav_to_mel(mom_wav, sr=SR)

    masks, _ = model(mom_mel, mom_wav)
    # T_mel -> T로 upsampling (보강 필요)
    masks_wav = torch.nn.functional.interpolate(masks, size=mom_wav.shape[-1], mode='linear')
    est_sources = masks_wav * mom_wav.unsqueeze(1)

    loss = mixit_loss(pair_wav, est_sources)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  print(f'Epoch {epoch+1}/{EPOCHS} Loss : {total_loss/len(loader):.4f}')

torch.save(model.state_dict(), "trackformer.pth")