import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.performer import PerformerSeperator
from MixIT.Loss.mixit_loss import mixit_loss
from MixIT.Loss.diversity_loss import diversity_loss
from MixIT.Loss.sparsity_loss import sparsity_loss
from MixIT.model import MixITModel
from dataloader import MoMDataset
from utils import wav_to_mel
from tqdm import tqdm

def train(
    root,
    lr,
    batch_size,
    epochs,
    segment_sec:float=10.0,
    sr=16000,
    # <performer> 
    freq_bins=80, 
    n_masks=8,
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
    msk_activate: str = 'sigmoid',
    lambda_div=0.1,
    lambda_sparsity=0.05,
    model_type='performer',
    save_path='trackformer.pth',
    device=None
):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset = MoMDataset(
        root_dir=root,
        sample_rate=sr,
        silence_thresh=1e-4,
        segment_sec=segment_sec,
        max_retry=10,
    )
  loader = DataLoader(dataset, batch_size =batch_size, shuffle=True, num_workers=2, drop_last=True)
  for i, (mom, x_pair) in enumerate(loader):
        print(f"Batch {i}: mom {mom.shape}, x_pair {x_pair.shape}")
        if i >= 1:
            break

  model = MixITModel(
    model_type=model_type,
    # <performer> 
    freq_bins=freq_bins, 
    n_masks=n_masks,
    performer_dim = performer_dim,
    performer_depth = performer_depth,
    performer_heads = performer_heads,
    performer_nb_features = performer_nb_features,
    performer_max_seq_len = performer_max_seq_len,
    # <convtasnet>
    num_sources = num_sources,
    enc_kernel_size = enc_kernel_size,
    enc_num_feats = enc_num_feats,
    msk_kernel_size = msk_kernel_size,
    msk_num_feats = msk_num_feats,
    msk_num_hidden_feats = msk_num_hidden_feats,
    msk_num_layers = msk_num_layers,
    msk_num_stacks = msk_num_stacks,
    msk_activate = msk_activate
  ).to(device)

  optimizer = optim.Adam(model.parameters(), lr = lr)

  for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    total_mixit = 0
    total_div = 0
    total_sparse = 0
    for mom_wav, pair_wav in loader:
      mom_wav = mom_wav.to(device)
      pair_wav = pair_wav.to(device)

      if model_type == 'performer':
        mom_mel = wav_to_mel(mom_wav, sr=sr)
        masks = model(mom_mel, mom_wav, device)

        est_sources = masks * mom_wav.unsqueeze(1)

        loss_mixit = mixit_loss(pair_wav, est_sources)
        loss_div = diversity_loss(masks)
        loss_sparsity = sparsity_loss(est_sources, mom_wav)

        loss = loss_mixit + lambda_div * loss_div + lambda_sparsity * loss_sparsity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mixit += loss_mixit.item()
        total_div += loss_div.item()
        total_sparse += loss_sparsity.item()

      elif model_type == 'convtasnet':
        est_sources = model(None, mom_wav, device)
        
        loss = mixit_loss(pair_wav, est_sources)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mixit += loss_mixit.item()
        total_div += loss_div.item()
        total_sparse += loss_sparsity.item()

      else:
        raise ValueError("Unknown model_type. Choose 'performer' or 'convtasnet'.")
    print(f'Epoch {epoch+1}/{epochs} Total loss = {total_loss/len(loader):.4f} = {(total_mixit/len(loader)):.4f}(mixit) + {(total_div/len(loader)):.4f}(div) + {(total_sparse/len(loader)):.4f}')

  torch.save(model.state_dict(), save_path)
  print(f'Model saved to {save_path}')