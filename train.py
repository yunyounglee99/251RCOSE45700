# ─────────────────────────────────────────────────────────────────────────────
# train_ddp.py
#  - DistributedDataParallel(DDP)로 MixIT + Performer/ConvTasNet 학습
#  - torch.multiprocessing.spawn()을 사용하여 프로세스별 GPU 1:1 매핑
#  - UpsampleBlock, mixit_loss, diversity_loss, sparsity_loss 모두 사용
# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ""
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

import argparse
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.performer import PerformerSeperator
from models.convtasnet import ConvtasnetSeperator
from MixIT.model import MixITModel
from MixIT.Loss.mixit_loss import mixit_loss
from MixIT.Loss.diversity_loss import diversity_loss
from MixIT.Loss.sparsity_loss import sparsity_loss
from dataloader import MoMDataset
from utils import wav_to_mel, UpsampleBlock, si_snr


def parse_args():
    """
    학습 시 필요한 인자들을 argparse로 선언
    """
    parser = argparse.ArgumentParser(description="MixIT + DDP 학습 스크립트")

    # ─── Data & I/O ──────────────────────────────────────────────────────────────
    parser.add_argument("--root", type=str, required=True,
                        help="혼합 오디오(.wav)들이 있는 최상위 폴더 경로")
    parser.add_argument("--save_path", type=str, default="mixit_trackformer.pth",
                        help="학습된 모델 파라미터를 저장할 경로")
    parser.add_argument("--model_type", type=str, default="performer",
                        choices=["performer", "convtasnet"],
                        help="사용할 분리기 모델: 'performer' 또는 'convtasnet'")

    # ─── 학습 하이퍼파라미터 ────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=2,
                        help="GPU 1장당 배치 사이즈. 총 배치 = batch_size × world_size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="전체 학습 epoch 수") 
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="초기 학습률")
    parser.add_argument("--mixit_threshold", type=float, default=10.0,
                        help="mixit_loss 내부에서 사용하는 threshold")
    parser.add_argument("--lambda_div", type=float, default=0.1,
                        help="diversity loss 가중치")
    parser.add_argument("--lambda_sparse", type=float, default=0.1,
                        help="sparsity loss 가중치")

    # ─── Performer 설정 ───────────────────────────────────────────────────────
    parser.add_argument("--freq_bins", type=int, default=80,
                        help="Mel-spectrogram freq bin 개수 (Performer input dim)")
    parser.add_argument("--n_masks", type=int, default=8,
                        help="Performer가 출력할 mask 개수")
    parser.add_argument("--performer_dim", type=int, default=768,
                        help="Performer 모델 차원 (AudioMAE 사용 시 768 고정)")
    parser.add_argument("--performer_depth", type=int, default=6,
                        help="Performer 레이어 수")
    parser.add_argument("--performer_heads", type=int, default=8,
                        help="Performer 어텐션 헤드 수")
    parser.add_argument("--performer_nb_features", type=int, default=128,
                        help="Performer random feature 수")
    parser.add_argument("--performer_max_seq_len", type=int, default=1024,
                        help="Performer 최대 시퀀스 길이 (patch 개수)")

    # ─── ConvTasNet 설정 ───────────────────────────────────────────────────────
    parser.add_argument("--num_sources", type=int, default=8,
                        help="ConvTasNet 분리할 source 개수 (MixIT M)")
    parser.add_argument("--enc_kernel_size", type=int, default=16,
                        help="ConvTasNet 인코더 커널 크기")
    parser.add_argument("--enc_num_feats", type=int, default=512,
                        help="ConvTasNet 인코더 feature 개수")
    parser.add_argument("--msk_kernel_size", type=int, default=3,
                        help="ConvTasNet 마스크 TCN 커널 크기")
    parser.add_argument("--msk_num_feats", type=int, default=128,
                        help="ConvTasNet 마스크 TCN channel 수")
    parser.add_argument("--msk_num_hidden_feats", type=int, default=512,
                        help="ConvTasNet TCN hidden channel 수")
    parser.add_argument("--msk_num_layers", type=int, default=8,
                        help="ConvTasNet TCN layer 수")
    parser.add_argument("--msk_num_stacks", type=int, default=3,
                        help="ConvTasNet TCN stack 수")
    parser.add_argument("--msk_activate", type=str, default="sigmoid",
                        help="ConvTasNet mask activation")

    # ─── 기타 설정 ─────────────────────────────────────────────────────────────
    parser.add_argument("--segment_sec", type=float, default=10.0,
                        help="랜덤 샘플링 시 segment 길이 (초)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="샘플링 레이트")
    parser.add_argument("--silence_thresh", type=float, default=1e-4,
                        help="무음 기준 임계값")
    parser.add_argument("--max_retry", type=int, default=10,
                        help="무음 파일 재시도 횟수")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="DataLoader num_workers")

    return parser.parse_args()


def ddp_worker(local_rank, world_size, args):
    """
    DDP용 워커 프로세스 함수
      - local_rank: 이 프로세스가 바인딩된 GPU index (0 ~ world_size-1)
      - world_size: 총 프로세스(GPU) 개수
      - args: parse_args()로 얻은 인자들
    """

    # ── 1. 프로세스 그룹 초기화 ─────────────────────────────────────────────────
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=local_rank)

    # ── 2. GPU 디바이스 세팅 및 시드 고정 ────────────────────────────────────────
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # (Optional) reproducibility를 위해 시드 고정
    seed = 42
    random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    torch.cuda.manual_seed(seed + local_rank)

    # ── 3. Dataset → DistributedSampler → DataLoader ──────────────────────────
    dataset = MoMDataset(
        root_dir=args.root,
        sample_rate=args.sr,
        silence_thresh=args.silence_thresh,
        segment_sec=args.segment_sec,
        max_retry=args.max_retry,
    )

    # DistributedSampler를 통해 각 프로세스(=GPU)가 다른 subset을 읽도록 설정
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

    for i, (mom, x_pair) in enumerate(dataloader):
            print(f"Batch {i}: mom {mom.shape}, x_pair {x_pair.shape}")
            if i >= 1:
                break

    # ── 4. 모델 생성 및 DDP 래핑 ──────────────────────────────────────────────
    model = MixITModel(
        model_type=args.model_type,
        # Performer 인자
        freq_bins=args.freq_bins,
        n_masks=args.n_masks,
        performer_dim=args.performer_dim,
        performer_depth=args.performer_depth,
        performer_heads=args.performer_heads,
        performer_nb_features=args.performer_nb_features,
        performer_max_seq_len=args.performer_max_seq_len,
        # ConvTasNet 인자
        num_sources=args.num_sources,
        enc_kernel_size=args.enc_kernel_size,
        enc_num_feats=args.enc_num_feats,
        msk_kernel_size=args.msk_kernel_size,
        msk_num_feats=args.msk_num_feats,
        msk_num_hidden_feats=args.msk_num_hidden_feats,
        msk_num_layers=args.msk_num_layers,
        msk_num_stacks=args.msk_num_stacks,
        msk_activate=args.msk_activate
    )

    # 모델을 현재 프로세스의 GPU에 올리기
    model.to(device)

    # DDP 래핑 (각 프로세스마다 local_rank GPU 할당)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],      # 이 프로세스가 사용할 GPU
        output_device=local_rank,     # 결과(gradient reduction 등)를 모을 GPU
        find_unused_parameters=True  # 사용되지 않는 파라미터 추적 끄기(성능 향상)
    )

    # ── 5. 옵티마이저, 스케줄러 설정 ────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50, verbose=True)

    # ── 6. UpsampleBlock 생성 (한 번만) ────────────────────────────────────────
    #    - Performer 분기마다 in_len, out_len, channels가 달라질 수 있으므로
    #      첫 배치에서 텐서 크기를 보고 생성하거나, 미리 args를 통해 계산해도 됩니다.
    #    - 여기서는 “첫 배치”를 만나서 바로 생성하는 형태로 구현
    upsampler = None  # 모델.module.upsampler로 attachment할 예정

    # ── 7. 메인 학습 루프 ─────────────────────────────────────────────────────
    for epoch in tqdm(range(args.epochs), desc="Epoch", leave=True):
        model.train()
        epoch_loss = 0.0
        epoch_mixit = 0.0
        epoch_div = 0.0
        epoch_sparse = 0.0
        ep_sisnr = 0.0            # ← 추가 / 위치 이동
        eval_batches = 0  

        # DistributedSampler 사용 시 매 epoch마다 set_epoch() 호출
        sampler.set_epoch(epoch)

        """
        # tqdm progress bar (프로세스마다 출력하지 않고, rank 0에서만 출력하도록)
        if local_rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch[{epoch+1}/{args.epochs}]")
        else:
            pbar = dataloader
        """

        for mom_wav, pair_wav in dataloader:
            # ── 7.1. 데이터 GPU에 올리기 ───────────────────────────────────────
            mom_wav = mom_wav.to(device)      # (B,1,T_wav)
            pair_wav = pair_wav.to(device)    # (B,2,T_wav)

            if args.model_type == "performer":
                # 1) mel 변환
                mom_mel = wav_to_mel(mom_wav, sr=args.sr, n_mels=args.freq_bins, hop_length=int(0.016*args.sr))
                # 모양: (B, freq_bins, T_mel) → PerformerSeperator 인풋 형태 (B, 1, T_mel, freq_bins)
                mom_mel = mom_mel.permute(0, 1, 3, 2)  # (B, 1, T_mel, freq_bins)

                # 2) DDP 래핑된 모델 forward → mel-domain mask (B, M, T_mel)
                masks = model(mom_mel, mom_wav, device)

                # 3) UpsampleBlock이 생성되지 않았다면, 첫 배치 크기로 생성
                if upsampler is None:
                    B, M, T_mel = masks.shape
                    T_wav = mom_wav.shape[-1]
                    upsampler = UpsampleBlock(in_len=T_mel, out_len=T_wav, channels=M).to(device)
                    # DDP 모델 내부에 붙이기
                    model.module.upsampler = upsampler

                # 4) mel-domain → wav-domain 마스크 (B, M, T_wav)
                masks_wav = model.module.upsampler(masks)

                # 5) 마스크 × wav → 분리된 source 파형 (B, M, T_wav)
                est_sources = masks_wav * mom_wav

                # 6) Loss 계산
                loss_mix = mixit_loss(pair_wav, est_sources, threshold=args.mixit_threshold)
                loss_div = diversity_loss(masks)            # (B, M, T_mel)
                loss_sp = sparsity_loss(est_sources, mom_wav)  # (B, M, T_wav)

                loss = loss_mix + args.lambda_div * loss_div + args.lambda_sparse * loss_sp

            else:  # args.model_type == "convtasnet"
                # ConvTasNet은 mel 변환 없이 wav 입력 → 직접 (B, M, T_wav) 반환
                est_sources = model(None, mom_wav, device)
                loss = mixit_loss(pair_wav, est_sources)

            # 7.2. backward & step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 7.3. 누적 손실 산정 (CPU로 옮김)
            epoch_loss += loss.item()
            if args.model_type == "performer":
                epoch_mixit += loss_mix.item()
                epoch_div += loss_div.item()
                epoch_sparse += loss_sp.item()

            with torch.no_grad():
                B, M, T = est_sources.shape
                bits = torch.arange(1, 2**M - 1, device=device)
                mask_mat = ((bits[:, None] >> torch.arange(M, device=device)) & 1).float()  # (C,M)
                C = mask_mat.size(0)

                est = est_sources.unsqueeze(1)                       # (B,1,M,T)
                g1 = (est * mask_mat.view(1, C, M, 1)).sum(2)        # (B,C,T)
                g2 = (est * (1-mask_mat).view(1, C, M, 1)).sum(2)    # (B,C,T)

                mix1, mix2 = pair_wav[:, 0], pair_wav[:, 1]          # (B,T)

                sisnr1 = si_snr(mix1.unsqueeze(1).expand(-1, C, -1), g1)  # (B,C)
                sisnr2 = si_snr(mix2.unsqueeze(1).expand(-1, C, -1), g2)
                sisnr_mean = (sisnr1 + sisnr2) / 2                   # (B,C)

                best_idx = torch.argmax(sisnr_mean, dim=1)           # (B,)
                best1 = g1[torch.arange(B, device=device), best_idx] # (B,T)
                best2 = g2[torch.arange(B, device=device), best_idx]

                batch_sisnr = (si_snr(mix1, best1) + si_snr(mix2, best2)) / 2
                ep_sisnr += batch_sisnr.mean().item()
                eval_batches += 1

        # ── 8. 에포크 종료 후 로그 및 스케줄러 업데이트 ───────────────────────────
        avg_loss = epoch_loss / len(dataloader)
        if local_rank == 0:
            if args.model_type == "performer":
                print(f"[Rank {local_rank}] Epoch {epoch+1}/{args.epochs} "
                      f"TotalLoss={avg_loss:.4f}  MixIT={epoch_mixit/len(dataloader):.4f}  "
                      f"Div={epoch_div/len(dataloader):.4f}  Sparse={epoch_sparse/len(dataloader):.4f}"
                      f"SI-SNR={ep_sisnr/eval_batches:.2f} dB")
            else:
                print(f"[Rank {local_rank}] Epoch {epoch+1}/{args.epochs} TotalLoss={avg_loss:.4f}"
                      f"SI-SNR={ep_sisnr/eval_batches:.2f} dB")

        # Scheduler step (DDP에서는 rank 0만 하면 충분)
        if local_rank == 0:
            scheduler.step(avg_loss)

    # ── 9. 학습 완료 후 모델 저장 ───────────────────────────────────────────
    if local_rank == 0:
        # DDP wrapping된 모델은 .module.state_dict() 사용
        torch.save(model.module.state_dict(), args.save_path)
        print(f"Model saved at {args.save_path}")

    # ── 10. 프로세스 그룹 종료 ──────────────────────────────────────────────
    dist.destroy_process_group()

# -------- train.py --------
def main():
    args = parse_args()

    # torchrun이 넣어준 env에서 rank 정보 가져오기
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 그냥 한 번만 호출
    ddp_worker(local_rank, world_size, args)

if __name__ == "__main__":
    main()