"""
from google.colab import drive

drive.mount('/content/drive')
%cd '/content/drive/MyDrive/딥러닝 팀프로젝트/mixtures'
%ls
%cd '2_stems'
%ls
"""
import os
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
torchaudio.set_audio_backend('soundfile')


class MoMDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        silence_thresh: float = 1e-4,
        segment_sec: int = 1.0,
        max_retry: int = 10,
    ):
        """
        Args:
            root_dir: mixtures/ 또는 mixtures/2_stems 등
            sample_rate: 모델 입력 SR
            silence_thresh: 무음 임계값
            segment_length: 출력 WAV 길이(샘플 수)
            max_retry: 무음 파일 재시도 횟수
        """
        # stem 하위 폴더와 직속 폴더 양쪽 모두 검색
        pattern_all    = glob.glob(os.path.join(root_dir, "*_stems", "*.wav"))
        pattern_single = glob.glob(os.path.join(root_dir,          "*.wav"))
        self.file_list = pattern_all if len(pattern_all) >= len(pattern_single) else pattern_single

        assert len(self.file_list) >= 2, f"wav 파일이 충분히(≥2) 있어야 합니다. 현재: {len(self.file_list)}개"

        self.sr             = sample_rate
        self.silence_thresh = silence_thresh
        self.segment_length = int(segment_sec * sample_rate)
        self.max_retry      = max_retry

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)     # wav: [C, T]
        # → Mono 처리: 여러 채널 평균
        if wav.dim() == 2:
            wav = wav.mean(dim=0)           # [T]
        # Resample
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # Silence 필터링
        if wav.abs().max() < self.silence_thresh:
            raise RuntimeError("silence")
        # 정규화
        wav = wav / torch.clamp(wav.abs().max(), min=1e-8)
        return wav                         # [T]

    def _fix_length(self, wav: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        wav: [T]
        → pad/trim 후 [1, target_len] 반환
        """
        L = wav.size(0)
        if L < target_len:
            pad = wav.new_zeros(target_len - L)  # [pad_len]
            wav = torch.cat([wav, pad], dim=0)   # [target_len]
        else:
            wav = wav[:target_len]
        return wav.unsqueeze(0)  # [1, target_len]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 랜덤 두 파일 선택 & 무음 재시도
        for _ in range(self.max_retry):
            try:
                p1, p2 = random.sample(self.file_list, 2)
                x1 = self._load_and_preprocess(p1)  # [T1]
                x2 = self._load_and_preprocess(p2)  # [T2]
                break
            except RuntimeError:
                continue
        else:
            p1, p2 = random.sample(self.file_list, 2)
            x1 = self._load_and_preprocess(p1)
            x2 = self._load_and_preprocess(p2)

        # 원하는 길이로 pad/trim
        x1 = self._fix_length(x1, self.segment_length)  # [1, L]
        x2 = self._fix_length(x2, self.segment_length)  # [1, L]

        # mixture of mixtures 생성 & 정규화
        mom = x1 + x2                                   # [1, L]
        mom = mom / torch.clamp(mom.abs().max(), min=1e-8)

        # 둘을 stack
        x_pair = torch.cat([x1, x2], dim=0)             # [2, L]

        return mom, x_pair

'''
if __name__ == "__main__":
    # ─── 설정 ────────────────────────────────────
    ROOT_DIR      = "/Users/nyoung/Library/CloudStorage/GoogleDrive-kembel0116@gmail.com/내 드라이브/딥러닝 팀프로젝트/mixtures"
    SAMPLE_RATE   = 16000
    SEGMENT_LEN   = 32000    # 예: 2초 분량
    BATCH_SIZE    = 4
    NUM_WORKERS   = 2
    # ────────────────────────────────────────────

    ds = MoMDataset(
        root_dir=ROOT_DIR,
        sample_rate=SAMPLE_RATE,
        silence_thresh=1e-4,
        segment_length=SEGMENT_LEN,
        max_retry=10,
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 테스트 출력
    for i, (mom, x_pair) in enumerate(loader):
        print(f"Batch {i}: mom {mom.shape}, x_pair {x_pair.shape}")
        if i >= 1:
            break

'''  
