# Trackformer - Music Separation Using Self-Supervised Learning


A novel self-supervised learning framework for music source separation leveraging Transformer architectures, specifically Performer, to address limitations of local contextual processing inherent in existing CNN-based methods.

## 🔍 Overview

**Music Separation Using SSL** is a Transformer-based self-supervised learning approach that replaces ConvTasNet with Performer enhanced by FAVOR+ for efficient global context learning from long audio sequences. The method eliminates reliance on labeled data by generating effective training signals using self-supervised techniques within the MixIT paradigm, while Audio Masked Autoencoder (Audio MAE) pre-training facilitates extraction of robust global audio features.

## 🚨 Motivation

Traditional CNN-based audio separation approaches like ConvTasNet suffer from:

- ⚠️ **Limited global context**: CNNs excel at local features but struggle with long-range dependencies essential for music (rhythm, melody, harmony)
- ⚠️ **Memory constraints**: Cannot process very long audio sequences due to CUDA out-of-memory issues
- ⚠️ **Dependency on labeled data**: Supervised methods require extensive labeled datasets, resulting in high costs and limited scalability

To overcome these limitations, this work introduces **Performer with FAVOR+ attention** and **Audio MAE pre-training** for efficient global context modeling in music separation.

## 🧠 Key Ideas

- **MixIT Framework**: Self-supervised training by matching combinations of model outputs to original mixtures using MixIT loss
- **Performer Architecture**: Replaces standard self-attention with FAVOR+ (Fast Attention Via Positive Orthogonal Random Features), reducing complexity from O(L²) to O(L·r)
- **Audio MAE Pre-training**: Masks portions of input audio and reconstructs them to learn robust global audio representations
- **Segment-wise Processing**: Processes fixed-length chunks (3s, 5s, 10s, 30s) to learn both local and global patterns
- **Scale-Invariant SNR**: Evaluation metric for separation quality

**FAVOR+ Attention Formula**:
```
Attention(Q, K, V) ≈ φ(Q) [φ(K)ᵀV]
```
where φ(·) is an orthogonal random feature mapping.

**SI-SNR Calculation**:
```
SI-SNR(ŝ, s) = 10 log₁₀(||s_target||² / ||e_noise||²)
```

## 🧪 Datasets & Settings

The model is evaluated on:

- **MOISESDB Dataset**
  - ~500 hours of music with multi-instrument splits
  - Restricted to songs with ≤4 stems to match MixIT conditions
  - Segment lengths: 3s, 5s, 10s, 30s
  - 80/20 train/validation split

## ⚙️ Implementation Details

- **MixIT Configuration**: M = 8 masks, Batch size = 4
- **Optimizer**: AdamW (LR = 10⁻⁴, weight decay = 0.01)
- **Audio MAE**: 30% masking ratio, 12 Performer layers, 256 random features
- **Hardware**: NVIDIA Tesla V100 (32 GB)
- **STFT Parameters**: FFT size = 1024, Hop size = 256, Hann window

## 🏆 Results

### Performance Comparison (SI-SNR in dB)

| Segment Length | ConvTasNet | Performer | Improvement |
|---------------|------------|-----------|-------------|
| 3s            | -9.11      | +1.66     | +10.77 dB   |
| 5s            | -8.52      | +2.00     | +10.52 dB   |
| 10s           | -9.73      | +2.33     | +12.06 dB   |
| 30s           | OOM Error  | +3.52     | N/A         |

### Training Efficiency (Minutes per Epoch)

| Segment Length | ConvTasNet | Performer | Speed-up |
|---------------|------------|-----------|----------|
| 3s            | 17.05      | 12.57     | 26% faster |
| 5s            | 20.33      | 16.04     | 21% faster |
| 10s           | 43.25      | 35.42     | 18% faster |
| 30s           | OOM Error  | 46.33     | N/A        |

## 📌 Why This Approach?

| Feature | Performer + Audio MAE | ConvTasNet |
|---------|----------------------|------------|
| Global context | ✅ FAVOR+ attention | ❌ Local convolutions |
| Memory efficiency | ✅ Linear complexity | ❌ Quadratic/OOM |
| Long sequences | ✅ Up to 30s | ❌ Fails at 30s |
| Training speed | ✅ 18-26% faster | ❌ Slower |
| Self-supervised | ✅ No labels needed | ⚠️ Requires labels |

## 🔬 Additional Contributions

### Novel Evaluation Metrics (No Ground Truth)
- **Mask Overlapping Measure**: Assesses redundancy between estimated source masks
- **Effective Mask Count via Entropy**: Quantifies how many masks are meaningfully used

### Auxiliary Losses
- **Diversity Loss**: Encourages mutually dissimilar masks for distinct sources
- **Sparsity Loss**: Promotes energy concentration in fewer masks

## 📉 Limitations

• **Dataset Size**: Only ~10 hours of 4-stem music led to overfitting (SI-SNR_train ≈ +3-4 dB vs SI-SNR_val << 0 dB)

• **Information Loss**: FAVOR+ dimensionality reduction may lose fine-grained details during masking/reconstruction

• **MixIT Constraints**: Fixed N=8 output streams may not capture all sources in complex musical pieces

• **Audio MAE Impact**: Ablation studies showed that removing Audio MAE sometimes yielded higher SI-SNR, suggesting potential redundancy in feature transformations

## 🚀 Future Work

- **Large-Scale Pretraining**: Scale to 100-1000 hours of diverse music data
- **Multi-Scale Feature Fusion**: Combine low-level and high-level representations during decoding
- **Dynamic Source Estimation**: Adaptive mechanism for estimating number of active sources
- **Multi-Channel Extension**: Extend to stereo and surround audio formats
- **Real-Time Applications**: Evaluate feasibility for live performance and broadcasting
