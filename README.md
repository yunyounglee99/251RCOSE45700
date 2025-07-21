Trackformer: Self-Supervised Music Source Separation using performer

Performer + MixIT + Audio MAE = Compact, context-aware music de-mixer

⸻

🔍 Overview

MixPerformer is a self-supervised framework that swaps the CNN-based ConvTasNet separator in MixIT for a linear-attention Performer Transformer and jump-starts learning with Audio Masked Autoencoder (Audio MAE) pre-training. By combining global‐context attention with label-free MixIT training, MixPerformer separates up to four musical stems while cutting training time and GPU memory. ￼

⸻

🚨 Motivation
	•	Local-only CNN limits – ConvTasNet excels at short-range features but struggles with long rhythmic or harmonic structures typical of music. ￼
	•	Memory & scalability – ConvTasNet breaks on long (30 s) segments with CUDA OOM, blocking high-fidelity separation. ￼

⸻

🧠 Key Ideas

Component	Role
Performer encoder (12 layers, FAVOR+)	Linear O(L · r) attention captures long-range music context. ￼
Audio MAE pre-training	Learns robust global audio tokens before MixIT fine-tuning. ￼
MixIT loss	Matches any permutation of 8 soft-mask outputs to two mixtures—no labels required. ￼
Auxiliary Diversity & Sparsity losses	Reduce mask overlap and encourage energy concentration without ground truth. ￼


⸻

🧪 Datasets & Protocols
	•	MOISESDB (≈ 500 h) – four-stem songs only. ￼
	•	Segment lengths : 3 s, 5 s, 10 s, 30 s. ￼
	•	Train / Val split : 80 % / 20 %. ￼

⸻

⚙️ Implementation Details
	•	Masks (N) : 8 • Batch : 4 • LR : 1 e-4 → 1 e-5 cosine • Opt : AdamW (wd = 0.01) ￼
	•	Audio MAE : 30 % masking, 256 random features. ￼
	•	Hardware : single Tesla V100 32 GB. ￼

⸻

🏆 Results

Segment	ConvTasNet SI-SNR (dB)	Performer SI-SNR (dB)	Speed-up
3 s	– 9.11	+1.66	1.36 × faster (17.1 → 12.6 min/epoch) ￼
5 s	– 8.52	+2.00	1.27 × faster ￼
10 s	– 9.73	+2.33	1.22 × faster
30 s	OOM	+3.52	✅ trains (46 min/epoch)

	•	+3 – 13 dB absolute gain over ConvTasNet across lengths. ￼
	•	No OOM on 30 s thanks to linear attention. ￼

Ablation – Removing Audio MAE boosts short-segment SI-SNR but loses memory savings (OOM at 30 s). ￼

⸻

📌 Why MixPerformer?

Feature	MixPerformer	ConvTasNet
Global context	✅ Performer attention	❌ Local conv
Self-supervised	✅ MixIT	✅
Handles 30 s audio	✅ Fits GPU	❌ OOM
Training time	↓ 20–30 %	–
Mask interpretability	✅ Diversity + Sparsity losses	⚠️ Overlapping masks

Key findings summarised in the paper confirm performance, efficiency and global-context advantages. ￼

⸻

📉 Current Limitations
	•	Small corpus (≈ 10 h) ⇒ over-fitting; val/test SI-SNR can drop below 0 dB. ￼
	•	Linear attention compression may lose fine details; up-sampling refinements planned. ￼
	•	Fixed output count (N=8); dynamic source estimation is future work. ￼

⸻

🔮 Roadmap
	1.	Scale pre-training to 100 – 1000 h diverse music. ￼
	2.	Multi-metric eval (SDR, PESQ, SI-SDR). ￼
	3.	Stereo / 5.1 & real-time extensions for live performance. ￼

⸻

Built with ❤️ by Team 4 (Yunyoung Lee et al.).
