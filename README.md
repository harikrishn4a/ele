# Ele: Robust AI-Generated Image Detection

Hackathon submission: detect AI-generated images even after JPEG compression, blur, resize, crop, noise, and color jitter.

**Status**: Building Phase 1 (baseline model + evaluation grid)

---

## Quick Start

### 1. Environment Setup
```bash
cd /Users/harikrishnannandakumar/Desktop/Ele
pip install -r requirements.txt
```

### 2. Download Training Data
```bash
python scripts/download_data.py
# Downloads: CIFAKE (10K), GenImage (10K), WildFake (2K)
# Total: ~22K real + synthetic images, multiple generator families
```

### 3. Train Baseline (Phase 1 target: hour 6)
```bash
python src/train.py \
  --backbone sigclip \
  --head linear \
  --epochs 3 \
  --batch_size 64 \
  --augmentation none \
  --output models/baseline_no_aug.pt
```

### 4. Evaluate on Transform Grid (Phase 1 target: hour 12)
```bash
python src/evaluate.py \
  --model models/baseline_no_aug.pt \
  --dataset data/test \
  --output results/baseline_no_aug.csv
```

### 5. Predict on New Images
```bash
python scripts/predict.py \
  --image_dir ~/my_images \
  --model models/baseline_no_aug.pt \
  --output predictions.json
```

---

## 72-Hour Roadmap

| Phase | Hours | What | Why | Deliverable |
|-------|-------|------|-----|-------------|
| **1. Baseline** | 0–12 | Load SigLIP → mean-pool patch tokens → linear probe. Evaluate on clean and each transform. | Establishes that a frozen VFM already separates real/fake. | `baseline_no_aug.csv` (clean AUC, robust AUC per transform) |
| **2. Robustness** | 12–24 | Apply degradation pipeline (JPEG/blur/resize/noise/crop/jitter) during training with probability 1.0, 1–3 composed ops. Re-evaluate. | Most robustness gains come from augmentation, not architecture. | `baseline_with_aug.csv` + ablation table |
| **3. Advanced** | 24–36 | Add pairwise clean/distorted training, SAFE transforms, flip-TTA, logit calibration. | Squeeze +3–5% robust AUC. | `advanced.csv` + ablation rows |
| **4. Production** | 36–48 | Write clean `predict.py`, README with commands, error analysis on hard reals (messy photos, screenshots, etc.). Demo video. | Judges score feasibility (15%) + technical execution (35%) + insight (20%). | Runnable submission + error analysis note |
| **5. Polish** | 48–72 | More generator diversity in training → attention pooling head → LoRA with anchor. | Incremental accuracy gains. | Final `sota.csv` |

---

## Architecture

### Backbone options (pick one, ranked by evidence)

| Backbone | Why | Training time | VRAM |
|----------|-----|---|---|
| **SigLIP-so400m** | 5th place NTIRE used SigLIP2-giant (0.873 robust AUC baseline) | 2–3h for 10K images | 8GB |
| **DINOv3-L** | 1st/2nd/4th place NTIRE. Strong on misaligned data. | 2–3h | 10GB |
| **CLIP ViT-L/14** | Reference. Most released code. | 1–2h | 6GB |

### Head options (ranked by robustness-per-code)

1. **Mean-pool over final patch tokens + logistic regression** (Phase 1)
   ```python
   # Pseudo-code
   patches = backbone.get_patches(image)  # (B, N, D)
   features = patches.mean(dim=1)          # (B, D)
   logits = linear_head(features)          # (B, 2)
   ```

2. **Tunable Attention Pooling (TAP)** (Phase 3, if time)
   ```python
   attention_weights = learn_attention(patches)  # (B, N, 1)
   features = (patches * attention_weights).sum(dim=1)
   ```

3. **LoRA on backbone + frozen anchor** (Phase 5, stretch goal)

---

## Key Findings from NTIRE 2026

- **Clean-vs-robust gap is the whole problem** (see `ntire2026_robustness_gap.png`): every top team loses 0.025–0.162 AUC when images are degraded.
- **Augmentation during training is the primary lever**, not architecture. Shallow Real (most sophisticated head) got 0.9953 clean / 0.8336 robust; winners got 0.9974 clean / 0.9723 robust (only 0.025 gap).
- **Generator diversity beats image count.** 10K curated images from 4+ families > 1M from 2 families.
- **High-frequency detectors (NPR, PatchCraft) fail on degraded images.** Their cue lives in frequencies that JPEG/blur destroy.

---

## Data Sources

### Public datasets used by NTIRE teams

| Dataset | Images | Generators | Note |
|---------|--------|-----------|------|
| **CIFAKE** | 10K | ProGAN, StyleGAN, StyleGAN2 | Curated, widely used baseline |
| **GenImage** | ~100K | 12+ (Stable Diffusion, Midjourney, DALL-E) | Largest modern dataset |
| **OpenFake** | ~4M | 40+ modern generators | Use a subsample (e.g., 50K) |
| **Community Forensics** | 2.7M | 4,803 different models | Highest diversity; subsample by generator |
| **WildFake** | ~10K | Real social media reosts | Realistic degradation |

For Phase 1–2, we recommend starting with **CIFAKE (10K) + GenImage subsample (10K)** = 20K balanced images across 4–6 generator families. This trains in ~3 hours and is what the 5th-place NTIRE team used.

### Setup

```bash
# (1) CIFAKE — direct download from Kaggle
# https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
# Extract to data/cifake/

# (2) GenImage — direct download
# https://github.com/donghao51/GenImage
# Extract to data/genimage/

# (3) Or use the provided download script
python scripts/download_data.py
```

---

## Evaluation Protocol

We evaluate on a **transform × severity grid**:

```
Transform         | q=90  | q=70  | q=50  | q=30  | Avg
JPEG              | 0.95  | 0.92  | 0.85  | 0.72  | 0.86
Gaussian Blur σ   | 0.97  | 0.94  | 0.88  | 0.78  | 0.89
Resize 0.5× then  | 0.96  | 0.93  | 0.87  | 0.75  | 0.88
  up-scale        |       |       |       |       |
...
Clean             | 0.99  | 0.99  | 0.99  | 0.99  | 0.99
Robust Avg        |       |       |       |       | 0.87
Gap                |      |       |       |       | 0.12
```

**Why this matters**: reports mean AUC per cell, so you see which transforms hurt most. Robust AUC is mean over degraded cells; gap is clean minus robust.

---

## Calibration & False Positives

Detectors systematically drift toward "real" under distribution shift (social media vs stock photos). We fit a **post-hoc logit scalar** on a small held-out validation set:

```python
# After training, on val set with frozen backbone:
original_preds = model.predict(val_images)  # (N,)
val_labels = val_labels  # (N,)

# Learn one scalar
scalar = learn_scalar(original_preds, val_labels)
calibrated_preds = original_preds * scalar

# Report: FPR @ 95% TPR, FNR @ 95% TPR
```

This controls false-positive rate without retraining.

---

## Pitfalls to Avoid

1. **Dataset bias** — public AIGI datasets often store real and fake at different JPEG qualities / resolutions. Re-encode everything identically before training. [Fake or JPEG?, arXiv:2403.17608]
2. **Semantic fallback** — fine-tuned models learn "polished = fake, messy = real". Keep content distribution similar across classes. [GSD, arXiv:2603.09242]
3. **High-frequency cues** — NPR, PatchCraft work on clean images but the signal lives in frequencies JPEG/blur destroy. Skip them. [arXiv:2312.10461, 2311.12397]
4. **Threshold tuning per transform** — if you tune threshold separately for each degradation, you overstate robustness. Tune once on clean validation, freeze for all degraded evaluations.

---

## File Structure Explained

```
src/
├── dataset.py         # DataLoader + augmentation pipeline (JPEG/blur/resize/noise/crop/jitter)
├── model.py           # Backbone factory (SigLIP/DINOv3/CLIP), head options
├── train.py           # Main training loop (standard CE + optional pairwise loss)
└── evaluate.py        # Transform grid evaluation, ablation tracking

scripts/
├── predict.py         # Inference: directory_path → JSON {image, prediction, confidence}
└── download_data.py   # Fetches CIFAKE, GenImage, WildFake to data/

models/
├── baseline_no_aug.pt           # Phase 1: frozen backbone + linear head, no augmentation
├── baseline_with_aug.pt         # Phase 2: + degradation augmentation
├── advanced_pairwise.pt         # Phase 3: + pairwise clean/distorted loss
└── sota_ensemble.pt             # Phase 5 (optional): multiple experts

results/
├── baseline_no_aug.csv          # Transform grid (clean AUC, robust AUC per cell)
├── baseline_with_aug.csv
├── advanced_pairwise.csv
└── ablation_summary.csv         # Side-by-side: no_aug vs with_aug vs pairwise vs TTA+calib
```

---

## Reproducibility

Every training run saves:
- `train_log.txt` — epoch, loss, val AUC
- `config.json` — exact hyperparameters
- `model.pt` — weights
- `results.csv` — transform grid

To reproduce Phase 2:
```bash
python src/train.py \
  --config saved_configs/phase2.json \
  --resume models/baseline_no_aug.pt \
  --epochs 5
```

---

## Next Steps

1. **Now**: Set up environment, download data (5 min)
2. **Hour 1–6**: Train Phase 1 baseline (no augmentation)
3. **Hour 6–12**: Add degradation augmentation, evaluate grid
4. **Hour 12+**: Pairwise training, SAFE, calibration, etc.

Start with:
```bash
cd /Users/harikrishnannandakumar/Desktop/Ele
pip install -r requirements.txt
python scripts/download_data.py
python src/train.py --backbone sigclip --epochs 3 --output models/baseline.pt
```

---

## References

- NTIRE 2026 Challenge: [arXiv:2604.11487](https://arxiv.org/abs/2604.11487)
- Cozzolino et al. (CLIP-based detection): [arXiv:2312.00195](https://arxiv.org/abs/2312.00195)
- LPT (pairwise training): [arXiv:2604.12307](https://arxiv.org/abs/2604.12307)
- SAFE (augmentation): [arXiv:2408.06741](https://arxiv.org/abs/2408.06741)
- Fake or JPEG (dataset bias): [arXiv:2403.17608](https://arxiv.org/abs/2403.17608)
- Full paper shortlist: `robust_aigi_paper_shortlist.csv`
