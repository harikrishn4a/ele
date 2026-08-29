# Robust AI-Generated Image Detection — literature-backed build plan

Scope: a 72-hour prototype that must hold accuracy under JPEG (q=90/70/50/30), Gaussian blur
(σ=0.5/1.0/2.0), 0.5×/0.25× resize-and-upscale, Gaussian noise (σ=0.02/0.05/0.10), ±20% colour
jitter, and 80% centre crop. Deliverables include a directory-in / JSON-out scoring script, a
clean-vs-transformed robustness table, and an error-analysis note.

Companion files: `robust_aigi_paper_shortlist.csv` (34 papers, tiered by how you should use them),
`ntire2026_clean_vs_robust.csv`, `ntire2026_robustness_gap.png`.

---

## 1. The single most relevant piece of evidence

The **NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild**
([arXiv:2604.11487](https://arxiv.org/abs/2604.11487), CVPRW 2026) is your task, run as a
competition four months ago. 108,750 real + 185,750 generated images from 42 generators, augmented
with 36 transformation types (Gaussian blur, white noise, JPEG, up to neural compression and
watermark-erasing attacks), 1–5 randomly composed distortions per image at random severities, both
classes degraded identically. Primary metric: ROC AUC over the distorted half of the test set.
511 registrants, 20 valid submissions.

Read the top-9 solution descriptions (Secs. 3 and 8 of that report) before writing any code. The
convergent finding across all of them:

> **A large pretrained vision foundation model with a light classification head, trained with the
> deployment-time degradation pipeline applied as augmentation.** Nobody won with a hand-crafted
> frequency or residual feature. Nobody won with a from-scratch CNN.

The differences between ranks are almost entirely (a) how aggressive and how *composed* the training
augmentation was, (b) how much extra generator diversity was added to the training data, and
(c) how many models were ensembled. (a) and (b) are free for you; (c) is what you drop.

The clean-versus-robust gap is the whole problem:

![NTIRE 2026 clean vs robust AUC]({{artifact:art_10c2f704-6dca-427f-b2f0-5f5ec3ef1989}})

Note **Shallow Real**: DINOv3-Large + LoRA + a six-token multi-aspect head + deep supervision +
supervised-contrastive loss reaches the third-best *clean* AUC (0.9953) and the *worst* robust AUC
(0.8336) — a 0.162 drop. Head sophistication bought clean accuracy and nothing else. The two
winners' gap is 0.025. **Robustness came from the data pipeline, not the architecture.** That single
comparison is the strongest argument in your pitch for why you built what you built.

---

## 2. Recommended build

### 2.1 Backbone — frozen, and as modern as you can afford

Ranked by evidence, pick the largest that fits your GPU:

| Choice | Evidence |
|---|---|
| **SigLIP2 (so400m or giant, patch16-384)** | NTIRE 5th place used SigLIP2-giant + mean-pool + one linear layer, nothing else, for 0.873 robust AUC. |
| **DINOv3-L / DINOv3-H** | Backbone of the 1st, 2nd and 4th place solutions. Frozen DINOv3 linear probes are strong even on badly-aligned training data ([arXiv:2608.15196](https://arxiv.org/abs/2608.15196)). |
| **CLIP ViT-L/14** | The reference point ([Ojha et al., CVPR 2023](https://arxiv.org/abs/2302.10174); [Cozzolino et al., arXiv:2312.00195](https://arxiv.org/abs/2312.00195)) — weaker than the above but has the most released code you can copy. |

Two supporting results: a plain linear probe on a current VFM beats bespoke detectors in the wild by
over 20% accuracy ([arXiv:2509.12995](https://arxiv.org/abs/2509.12995)), and a benchmark across VFM
families found the best modern backbone beats original CLIP by >12% accuracy *out of the box*
([TAP, arXiv:2604.26772](https://arxiv.org/abs/2604.26772)). Backbone choice is the highest-leverage
decision you make, and it costs you nothing but a download.

Caveat worth one sentence in your README: recent VLMs partially "know" the concept of AI-generated
imagery from their own pre-training data, and that advantage collapses on images scraped after the
pre-training cut-off ([arXiv:2509.12995](https://arxiv.org/abs/2509.12995)).

### 2.2 Head — start with the cheapest thing that works

1. **Global average pool over final-layer patch tokens → one linear layer.** The NTIRE 5th-place
   team evaluated CLS-token extraction, attention pooling and multi-layer concatenation, and found
   plain mean-pooling over patch tokens the most robust and stable.
2. If it plateaus, in order of cost: **tunable attention pooling** ([TAP](https://arxiv.org/abs/2604.26772)),
   then **multi-layer features with learned block weights** ([RINE](https://arxiv.org/abs/2402.19091),
   ECCV 2024 — +10.6% average over SOTA, and the best model trains in one epoch, roughly eight
   minutes, on released code).
3. Only if you still have time: **LoRA on the backbone with a frozen anchor classifier**
   ([ARA](https://arxiv.org/abs/2608.15196)) — the regularised way to fine-tune without destroying
   the pretrained representation.

### 2.3 Augmentation — this is where robustness comes from

Implement the seven evaluation transforms as a **composable, random-severity training augmentation**
and apply it with probability ~1.0, sampling 1–3 operations per image at severities that *extend
past* the evaluation grid (e.g. JPEG down to q=20, σ up to 2.5). The 5th-place report is explicit
that setting distortion probability to 1.0 with up to three composed operations at five severity
levels "is a key driver of our Robust ROC AUC improvements". The two winners went further with
tiered difficulty (clean / 1–3 mild / 3–6 moderate / 6 heavy) and sampled across tiers.

**Compose, don't apply singly.** Real laundering is JPEG-then-resize-then-JPEG. NTIRE degrades with
1–5 chained operations; a detector tuned on single transforms overstates its own robustness.

Three cheap additions with their own evidence:

- **Crop instead of down-sample in preprocessing**, plus ColorJitter and RandomRotation, plus
  patch-wise random masking — the three transforms of [SAFE](https://arxiv.org/abs/2408.06741)
  (KDD 2025). Down-sampling distorts the artifact itself; ColorJitter/rotation prevent overfitting
  to colour and semantic biases; masking forces local rather than global evidence.
- The counter-example: the 5th-place NTIRE team deliberately used a **"squish" resize to 384×384**
  rather than random-resized-crop, arguing crops can remove localised forensic cues. Both work;
  with a semantic VFM backbone, resize hurts less than it does for a high-frequency detector. Pick
  one, and note in your README that you tested it — it is a nice ablation row.
- **Horizontal flip** at train time, and flip-TTA at inference (used by half the NTIRE field).

### 2.4 The one non-obvious upgrade: pairwise clean/distorted training

[LoRA-based Pairwise Training](https://arxiv.org/abs/2604.12307) (TeleAI, NTIRE 3rd place, 0.9251
robust AUC) puts **each image and its distorted copy in the same batch** and trains:

```
L = CE(x, y) + α · KL(p(x) ‖ p(x̂)) + β · MSE(f_x, f'_x̂)      α = 0.5, β = 0.25
```

where `x̂` is the distorted version and `f'` is the distorted features passed through a small
correction FFN. It explicitly decouples "learn to detect" from "learn to be invariant", which is why
they lose only 0.053 AUC to degradation while teams above them in clean AUC lose 0.077–0.162. This is
maybe forty lines on top of a standard training loop and it is the single best robustness-per-effort
item on this list.

A distillation-shaped alternative if blur specifically is your weak axis: **DINO-Detect**
([arXiv:2511.12511](https://arxiv.org/abs/2511.12511)) freezes a teacher on sharp images and
distills feature + logit responses into a student trained on blurred copies.

### 2.5 Training data — diversity of *generators*, not volume of images

- Detection accuracy rises monotonically with the **number and diversity of generators** in the
  training set ([Community Forensics](https://arxiv.org/abs/2411.04125), CVPR 2025 — 2.7M images
  from 4,803 models; subsample it).
- You do **not** need a big set. [Cozzolino et al.](https://arxiv.org/abs/2312.00195) get
  state-of-the-art generalisation from "a handful of example images from a single generative model";
  [SSAFE](https://arxiv.org/abs/2606.08634) shows 10K curated images beating 288K and 4M with a
  frozen encoder. Spend your data budget on covering generator families (GAN / latent diffusion /
  DiT / autoregressive / commercial API), not on image count.
- Other ready sources: [OpenFake](https://arxiv.org/abs/2509.09495) (~4M, modern generators),
  GenImage, WildFake, AIGIBench, Chameleon — all used by NTIRE teams.

### 2.6 Calibration — do this, it is one scalar

Detectors systematically drift toward predicting "real" under distribution shift. Fit a **learnable
scalar logit correction on a small held-out validation set with the backbone frozen**
([arXiv:2602.01973](https://arxiv.org/abs/2602.01973), AAAI 2026, code released). It costs minutes,
needs no retraining, and it is what lets you make an honest claim about your false-positive rate —
which the error-analysis deliverable will ask you for anyway.

### 2.7 What to skip, and why (say this out loud in the pitch)

- **High-frequency / residual / patch-texture detectors** — NPR ([2312.10461](https://arxiv.org/abs/2312.10461)),
  PatchCraft ([2311.12397](https://arxiv.org/abs/2311.12397)), SSP ([2402.01123](https://arxiv.org/abs/2402.01123)).
  Excellent cross-generator numbers on clean images; the cue lives exactly in the frequencies that
  JPEG, blur and resize destroy. SSP's own paper reports the decline on low-quality images.
- **Reconstruction-error methods** — DIRE ([2303.09295](https://arxiv.org/abs/2303.09295)),
  AEROBLADE ([2401.17879](https://arxiv.org/abs/2401.17879)). Genuinely robust and elegant, but DIRE
  needs a diffusion inversion per image and AEROBLADE only sees latent-diffusion families. Both
  break your "directory in, JSON out, runs in the demo" requirement.
- **Ensembles.** The NTIRE winners ran 2×DINOv3-7B (14B parameters, 2.21 img/s and 78 GB VRAM on an
  A100) and 5–7 model committees. That is a leaderboard artifact, not a product. Your feasibility
  score (15% of judging) rewards the opposite. One model, one forward pass, flip-TTA at most.

---

## 3. Evaluation strategy

The hackathon asks for a clean-vs-transformed comparison; make it a grid and it becomes your
strongest slide. Model it on [AIGIBench](https://arxiv.org/abs/2505.12335) (NeurIPS 2025 D&B) and
[RRDataset](https://arxiv.org/abs/2509.09172) (ICCV 2025).

1. **Transform × severity grid**: 7 transforms × their severity levels, plus a clean row, plus at
   least one **composed chain** (e.g. resize 0.5× → JPEG 70 → blur 0.5) representing an actual repost.
   Report AUC per cell, and accuracy at a **fixed threshold chosen once on clean validation data** —
   re-tuning the threshold per transform is the most common way robustness gets overstated.
2. **Headline numbers**: clean AUC, mean robust AUC over all degraded cells, and the **gap** between
   them. NTIRE ranks on robust AUC alone; the gap is what your judges will remember.
3. **Held-out generators**: at least one generator family never seen in training, since real
   deployment is always against an unseen model.
4. **False positives on hard reals**: heavily-filtered phone photos, screenshots, low-light noisy
   captures, scans, memes. [CLIP-based detectors score "clean, polished, compositionally controlled"
   images as more synthetic](https://arxiv.org/abs/2602.12381) — so professional photography and
   stock imagery are your expected false positives, and messy amateur capture is your expected
   true-negative comfort zone. Say this before a judge finds it.
5. **FPR at a fixed TPR** (e.g. FPR@95%TPR) per transform. Accuracy alone hides the asymmetry that
   matters: falsely accusing a real photographer is the costly error.

### The trap that will silently inflate your numbers

[**Fake or JPEG?**](https://arxiv.org/abs/2403.17608) — most public AIGI datasets have real and fake
images stored at different JPEG qualities and different resolutions, so detectors learn the *codec*,
not the generator. Removing those biases moved cross-generator performance by more than 11 points.
Concretely: **re-encode every real and every fake identically** (same quality, same resize policy)
before training, and verify your fake/real class balance is not confounded with resolution. If you
do only one hygiene step from this document, do this one. [BIAS-ID](https://arxiv.org/abs/2605.31153)
gives you a protocol for reporting how bias-driven your detector actually is, which is a
ready-made structure for the error-analysis note.

Second trap: **semantic fallback** ([GSD, arXiv:2603.09242](https://arxiv.org/abs/2603.09242)) —
fine-tuned representations often stay organised by image *content* rather than forensic cues, so
your detector partly learns "cats are fake, weddings are real". Mitigate by keeping content matched
across classes where possible (caption-paired reals and fakes) and by testing on a content
distribution you did not train on.

---

## 4. Build order for 72 hours

Each step is independently demoable, so you always have something that runs.

1. **Hours 0–6.** Frozen backbone → mean-pooled patch tokens → cached features for a small balanced
   set (10–20K images, 4+ generator families, identically re-encoded). Fit logistic regression on
   the cached features. This is your baseline row and it takes minutes to train.
2. **Hours 6–14.** Build the degradation pipeline as a `torchvision` transform (the seven transforms,
   composable, random severity). Retrain the head on augmented features. Build the transform ×
   severity evaluation grid and generate the table. **You now have the required deliverable**;
   everything after this is upside.
3. **Hours 14–28.** Add pairwise clean/distorted training (§2.4), the SAFE preprocessing choices,
   flip-TTA, and the calibration scalar. Re-run the grid; keep every ablation row.
4. **Hours 28–48.** `predict.py --image-dir → JSON {image_path, pred}`, README, error analysis on
   hard reals, demo video. Do this *before* chasing more accuracy.
5. **Remaining time.** In order of expected value: more generator diversity in training data
   (§2.5) → attention-pooling or RINE head (§2.2) → LoRA with an anchor (§2.2).

**Ablation table to keep as you go** — this is the "deliberate decision-making" the 35%
technical-execution criterion is looking for:

| Variant | Clean AUC | Robust AUC | Gap |
|---|---|---|---|
| Frozen VFM + linear probe, no augmentation | | | |
| + degradation augmentation | | | |
| + pairwise clean/distorted loss | | | |
| + flip-TTA + calibration | | | |

If you have a spare thirty minutes, add a **training-free** row as an honesty check and a cold-start
story for unseen generators: [RIGID](https://arxiv.org/abs/2405.20112) (real images are more robust
to tiny noise perturbations than fakes, in VFM embedding space — no training at all) or
[WaRPAD](https://arxiv.org/abs/2511.14030) (wavelet sensitivity averaged over patches, crop- and
resolution-robust). Both are ~50 lines on top of the backbone you already loaded.

---

## 5. Trade-offs to discuss (20% of the score is problem insight)

- **Robustness vs clean accuracy.** Aggressive augmentation costs a little clean AUC and buys a lot
  of robust AUC. The Shallow Real vs winners comparison in §1 is the empirical anchor: 0.9953 clean
  / 0.8336 robust versus 0.9974 / 0.9723.
- **Generalisation vs specialisation.** Training on one generator family and testing on another is
  the realistic setting; generator diversity in training data is the only reliable fix found so far
  ([Community Forensics](https://arxiv.org/abs/2411.04125)), and even that decays as new generators
  ship. Frame your tool as needing continuous data refresh, not as a solved classifier.
- **False positives are the deployment-blocking error.** A detector that flags authentic
  photojournalism is worse than useless. Report FPR at fixed TPR, calibrate, and be explicit that
  the output is a confidence score for triage, not a verdict.
- **The moving target.** Detection is adversarial and non-stationary; the honest claim is a
  maintained pipeline (new generators sampled continuously, thresholds recalibrated), which is also
  the most credible answer to "how does this survive past the hackathon".
- **What robustness does *not* cover.** Adversarial perturbations, watermark removal attacks, and
  re-digitisation (screenshot / photo-of-screen) are separate axes; RRDataset
  ([2509.09172](https://arxiv.org/abs/2509.09172)) measures the last one, and NTIRE included
  watermark-erasing attacks. Naming the boundary of your claim reads as insight, not weakness.

---

## 6. Twelve papers, in reading order

1. [2604.11487](https://arxiv.org/abs/2604.11487) — NTIRE 2026 robust detection challenge. Your task, solved nine ways.
2. [2403.17608](https://arxiv.org/abs/2403.17608) — Fake or JPEG? Read before you build a dataset.
3. [2302.10174](https://arxiv.org/abs/2302.10174) — Ojha et al. Why frozen features beat trained detectors.
4. [2312.00195](https://arxiv.org/abs/2312.00195) — Cozzolino et al. Lightweight CLIP detector, +13% on laundered data.
5. [2604.12307](https://arxiv.org/abs/2604.12307) — LPT. The pairwise clean/distorted loss.
6. [2408.06741](https://arxiv.org/abs/2408.06741) — SAFE. Three preprocessing/augmentation choices.
7. [2402.19091](https://arxiv.org/abs/2402.19091) — RINE. Best accuracy-per-training-minute head.
8. [2509.12995](https://arxiv.org/abs/2509.12995) — Modern VFM baselines beat specialised detectors.
9. [2602.01973](https://arxiv.org/abs/2602.01973) — Post-hoc calibration in one scalar.
10. [2505.12335](https://arxiv.org/abs/2505.12335) — AIGIBench. Evaluation protocol to imitate.
11. [2507.10236](https://arxiv.org/abs/2507.10236) — What truly matters in the wild.
12. [2502.19716](https://arxiv.org/abs/2502.19716) — Review, for orientation and further reading.

Full annotated list with tiering and per-paper usage notes: `robust_aigi_paper_shortlist.csv`.
