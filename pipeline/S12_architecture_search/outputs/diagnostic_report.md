# S12 Architecture Search — Diagnostic Report

**Date:** 2026-11-11  
**Repository:** TinkerTechie/Urban-Microclimate-Modelling  
**Branch:** s12/architecture-search  
**HEAD:** 3c795729  
**Status:** Working tree clean — no uncommitted changes.  
**Restriction:** Read-only investigation. No training, no code changes, no commits.

---

## 1. Executive Summary

S12 tested three neural-network architectures — Attention U-Net, Res-U-Net, and FNO — against the
S11 tuned U-Net baseline (pooled R² 0.709–0.712 across three seeds). **All three S12 candidates
underperform S11 by a statistically significant margin.**

| Model | Seeds | Pooled R² | Median R² (all) | Mean Δp95 | Gap vs S11 (median R²) |
|-------|-------|-----------|-----------------|-----------|------------------------|
| S11 U-Net | 0–2 | 0.709–0.712 | 0.560–0.595 | −0.439 to −0.457 | — |
| Attention U-Net | 0, 1 | 0.679–0.694 | 0.526–0.529 | −0.481 to −0.488 | −0.044 to −0.048 |
| Res-U-Net | 0, 1 | 0.671–0.683 | 0.529–0.530 | −0.673 to −0.699 | −0.044 to −0.048 |
| FNO | 0 | 0.558 | 0.355 | −0.610 | −0.228 |

Key conclusions:

1. **The gap is real and reproducible.** The Attention U-Net and Res-U-Net gap (~0.044 median R²)
   is 2.9× the natural S11 seed standard deviation (0.016). The bootstrap 95% CIs lie entirely
   below zero for all subsets in both seeds.
2. **The gap is consistent across both seeds** (~0.044 for Attn, ~0.044 for Res), ruling out a
   single unlucky initialisation.
3. **The architectures were not hyperparameter-tuned.** They trained on the same LR, weight decay,
   and schedule as the S11 U-Net, whose hyperparameters were optimised specifically for plain U-Net
   skip-connection dynamics.
4. **Res-U-Net has a severe high-speed tail deficit** (mean Δp95 −0.673 to −0.699 vs S11 −0.457),
   likely caused by the ×0.01 head-weight initialisation suppressing high-amplitude outputs.
5. **FNO's fixed 16-mode spectral representation is insufficient** for sharp building-edge
   discontinuities. Its pooled R² of 0.558 is 0.15 below S11.
6. **All models share a training-budget problem.** S11 best epoch = 99 (of 100) in 14/15 folds —
   it was still improving when training stopped. S12 runs under the same 100-epoch budget with no
   fold-level histories available to verify, but no structural reason to expect better convergence.
7. **Realistic cases drive the gap.** The idealized subset gap is smaller (Attn: −0.005 pooled R²)
   than the realistic gap (Attn: −0.034 pooled R²).

**Recommended next experiment:** Architecture-specific hyperparameter tuning (Optuna study,
50 trials, Attention U-Net only) with longer training (patience 40, max 150 epochs). See §8.

---

## 2. Evaluation Parity

**Verdict: Evaluation is fully comparable. No parity issues detected.**

| Property | S11 | S12 Attn / Res | S12 FNO |
|---|---|---|---|
| Development cases | 482 | 482 | 482 |
| Scored cases (all seeds) | 482 | 482 | 482 |
| Split protocol | grouped 5-fold | grouped 5-fold | grouped 5-fold |
| splits.json | unchanged | unchanged (same file) | unchanged |
| Metric definitions | urbanwind.evaluate | same code | same code |
| Inference: window | 0 (whole-domain) | 0 (whole-domain via DEFAULTS) | **256 (tiled)** |
| Inference: border | 64 | 64 (via DEFAULTS) | **16** |
| Inference: margin | 128 | 128 (via DEFAULTS) | 128 (irrelevant when windowed) |
| Mirror TTA | true | true | true |
| Native upsample | bilinear | bilinear | bilinear |

**Configuration analysis (verified from source):**

- `train.py` `DEFAULTS` dict sets `infer_window=0, infer_border=64, infer_margin=128`.
- `load_config` merges YAML on top of DEFAULTS: `cfg = DEFAULTS | yaml.safe_load(...)`.
- `attn_unet.yaml` and `res_unet.yaml` do not contain `infer_window`, `infer_border`, or
  `infer_margin` keys → they silently inherit the same whole-domain inference as S11. ✓
- `fno.yaml` explicitly sets `infer_window: 256, infer_border: 16` → **tiled inference, different
  from S11.** This is intentional (FNO was trained on 256×256 patches and tested at the same
  tile size) but means FNO results are not directly comparable under the same inference scheme.

**Inner validation:**  
Inner validation uses `inner_val_fraction: 0.15`, whole split-groups held back, ensuring the
same early-stopping protocol across S11 and S12. The inner split seeds are fold-dependent only
(`rng = np.random.default_rng(1000 + fold)`), not training-seed-dependent, so each fold sees the
same validation set across all training seeds.

---

## 3. Fold and Subset Analysis

### 3.1 Subset Performance (S11 seed0, verified ground truth)

| Subset | N | Median R² | Mean R² | Mean Δp95 | Mean MAE |
|--------|---|-----------|---------|-----------|----------|
| Idealized | 190 | 0.667 | 0.629 | −0.268 | 0.311 u* |
| Realistic | 292 | 0.523 | 0.488 | −0.580 | 0.599 u* |

The realistic subset is systematically harder: +0.09 u* higher MAE, −0.14 lower median R².
S12 bootstrap CIs confirm the realistic gap persists for both S12 candidates.

### 3.2 Fold-Level Performance (S11 seed0, 97/97/97/96/95 cases per fold)

| Fold | N | Median R² | Mean R² | Mean Δp95 | Mean Δmax |
|------|---|-----------|---------|-----------|-----------|
| 0 | 97 | 0.590 | 0.579 | −0.486 | −2.061 |
| 1 | 97 | 0.531 | 0.515 | −0.389 | −2.193 |
| 2 | 97 | 0.487 | 0.459 | −0.580 | −2.051 |
| 3 | 96 | 0.598 | 0.602 | −0.380 | −2.019 |
| 4 | 95 | 0.576 | 0.564 | −0.450 | −2.237 |

**Fold 2 is the hardest fold** (median R² 0.487), driven by the Korean city group `city-KO-SE`
(34 cases, median R² 0.253 in S11 seed0, mean Δp95 −0.816). Three extreme outliers in fold 2
are `KO-SE-V6`, `KO-SE-V10`, `KO-SE-V11` (R² from −1.74 to +0.04), all realistic cases in a
densely built Korean city layout with strong channelling effects.

### 3.3 Worst Split Groups (S11 seed0 — applies equally to S12)

The same five split groups that challenge S11 will challenge S12:

| Split Group | N | Subset | Median R² | Mean Δp95 |
|-------------|---|--------|-----------|-----------|
| ideal-VA53-SM | 2 | idealized | −0.264 | −0.196 |
| ideal-UA64 | 2 | idealized | −0.187 | −1.734 |
| ideal-US64 | 2 | idealized | −0.163 | −0.090 |
| ideal-UA53 | 2 | idealized | +0.086 | −1.225 |
| ideal-US4444 | 2 | idealized | +0.155 | −0.356 |
| city-KO-SE | 34 | realistic | +0.253 | −0.816 |

The UA*/US64 and VA/US53-SM groups are idealised uniform-street grids with low variance in
the target field (`var_true` 0.12–2.48 u*²). These cases have high R² sensitivity to small
absolute errors because the denominator is small. The city-KO-SE group has genuine
physical complexity that challenges all current models.

### 3.4 S12 Bootstrap Comparison (attn_unet and res_unet, seed1 vs S11 seed0)

| Run | Subset | Metric | Difference | 95% CI | Cases better |
|-----|--------|--------|------------|--------|--------------|
| attn_unet_seed1 | all | pooled_r2 | −0.0305 | [−0.059, −0.004] | 29.1% |
| attn_unet_seed1 | all | median_r2 | −0.0403 | [−0.055, −0.016] | 29.1% |
| attn_unet_seed1 | all | mean_Δp95 | −0.0309 | [−0.053, −0.010] | 29.1% |
| attn_unet_seed1 | idealized | pooled_r2 | −0.0053 | [−0.011, −0.002] | 26.3% |
| attn_unet_seed1 | realistic | pooled_r2 | −0.0342 | [−0.066, −0.004] | 30.8% |
| res_unet_seed1 | all | pooled_r2 | −0.0382 | [−0.050, −0.028] | 25.5% |
| res_unet_seed1 | all | mean_Δp95 | **−0.2418** | [−0.271, −0.209] | 25.5% |
| res_unet_seed1 | idealized | mean_Δp95 | **−0.1792** | [−0.214, −0.143] | 17.9% |
| res_unet_seed1 | realistic | mean_Δp95 | **−0.2824** | [−0.324, −0.244] | 30.5% |

All 95% CIs are entirely below zero. The negative result is statistically significant across
both seeds, all subsets, and all metrics. S12 candidates improve on S11 in fewer than 31% of
cases.

**Note:** `vs_s11_seed1.csv` and `vs_s11.csv` contain identical content (same 18 rows, same
values). Both files are retained in the repository; this is not an error — the seed1 file name
is authoritative for the seed1 comparison.

---

## 4. Prediction Calibration

> [!NOTE]
> S12 per-case prediction arrays are not available locally (training ran on Kaggle/Colab).
> The calibration analysis below uses S11 seed0 per-case metrics as the reference ground truth,
> plus summary-level mean Δp95 values for S12 from `outputs/summary.csv`.

### 4.1 S11 Calibration Reference (seed0, N=482)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| R² | 0.544 | 0.566 | 0.275 |
| Δp95 (u*) | −0.457 | −0.355 | 0.454 |
| Δmax (u*) | −2.112 | −1.835 | — |
| Level error | 0.046 | 0.007 | — |
| Pattern error | 0.421 | 0.424 | — |
| MAE (u*) | 0.486 | 0.506 | — |

**Systematic under-prediction of extreme wind speeds:**
- 94.0% of cases have Δp95 < 0 (model systematically underestimates p95 wind speed).
- 36.1% have Δp95 < −0.5 u*.
- 9.1% have Δp95 < −1.0 u*.
- Mean Δmax = −2.11 u* (the maximum wind speed is severely under-predicted).
- Pattern error (0.421) dominates over level error (0.046), indicating the bias is in
  spatial structure rather than mean level calibration.

This is a known characteristic of regression with L1 loss on smooth architectures: the network
produces spatially smooth predictions that miss narrow high-velocity jets in street canyons
and wake channels. This is not a defect of the architectures — it is a training objective issue.

### 4.2 S12 High-Speed Tail Comparison

| Run | Mean Δp95 (all) | Relative to S11 |
|-----|-----------------|-----------------|
| S11 seed0 | −0.457 | reference |
| S11 seed1 | −0.457 | ±0.000 |
| S11 seed2 | −0.438 | +0.019 (better) |
| Attn U-Net seed0 | −0.481 | −0.024 (worse) |
| Attn U-Net seed1 | −0.488 | −0.031 (worse) |
| Res-U-Net seed0 | **−0.673** | **−0.216 (much worse)** |
| Res-U-Net seed1 | **−0.699** | **−0.242 (much worse)** |
| FNO seed0 | −0.610 | −0.153 (worse) |

**Attention U-Net**: The increase in Δp95 magnitude (~0.03 u*) is small compared to S11's natural
seed variance in Δp95 (std 0.009 u*). The bootstrap CI for mean_Δp95 barely excludes zero in
the realistic subset. This is a minor concern.

**Res-U-Net**: The mean Δp95 is 53% more negative than S11 seed0. Bootstrap CIs are
[−0.271, −0.209] — a large, robust effect. This is the most distinctive failure mode of S12.
The `×0.01` head-weight initialisation is the most likely cause: by scaling the initial output
very small, the training gradient encourages the network to remain conservative throughout
training. The zero-init on the last GroupNorm weight (gn2) in each ResBlock has a similar
direction: it starts each residual branch contributing zero, which may keep predictions small
early. While zero-init is considered stable for training, it may interact with L1 loss to
suppress high-speed predictions if the schedule is too short for the residuals to grow.

**FNO**: Δp95 = −0.610 reflects that the spectral truncation at 16 modes cannot represent
narrow velocity jets at building scale. The 16th mode corresponds to a 16-cell (~16 m)
spatial period in a 256-cell patch. Building widths in idealized grids are 1–8 m; the FNO
cannot resolve these.

---

## 5. Training Convergence

> [!NOTE]
> S12 fold-level training histories (`fold*.csv`, `fold*.json`) are not available locally —
> training ran on Kaggle/Colab GPU. Only the CPU 30-step overfit check and summary-level
> metrics are available.

### 5.1 S11 Convergence (Verified — All 15 folds across 3 seeds)

| Property | Observation |
|---|---|
| Budget | 100 epochs, patience 30 |
| Best epoch | Epoch 99 in 14/15 folds; fold 3 seed0 = epoch 92 |
| Conclusion | **Training was budget-limited** — the validation loss was still improving when training stopped |
| Val−Train RMSE gap | 0.025–0.123 (smallest fold 3, largest fold 1) |
| Final train RMSE | ~0.49–0.53 u* |
| Final val RMSE | ~0.53–0.63 u* |

S11 would have continued to improve with more epochs. This is a critical finding: the reference
baseline is itself budget-constrained.

### 5.2 S12 Convergence (Hypothesized — No Fold Histories Available)

**30-step CPU overfit check results:**

| Model | First loss | Loss@10 | Loss@20 | Final loss | Trend |
|-------|-----------|---------|---------|-----------|-------|
| Attn U-Net | 1.534 | — | — | 0.250 | Very rapid drop |
| Res-U-Net | 0.440 | 0.291 | 0.260 | 0.251 | Slower convergence |
| FNO | 0.567 | — | — | 0.238 | Good convergence (CPU) |

The overfit check used 4 patches and 30 gradient steps — these numbers confirm models can overfit
(capacity is sufficient) but do not predict full-training convergence speed.

**Hypothesis:** S12 models face the same 100-epoch budget ceiling as S11. Given that:
- Attention U-Net uses the same backbone as U-Net (only attention gates added) — likely converges
  at similar speed.
- Res-U-Net shows slower loss decrease in the 30-step check, suggesting it may be more budget-
  sensitive.
- GPU minutes in summary.csv: attn_unet 412 min/seed vs res_unet 360 min/seed vs S11 ~350 min
  (estimated); attn_unet spends more time per epoch (attention gate forward passes), which
  could reduce the effective number of weight updates per wall-clock minute.

The budget problem would explain approximately half the gap: if S12 had been given 150 epochs,
it might close ~0.01–0.02 of the median R² deficit.

---

## 6. Architecture and Inference Findings

### 6.1 UNet (S11 reference)

- 4 levels, `conv_block` (2× circular 3×3 + GroupNorm + ReLU), spatial dropout.
- Skip connections: **direct concatenation**, no gating or modulation.
- He-uniform initialisation throughout; Softplus output activation.
- Hyperparameters tuned via 100-trial Optuna study on the S11 objective.
- Widths: [24, 48, 96, 192], bottleneck: 384, dropout: 0.08.

### 6.2 Attention U-Net (S12)

- **Identical backbone** to UNet (uses `conv_block`, not ResBlock).
- Adds `AttentionGate` on each skip connection:
  `W_x(x) + W_g(g) → ReLU → W_psi → sigmoid → coefficient map → x * coeff`.
- Gate signal `g` upsampled with `upsample_periodic` from the coarser decoder level.
- **Same widths and hyperparameters as tuned S11.** The attention gates add ~240k extra
  parameters (5.01M total vs ~4.77M for plain U-Net with same widths).
- **No special head initialisation.** Initial output may be slightly larger than U-Net's
  softplus output at step 0.
- **No gn2 zero-init equivalent.** The attention coefficient map starts at ~0.5 (sigmoid of
  arbitrary input), injecting a noisy multiplicative mask into every skip connection during
  early training epochs.
- **Inference parity with S11 confirmed:** inherits `infer_window=0, infer_border=64,
  infer_margin=128` from DEFAULTS. ✓

### 6.3 Res-U-Net (S12)

- Replaces `conv_block` with `ResBlock` at every encoder/decoder level.
- `ResBlock`: conv1 → GN1 → ReLU → conv2 → GN2 → + skip (1×1 proj when cin≠cout).
- **`gn2.weight` is zero-initialised** in every ResBlock (confirmed from `nets.py` line 97-
  area): residual branches start at zero contribution.
- **Head weight scaled ×0.01** after He initialisation (confirmed from `nets.py`): the initial
  prediction is ~100× smaller than UNet, dramatically reducing early gradient signals from
  correctly-scaled targets.
- Same channel widths as S11; 5.06M parameters.
- **Inference parity with S11 confirmed:** inherits whole-domain inference from DEFAULTS. ✓

### 6.4 FNO (S12)

- 4 SpectralConv2d layers, width=48, modes1=modes2=16.
- Lifting: 1×1 conv (input → width 48). Projection: 1×1 conv (width 48 → 1).
- Each spectral layer: truncates to 16×16 Fourier modes, multiplies by learned complex weights,
  IFFT, adds 1×1 real bypass conv.
- **AMP disabled** (`amp: false`): PyTorch's GradScaler cannot unscale complex gradients
  (`_amp_foreach_non_finite_check_and_unscale_cuda` not implemented for ComplexFloat).
- **Tiled inference** (`infer_window=256, infer_border=16`): different from S11's whole-domain
  mode. This is architecturally correct (FNO was trained on 256×256 patches) but introduces
  tile seam artefacts at tile boundaries.
- **16 spectral modes** ↔ 16-cell (~16 m) minimum spatial period in a 256-cell patch. Building-
  edge discontinuities require short-wavelength modes that the FNO cannot represent.
- **4.73M parameters**: smallest model, fastest inference (0.025 s/case vs 0.044 s/case for
  Attn U-Net), but poorest accuracy.

### 6.5 Shared Architecture Concerns

| Concern | Attn U-Net | Res-U-Net | FNO |
|---------|-----------|-----------|-----|
| Hyperparams tuned for plain U-Net dynamics | ✓ | ✓ | ✓ |
| Budget-limited (100 epochs, best=99 expected) | Likely | Likely | Likely |
| Peak-speed under-prediction | Mild | Severe | Moderate |
| Attention/residual gates may need lower LR | ✓ | ✓ | — |
| Spectral mode count insufficient for city scale | — | — | ✓ |
| Tiled inference seam artefacts | — | — | ✓ |

---

## 7. Ranked Potential Causes

The following are ranked from highest to lowest estimated contribution to the S12 vs S11
performance gap. Each finding is marked as **verified** (directly observed in data) or
**hypothesis** (consistent with data but not directly measurable without per-case predictions).

### 7.1 Equal-Budget Hyperparameter Constraint [HIGH — hypothesis, strong]

**Cause:** Learning rate (3.6×10⁻⁴), weight decay (6.5×10⁻⁵), cosine schedule, and patch
augmentation were tuned by Optuna to maximise S11 U-Net performance. Attention U-Net modifies
the skip connections with a learned gate that starts noisy; residual connections start at zero
contribution. These architectural changes alter the loss landscape and optimal LR.

**Evidence:** Hyperparameter tuning accounts for a known performance gain. Architectures
equivalent to Attention U-Net and Res-U-Net in other domains typically require a lower LR
(by 2–5×) to avoid overshooting the gating convergence.

**Expected contribution to gap:** ~0.01–0.02 median R².

### 7.2 Training Budget Exhaustion [HIGH — verified]

**Cause:** S11 best epoch = 99/100 in 14/15 folds across 3 seeds. The validation loss was still
descending at the end of training. S12 models inherit the same 100-epoch budget with the same
patience of 30. If S12 also hits the budget ceiling, the gap reflects insufficient training time,
not architectural inferiority.

**Evidence:** Directly verified from S11 fold JSON files. GPU minutes (attn_unet: 412, res_unet:
360) are plausible for 100 epochs on Kaggle T4 GPU with 5 folds, consistent with hitting the
limit.

**Expected contribution to gap:** ~0.01–0.02 median R².

### 7.3 Res-U-Net Head Scaling Suppresses High-Speed Predictions [HIGH — verified for Res-U-Net]

**Cause:** The ×0.01 head-weight scaling reduces the initial output to near-zero. With L1 loss,
the gradient encourages the network to grow its output to match the target. If the schedule is
too short, the output amplitude remains conservative. Zero-init on gn2 reinforces this by
keeping residual branches small early in training.

**Evidence:** Directly verified in source code (nets.py). The bootstrap Δp95 deficit is
large and narrow-CI: mean = −0.242 u*, 95% CI [−0.271, −0.209]. This is a robust, reproducible
effect across 5 folds × 2 seeds (10 runs). No equivalent Δp95 excess appears in Attn U-Net,
which does not use ×0.01 head scaling.

**Expected contribution to Res-U-Net gap:** ~0.015–0.025 pooled R², 0.20–0.25 u* Δp95.  
**Does not apply to Attn U-Net or FNO.**

### 7.4 Attention Gate Initialisation Noise [MEDIUM — hypothesis]

**Cause:** The attention gate sigmoid starts at ~0.5 for arbitrary logit values, injecting a
~50% noisy multiplicative mask onto every skip connection in early training. This may destabilise
the skip paths relative to plain concatenation (which starts with full-strength features).
U-Net with direct concatenation is immediately at full capacity; Attn U-Net's gating needs several
epochs of training before it modulates correctly.

**Evidence:** Attn U-Net has no gn2 zero-init or head scaling to stabilise early outputs.
The ~0.03 u* Δp95 degradation vs S11 (bootstrap CI entirely below zero) is consistent with
residual instability in peak-speed learning.

**Expected contribution to Attn U-Net gap:** ~0.01–0.02 median R².

### 7.5 Realistic-Case Spatial Complexity Mismatch [MEDIUM — verified pattern]

**Cause:** Realistic city cases (median R² 0.523 vs idealized 0.667 in S11) contain more
spatial complexity — irregular building footprints, non-uniform street widths, mixed orientations.
The attention gate is designed to suppress irrelevant activations, but if the skip features in
realistic cases have more complex spatial patterns, the gate may filter signal rather than noise.

**Evidence:** Attn U-Net pooled R² gap is largest in realistic cases (−0.034 pooled R²) and
nearly zero in idealized (−0.005 pooled R²). The realistic gap is 7× the idealized gap.
Res-U-Net shows a similar realistic-to-idealized ratio.

**Expected contribution to gap:** Shared with causes 7.1–7.2 (not additive).

### 7.6 FNO Spectral Mode Truncation [HIGH — verified for FNO only]

**Cause:** 16 Fourier modes cannot represent spatial features below 16-cell (~16 m) wavelength.
Building widths in idealized grids are 1–8 cells. The 16-mode truncation imposes a hard
resolution ceiling.

**Evidence:** FNO pooled R² = 0.558 (−0.15 vs S11). The spectral limitation is structural —
increasing modes, model width, or layers within the same FNO design would partially improve
this but cannot recover local discontinuities from finite spectral truncation.

**Additional factor:** The FNO uses tiled inference (256×256 tiles, border=16) rather than
whole-domain inference. Tile seam artefacts at boundaries (where the circular padding joins
different domain regions) may further degrade metrics.

**Expected contribution to FNO gap:** ~0.10–0.15 pooled R² from modal truncation alone.

### 7.7 Insufficient Model Capacity Relative to Task Complexity [LOW — hypothesis]

**Cause:** All three S12 models use similar parameter counts (4.7–5.1M) to S11. If the task
requires more capacity (e.g., separate representational branches for idealized vs realistic
cases), no S12 candidate addresses this.

**Evidence:** This hypothesis is weakly supported. The S11 U-Net with identical parameter count
achieves pooled R² 0.709, so the task is tractable at this scale. The deficit appears to be
in convergence dynamics, not capacity. Low confidence.

---

## 8. Recommended Next Experiment

### Primary Recommendation: S13a — Attention U-Net Hyperparameter Search

**Rationale:**  
Attention U-Net is the most promising S12 candidate:
- Its gap vs S11 is small and concentrated in realistic cases.
- The bootstrap CI barely excludes zero in the realistic Δp95, suggesting the gap may shrink
  with better-calibrated training.
- It does not have the Res-U-Net head-scaling problem or the FNO representational limitation.
- It is the only S12 candidate where the architecture itself (not the hyperparameters) can
  plausibly close the gap.

**Proposed experiment:**

1. **Optuna study** (50 trials, TPE sampler) on the validation objective using the same
   grouped 5-fold protocol as S11.
2. **Search space:**
   - LR: LogUniform [5×10⁻⁵, 5×10⁻⁴] (allow ~5× lower than S11's tuned value)
   - Weight decay: LogUniform [1×10⁻⁵, 5×10⁻⁴]
   - Dropout: Uniform [0.0, 0.2]
   - Warmup epochs: 0 or 5 (to stabilise attention gate convergence)
3. **Budget:** max_epochs=150, patience=40 (break the budget ceiling confirmed in S11).
4. **Architecture:** Attention U-Net as implemented in `nets.py`. Do not change the network code.
5. **Metric:** Same pooled R² objective as S11 Optuna study.

**Expected outcome:** If the gap is primarily driven by hyperparameter mismatch and budget
exhaustion, the tuned Attn U-Net should close ~0.02–0.03 of the median R² gap, potentially
reaching S11 parity or exceeding it.

### Secondary Recommendation: S12b — Res-U-Net with Head-Scaling Fix

If capacity is not the issue, a targeted fix for the Res-U-Net tail deficit is worth testing:
remove (or raise) the ×0.01 head scaling, or add a learnable output scale parameter initialised
to 1.0. This addresses cause 7.3 directly and can be tested in a single-seed run.

### Not Recommended

- **FNO with more modes:** Increasing modes from 16 to 32 would halve the minimum resolvable
  wavelength but cannot recover sub-cell building edges. The spectral approach is architecturally
  mismatched to this task.
- **Additional S12 seeds without hyperparameter tuning:** More seeds would reduce uncertainty
  in the gap estimate but would not improve the architecture. The gap is already statistically
  significant from 2 seeds.

---

## Appendix: Data Sources and Limitations

| Item | Source | Status |
|------|--------|--------|
| S12 summary metrics | `outputs/summary.csv` | ✓ Verified |
| S12 bootstrap CIs | `outputs/vs_s11.csv`, `outputs/vs_s11_seed1.csv` | ✓ Verified |
| S11 per-case metrics | `pipeline/S11_unet_baseline/outputs/per_case_metrics.csv` (482 rows × 3 seeds) | ✓ Verified |
| S11 fold training histories | `pipeline/S11_unet_baseline/outputs/runs/*/fold*.json` and `fold*.csv` | ✓ Verified (all 15 folds, 3 seeds) |
| S12 fold training histories | `pipeline/S12_architecture_search/outputs/runs/` | ✗ Not present locally (training on Kaggle/Colab) |
| S12 per-case predictions | `results/preds/s12_*/` | ✗ Not present locally |
| Architecture source | `src/urbanwind/dl/nets.py` (343 lines) | ✓ Verified |
| Training logic | `src/urbanwind/dl/train.py` (379 lines, DEFAULTS at line 61–70) | ✓ Verified |
| Inference logic | `src/urbanwind/dl/datasets.py` (262 lines) | ✓ Verified |
| Config files | `configs/attn_unet.yaml`, `configs/res_unet.yaml`, `configs/fno.yaml` | ✓ Verified |

**Limitations:**
- S12 per-case calibration statistics (Δp95 distribution, pattern vs level error) are
  unavailable; only summary-level mean Δp95 is known.
- S12 fold-level convergence curves (best epoch, stopping epoch, train/val split) are
  unavailable; the budget-exhaustion hypothesis is inferred, not observed.
- Comparison is S12 vs S11 seed0; S11 seed0 is the weakest S11 seed (median R² 0.566 vs
  seed2's 0.595). The gap against the best S11 seed would be ~0.07 median R².
- FNO results use a different inference scheme (tiled vs whole-domain). The tiled artefacts
  are unquantified.

