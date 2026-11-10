# S12 Architecture Search

Tests whether alternative deep-learning architectures predict the pedestrian-level wind pattern $U/u^*$ better than the S11 periodic U-Net under strictly matched parameter and training budgets.

| | |
| --- | --- |
| **Inputs** | S10 cache (`data/processed/dl/`), `data/splits.json`, S11 baseline weights / predictions |
| **Outputs** | `configs/` (`res_unet.yaml`, `attn_unet.yaml`, `fno.yaml`); `overfit_check_<model>.json`; `outputs/summary.csv`, `outputs/vs_s11.csv`, figures; `kaggle_jobs.json` |
| **Code** | `src/urbanwind/dl/nets.py` (added `ResBlock`, `ResUNet`, `AttentionGate`, `AttnUNet`, `SpectralBlock`, `FNO`); tests in `src/urbanwind/dl/tests/test_s12_nets.py`; stage scripts `param_counts.py`, `overfit_check.py`, `s12_results.py` |

---

## 1. Candidate Architectures & Parameter Budget

All candidate models are implemented in `src/urbanwind/dl/nets.py` and registered via `@register("<name>")`. Convolutions of kernel size $\ge 3$ use circular padding and upsampling uses `upsample_periodic` to ensure exact periodic shift-equivariance across neighbourhood boundaries. Outputs pass through `softplus` to enforce positive wind speed predictions.

Baseline: **Tuned S11 periodic U-Net** (`widths: [24, 48, 96, 192], bottleneck: 384, dropout: 0.08`, **4,860,313 parameters**).
Every architecture is constrained to be within $\pm 25\%$ of the baseline budget (3,645,235 to 6,075,391 parameters).

| Architecture | Registry Name | Key Components | Parameters | Ratio vs Baseline | Multiple |
| --- | --- | --- | --- | --- | --- |
| **Tuned U-Net (Baseline)** | `unet` | 4-level U-Net, double circular convs, GroupNorm, spatial dropout | 4,860,313 | 100.0% | 16 |
| **Periodic Res-U-Net** | `res_unet` | Residual blocks (two circular 3x3 convs + GroupNorm + ReLU with 1x1 projection shortcuts) | 5,057,377 | 104.0% (+4.0%) | 16 |
| **Periodic Attention U-Net** | `attn_unet` | Attention gates (Oktay et al. 2018) on skip connections using 1x1 convs and periodic upsampling | 5,009,717 | 103.1% (+3.1%) | 16 |
| **Fourier Neural Operator (FNO)** | `fno` | 1x1 lifting, 4 spectral conv layers (float32 2D rfft2/irfft2 + complex weights + 1x1 bypass + GELU), 1x1 projection | 4,728,337 | 97.3% (−2.7%) | 1 |

All three models fall within $4.1\%$ of the target budget.

---

## 2. Equal Budget Rules

To isolate the impact of architecture design from hyperparameter advantages:
1. **Identical Optimisation & Schedule:**
   - AdamW with base learning rate $3.6 \times 10^{-4}$, weight decay $6.5 \times 10^{-5}$, min lr $1.0 \times 10^{-5}$ (cosine annealing over 100 epochs).
   - Batch size 16, 70 gradient steps per epoch, gradient clipping at 1.0.
   - Masked L1 loss on open ground cells.
2. **Identical Data & Folds:**
   - 256×256 wrap-around training patches with border margin 16.
   - Same grouped 5-fold cross-validation (`splits.json`). The frozen test set is never read.
   - Same 15% inner-validation set held back per fold with early stopping patience 30 on balanced validation RMSE.
   - Exponential Moving Average (EMA, decay 0.999 with warm-up) evaluated after every epoch.
3. **No Per-Architecture Hyperparameter Tuning:**
   - No architecture-specific learning rate or decay tuning was conducted.

---

## 3. FNO Mode Representation & Tiling

- **Grid-cell mode counting:**
  Modes in FNO are indexed in grid cells. In a domain of $N$ cells with grid spacing $\Delta x = 1\text{ m}$, Fourier mode index $k$ corresponds to physical wavelength $\lambda = \frac{N \Delta x}{k}$.
  For example, mode 16 represents a 16 m wave on a 256-cell patch, but a 64 m wave on a 1,024-cell full domain.
- **Wrap-around tiling inference:**
  To guarantee that frequency modes correspond to the exact same physical spatial wavelengths during inference as during training, the FNO config enforces `infer_window: 256` and `infer_border: 16`. Full domains are inferred via overlapping 256×256 wrap-around tiles matching the training distribution.
- **Float32 Spectral Execution under Mixed Precision:**
  PyTorch AMP autocast is explicitly disabled inside `SpectralBlock`. FFT and complex multiplication run in `float32` because half-precision (fp16) cuFFT restricts sizes to powers of two and risks precision loss, whereas float32 FFT supports arbitrary non-power-of-two domain sizes.
- **AMP Disabled for FNO (`amp: false`):**
  FNO uses AMP disabled because its complex-valued spectral gradients are incompatible with the current PyTorch GradScaler unscale operation. Res-U-Net and Attention U-Net retain AMP enabled. This is an implementation compatibility adjustment, not an architectural change.

---

## 4. Design Sanity Checks

| Check | Tool / Test | Result |
| --- | --- | --- |
| **Parameter Budget** | `param_counts.py` | All candidates within 4.1% of baseline (4.73M to 5.06M). |
| **Periodicity & Shift Equivariance** | `test_s12_nets.py` | Machine-precision shift equivariance ($< 10^{-10}$ in float64) under periodic roll for all candidates (including non-multiple shifts of 7 and 13 cells for FNO). |
| **Autocast & Precision** | `test_s12_nets.py` | FNO executes under `torch.autocast` on CPU/bfloat16 with an internal float32 spectral block. Under CUDA fp16 the GradScaler cannot unscale complex gradients, so the FNO is trained with `amp: false` (section 3). |
| **Overfit 4 patches (sanity check)** | `overfit_check.py` | 30-step CPU smoke check confirms rapid loss decrease without collapse (`res_unet` $0.44 \to 0.25$, `attn_unet` $1.53 \to 0.25$, `fno` $0.57 \to 0.24$). Full 2,000-step overfit validation runs on Kaggle GPU. |

---

## 5. Running the Pipeline

### Local Sanity & Verification
```bash
# Verify model parameters and budget compliance
uv run python pipeline/S12_architecture_search/param_counts.py

# Run unit and integration tests
URBANWIND_DEVICE=cpu uv run pytest src/urbanwind/dl/tests/test_s12_nets.py -v

# Run fast CPU overfit sanity check (30 steps)
URBANWIND_DEVICE=cpu uv run python pipeline/S12_architecture_search/overfit_check.py --model all --steps 30
```

### Full Training on Kaggle GPU
The job queue in `kaggle_jobs.json` defines the execution order for `pipeline/S11_unet_baseline/kaggle_runner.py`:
1. **Screening Phase:** Fold 0, Seed 0 for all 3 models (`res_unet`, `attn_unet`, `fno`).
2. **Remaining Folds:** Folds 1–4, Seed 0 for all models.
3. **Multi-Seed Confirmation:** Seeds 1 and 2 for the top 2 performing architectures across all 5 folds.

To run a single fold locally or on GPU:
```bash
uv run python -m urbanwind.dl.train --config pipeline/S12_architecture_search/configs/res_unet.yaml --seed 0 --folds 0
```

### Scoring and Comparison Tooling
Once training completes:
```bash
# Score finished runs and generate outputs/summary.csv and outputs/vs_s11.csv
uv run python pipeline/S12_architecture_search/s12_results.py

# Plot visual comparison between S12 candidates and S11 baseline range
uv run python pipeline/S12_architecture_search/plot_comparison.py
```

---

## 6. Experimental Results

All three architectures were trained under the equal-budget protocol (section 2): 5 outer folds, seed 0, then scored on all 482 development cases through `urbanwind.evaluate`. Raw numbers: `outputs/summary.csv`.
The S11 rows are copied from `pipeline/S11_unet_baseline/outputs/summary.csv` (grouped split) and the GPU minutes are the sums of the `minutes` field in its fold records.

| Model | Seed | Parameters | GPU minutes | Pooled $R^2$ | Median per-case $R^2$ (all) | Median per-case $R^2$ (idealized) | Median per-case $R^2$ (realistic) | Mean $\Delta P_{95}$ | Inference s / case |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| S11 tuned U-Net (ref) | 0 | 4,860,313 | 327.8 | 0.7092 | 0.5662 | 0.6673 | 0.5234 | −0.4571 | n/a |
| S11 tuned U-Net (ref) | 1 | 4,860,313 | 329.9 | 0.7122 | 0.5597 | 0.6210 | 0.5227 | −0.4569 | n/a |
| S11 tuned U-Net (ref) | 2 | 4,860,313 | 337.2 | 0.7111 | 0.5953 | 0.6828 | 0.5437 | −0.4385 | n/a |
| Attention U-Net | 0 | 5,009,717 | 412.2 | 0.6937 | 0.5291 | 0.6287 | 0.4988 | −0.4813 | 0.0442 |
| Res-U-Net | 0 | 5,057,377 | 359.5 | 0.6830 | 0.5288 | 0.6274 | 0.4998 | −0.6733 | 0.0382 |
| FNO | 0 | 4,728,337 | 251.1 | 0.5578 | 0.3552 | 0.3808 | 0.3434 | −0.6099 | 0.0245 |

Seeds 1 and 2 for the two best architectures were **not run** (see limitations). The S11 inference time was not measured with this harness, so no S11 value is given.

### Reading the table

- **No new architecture beats the S11 tuned U-Net.** The best candidate, the Attention U-Net, has pooled $R^2$ 0.694 against S11's 0.709–0.712 (S11 seed spread 0.003), median per-case $R^2$ 0.529 against 0.560–0.595, and mean $\Delta P_{95}$ −0.481 against −0.439 to −0.457. It is below S11 on all three headline metrics.
- **Where the gap is.** On idealized cases the Attention U-Net's median per-case $R^2$ (0.629) lies inside the S11 seed range (0.621–0.683), so one seed cannot separate them. On realistic cases (0.499 against 0.523–0.544) and in $\Delta P_{95}$ it is below the whole S11 range.
- **Attention U-Net and Res-U-Net are tied on per-case $R^2$** (0.5291 against 0.5288 overall; within 0.001 in each subset). Pooled $R^2$ differs by 0.011 and $\Delta P_{95}$ by 0.19, from one seed each, so no ranking between them is claimed.
- **The FNO is clearly the weakest** (pooled $R^2$ 0.558, median per-case 0.355). It is the cheapest to train (251 GPU minutes) and run (0.0245 s/case).
- **$\Delta P_{95}$ is negative for every model**, so all of them (including S11) under-predict the high-speed tail on average. The S12 models do so by more than S11.
- **Hypothesis, not tested:** residual blocks and attention gates add capacity but no new inductive bias for the building-wake structure; the FNO's fixed low-frequency modes may be a poor match for sharp building edges.

---

## 7. Definition of Done

At least three architectures compared under equal budgets. Either the winner beats the S11 tuned U-Net on median per-case $R^2$ or mean $\Delta P_{95}$ by more than the seed spread (S11 seeds 0–2: median per-case $R^2 \in [0.560, 0.595]$, mean absolute error $\in [0.472, 0.486]\ u^*$), or the negative result is documented honestly.

**Outcome:** Negative result documented; S11 remains best.

---

## 8. Limitations

1. **Equal-budget constraint:** Hyperparameters (learning rate $3.6 \times 10^{-4}$, weight decay $6.5 \times 10^{-5}$) were tuned specifically for the baseline U-Net in S11. FNO or ResNet architectures may prefer different learning rates or schedules; holding them constant is a deliberate protocol choice to ensure equal training budgets, but represents a known limitation for architectures with different optimization dynamics.
2. **Wavelength tiling trade-off:** While 256×256 tiled inference preserves physical wavelength interpretation for FNO, it prevents the model from exploiting global cross-neighbourhood low-frequency modes across multi-kilometer domains in a single pass.
3. **Single seed per architecture:** FNO, Res-U-Net, and Attention U-Net were evaluated on seed 0 only. S12 has no multi-seed spread estimate; S11's spread is taken from its three seeds.
4. **FNO numeric precision (`amp: false`):** The FNO was trained with `amp: false`, because fp16 `GradScaler` cannot unscale complex gradients. The other two used AMP, so numeric precision is not identical.
5. **Approximate GPU minutes and resumed fold 4:** GPU minutes are approximate: folds ran on Kaggle and Colab T4 GPUs, two jobs shared a node on Kaggle, and resumed folds count only the minutes of the final segment. Fold 4 was resumed from checkpoints created on a different machine (same code and configuration).
6. **Inference time benchmarking:** Inference time was measured on a single GPU model in one session and is only comparable among the S12 rows.
7. **Paired bootstrap comparison:** The group-bootstrap comparison against S11 (`outputs/vs_s11.csv`) was not produced.
