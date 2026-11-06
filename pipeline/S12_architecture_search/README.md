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

---

## 4. Design Sanity Checks

| Check | Tool / Test | Result |
| --- | --- | --- |
| **Parameter Budget** | `param_counts.py` | All candidates within 4.1% of baseline (4.73M to 5.06M). |
| **Periodicity & Shift Equivariance** | `test_s12_nets.py` | Machine-precision shift equivariance ($< 10^{-10}$ in float64) under periodic roll for all candidates (including non-multiple shifts of 7 and 13 cells for FNO). |
| **Autocast & Precision** | `test_s12_nets.py` | FNO executes seamlessly under `torch.autocast` on CPU/bfloat16 and CUDA/fp16 with internal float32 spectral block. |
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
```

---

## 6. Experimental Results

*(To be filled after the Kaggle GPU runs — no numbers fabricated)*

| Model | Seed | Parameters | GPU Minutes | Pooled $R^2$ | Median Per-Case $R^2$ (All) | Median Per-Case $R^2$ (Idealized) | Median Per-Case $R^2$ (Realistic) | Mean $\Delta P_{95}$ | Inference Time / Case (s) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S11 Tuned U-Net (Ref)** | 0 | 4,860,313 | 45.2 | 0.7092 | 0.5662 | 0.6673 | 0.5234 | −0.4571 | *measured* |
| **S11 Tuned U-Net (Ref)** | 1 | 4,860,313 | 44.8 | 0.7122 | 0.5597 | 0.6210 | 0.5227 | −0.4569 | *measured* |
| **S11 Tuned U-Net (Ref)** | 2 | 4,860,313 | 45.0 | 0.7111 | 0.5953 | 0.6828 | 0.5437 | −0.4385 | *measured* |
| **Res-U-Net** | 0 | 5,057,377 | — | — | — | — | — | — | — |
| **Attention U-Net** | 0 | 5,009,717 | — | — | — | — | — | — | — |
| **FNO** | 0 | 4,728,337 | — | — | — | — | — | — | — |
| *Top Architecture 1* | 1 | — | — | — | — | — | — | — | — |
| *Top Architecture 1* | 2 | — | — | — | — | — | — | — | — |
| *Top Architecture 2* | 1 | — | — | — | — | — | — | — | — |
| *Top Architecture 2* | 2 | — | — | — | — | — | — | — | — |

---

## 7. Definition of Done

At least three architectures compared under equal budgets. Either the winner beats the S11 tuned U-Net on median per-case $R^2$ or mean $\Delta P_{95}$ by more than the seed spread (S11 seeds 0–2: median per-case $R^2 \in [0.560, 0.595]$, mean absolute error $\in [0.472, 0.486]\ u^*$), or the negative result is documented honestly.

---

## 8. Limitations

1. **Equal-budget constraint:** Hyperparameters (learning rate $3.6 \times 10^{-4}$, weight decay $6.5 \times 10^{-5}$) were tuned specifically for the baseline U-Net in S11. FNO or ResNet architectures may prefer different learning rates or schedules; holding them constant is a deliberate protocol choice to ensure equal training budgets, but represents a known limitation for architectures with different optimization dynamics.
2. **Wavelength tiling trade-off:** While 256×256 tiled inference preserves physical wavelength interpretation for FNO, it prevents the model from exploiting global cross-neighbourhood low-frequency modes across multi-kilometer domains in a single pass.

