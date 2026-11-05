# S11 Periodic U-Net and the shared trainer

Trains a convolutional network that predicts the pedestrian-level wind speed U/u* over a whole neighbourhood from the S10 inputs, under the same grouped five-fold protocol as Phase 1, and provides the training code every later deep-learning stage reuses.

| | |
| --- | --- |
| Inputs | the S10 cache (`data/processed/dl/`), `data/splits.json` |
| Outputs | `results/preds/<exp_id>/<case>.npz` and `results/checkpoints/<exp_id>/` (not tracked); `outputs/runs/<exp_id>/fold<k>.json` and `.csv`; `outputs/summary.csv`, `outputs/vs_s07.csv`, figures |
| Code | `src/urbanwind/dl/nets.py`, `losses.py`, `train.py`, `datasets.py`; `configs/`, `overfit_check.py`, `tune_optuna.py`, `s11_results.py`, `kaggle_runner.py`; tests in `src/urbanwind/dl/tests/` |

Status: the code is complete and tested. The final results (tuned settings, three seeds, five folds, and the random-split comparison) follow in a second pull request that closes #35. The fold-0 result below is preliminary.

## Network

`UNet` in `nets.py`, 8,636,961 parameters with the default widths:
- four levels of 32, 64, 128 and 256 channels and a 512-channel bottleneck; two 3×3 convolutions with GroupNorm and ReLU per block;
- **periodic everywhere**: circular padding in every convolution, and bilinear upsampling that wraps across the edges. A test checks in float64 that shifting the input by a multiple of 16 cells shifts the output by exactly the same amount;
- spatial dropout after the bottleneck and the decoder blocks, but not before the output layer; He initialisation; a softplus output, so speeds stay positive.

New architectures register themselves with `@register("name")` and are built with `build_model(name, in_ch, **kwargs)`.

## Training protocol (`train.py`)

1. **Folds.** Each outer fold trains on the other four grouped folds of `splits.json` and is scored on its own held-out cities and layout families. The frozen test set is never read.
2. **Inner validation.** Whole groups from the training folds are held back for model selection: 15% of the idealized and 15% of the realistic training cases, with no single city or layout family above a quarter of the set (61 to 67 cases per fold, at most 14 from one group). The choice depends on the fold only, so seeds differ in training alone.
3. **Optimisation.**
   - AdamW with cosine decay from the learning rate to 1e-5, weight decay, batch 16;
   - mixed precision with the loss in float32, and gradient clipping at 1.0;
   - masked L1 or MSE loss (`losses.py`), computed on valid open-ground cells only.
4. **Equal budgets.** Every epoch is 70 steps of 256×256 wrap-around patches, so all folds get the same training. Patches are mirrored at random, and patches of 45° cases are also transposed, which keeps the wind at 45°.
5. **Weights and selection.** A moving average of the weights (EMA, decay 0.999 with a warm-up) is validated after every epoch: each held-out case is predicted whole, its RMSE is averaged within each subset, then over the two subsets. The best EMA weights are kept, and training stops after `patience` epochs without improvement. With `inner_val_fraction: 0` the network instead trains on every training case for the full schedule and keeps its final weights.
6. **Records.** Every fold writes the code version (commit and whether the code differed from it), a hash of its settings, its inner-validation cases and its per-epoch history. A stopped run resumes from its last finished epoch and refuses to resume with changed settings.

## Prediction (`datasets.predict_arrays`)

- **One pass over the whole domain.** A side that the network's pooling already divides needs no padding, because circular padding then wraps the domain onto itself. Any other side gets at least 128 cells of its own periodic continuation.
- **Symmetric copies.** The prediction is averaged over the exact copies of the case: the mirror, and at 45° also the transpose.
- **Building cells.** These take the prediction of the nearest open cell, using the building map only.
- **Back to the native grid.** The result is turned back out of the canonical frame. For the cases stored at 0.5 m, the 1 m prediction is spread by periodic bilinear interpolation.

## Design checks (measured)

| Check | Result |
| --- | --- |
| Overfit 4 patches, 2,000 steps (`overfit_check.py`) | masked L1 0.983 → 0.0365, 2% of the mean target |
| Prediction mode, on fold 0's inner-validation cases (balanced RMSE) | whole domain 0.666 (margin 64 or 128 alike); tiled 256-cell windows 0.676 to 0.678. Whole-domain prediction is kept. |
| A perfect 1 m prediction on the 194 cases stored at 0.5 m | copied to sub-cells: median per-case R² 0.987 (worst 0.846), MAE 0.052 u*; bilinear: 0.992 (worst 0.890), MAE 0.030 u*, better in 193 of 194 cases. Bilinear is kept; the rest is the cost of the 1 m grid. |
| Speed per training epoch of the default network | about 95 s on an Apple M3 Pro GPU, about 52 s on a Kaggle T4 |

## Preliminary result: fold 0, one seed, before the final recipe

The first version of the recipe trained on fold 0. That version had unbalanced inner validation, selection on L1, no EMA, and copied 0.5 m predictions. It was then scored on fold 0's 97 held-out cases, the same cases as S07:

| Fold 0, 97 cases | U-Net | S07 pixel model |
| --- | --- | --- |
| Pooled R² | 0.777 | 0.658 |
| Median per-case R², idealized | 0.467 | 0.303 |
| Median per-case R², realistic | 0.560 | 0.259 |
| Mean absolute error | 0.478 u* (0.099 m/s) | 0.605 u* (0.125 m/s) |
| Cases where the U-Net has the higher per-case R² | 34 of 38 idealized, 59 of 59 realistic | |

This is one fold and one seed, and S07 scores above its own average on this fold (0.658 against 0.591 pooled over all folds). The claim against S07 rests on the five-fold, three-seed results with group-bootstrap intervals in `outputs/vs_s07.csv`.

## Running it

```bash
uv run python pipeline/S11_unet_baseline/overfit_check.py
uv run python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet.yaml --seed 0            # all five folds
uv run python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet.yaml --seed 0 --repredict  # new predictions from saved weights
uv run python pipeline/S11_unet_baseline/s11_results.py
```

`URBANWIND_DEVICE=cpu` forces the CPU. Each `--folds k` call trains one fold; the run is scored once all five folds exist.

**On Kaggle or Colab.**
1. Attach the private dataset `mahirm01/urbantales-dl-cache` on Kaggle (ask for access). On Colab, download it with the Kaggle API.
2. Clone the repository, link the cache to `data/processed/dl`, and run with the platform's own PyTorch:

```bash
git clone https://github.com/mahir-m01/Urban-Microclimate-Modelling.git repo && cd repo
mkdir -p data/processed && ln -s "$(dirname "$(find /kaggle/input -name UA25_d45.npz | head -1)")" data/processed/dl
PYTHONPATH=src:. python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet.yaml --seed 0 --folds 0
```

The cache stores the native-grid truth, so the evaluator files can be written without the raw simulation files. `kaggle_runner.py` runs a queue of trainings on all GPUs of a Kaggle machine and keeps only the small results.

**Tuning.** `tune_optuna.py` searches the learning rate, weight decay, dropout, width, depth, patch context and loss on fold 0's inner-validation groups:
- **Trials:** 25 epochs each, with median pruning, and the first trial is the default setting.
- **Workers:** several can share one study, one per GPU.
- **Deadline:** a trial cut short is recorded as pruned and is never ranked.
- **Optimism:** the inner-validation cases it tunes on are scored in folds 1 to 4. The tuned five-fold result is therefore slightly optimistic, and the frozen test set gives the clean number.

## Interfaces for later stages

| Call | Use |
| --- | --- |
| `build_model(name, in_ch, **kwargs)`, `@register(name)` | S12a adds architectures |
| `LOSSES[name](pred, y, mask)` | S12b adds losses with this signature (float32, mean over valid cells, 0 when none) |
| `python -m urbanwind.dl.train --config <yaml> --seed <n>` | every experiment; configs only set what differs from `train.DEFAULTS` |
| `predict_arrays(model, with_inputs(build_arrays(load_case(case))))` | final scoring of test cases without putting them in the cache |
| `model.train()` before `predict_native` | S14 keeps dropout on for MC dropout |

Tests that import PyTorch go in `src/urbanwind/dl/tests/` (on macOS they run in a process of their own; see the root `conftest.py`).

## Limitations

- The network sees the building map and the wind angle only. Anything about a case that the raster does not show, such as the height of the simulated domain, is out of its reach, and it has no case-level descriptors. S13 adds a case-level model.
- Predictions are made on the 1 m grid; the cost on the 0.5 m cases is the ceiling in the table above.
- The validation cases of each fold are not used for training, unless the run sets `inner_val_fraction: 0`.
