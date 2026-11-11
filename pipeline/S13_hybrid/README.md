# S13 Hybrid ML + DL, and beating the S11 U-Net

Asks where the S11 U-Net's remaining error lies, whether the Phase 1 tree models can fix it (the hybrid), and whether a different network or a longer schedule does better.

| | |
| --- | --- |
| Inputs | the S11 tuned predictions (`results/preds/s11_unet_tuned_seed{0,1,2}/`, not tracked), `pipeline/S11_unet_baseline/outputs/per_case_metrics.csv`, the S06c out-of-fold predictions, the S10 cache |
| Outputs | `outputs/level_test.csv`, `outputs/error_anatomy.csv`, `outputs/fold0_screen.csv`; `outputs/runs/<exp_id>/fold<k>.json` and `.csv` |
| Code | `s13_level_pattern.py`, `s13_fold0.py`, `configs/`; `ConvNeXtUNet` in `src/urbanwind/dl/nets.py`; tests in `src/urbanwind/dl/tests/test_s13_*.py` |

Status: in progress. The level test and the error anatomy are complete; the fold-0 screen of two candidates is running.

## 1. Where the S11 error is (`s13_level_pattern.py`)

No training: both measurements use the saved predictions of the three S11 tuned seeds.

### Error anatomy (`outputs/error_anatomy.csv`)

- **Realistic cases are the problem.** They hold 84% of the scored cells but 97% to 98% of the squared error. The overall pooled R² is therefore close to the realistic one.
- **Within realistic cases, the error is the pattern, not the level.** 85% to 89% of the realistic squared error is the pattern inside each case; 11% to 15% is the case's overall speed level. The truth itself varies mostly within cases: 70% of the realistic variance is within cases.
- **The error is spread out.** The 20 worst realistic cases hold 28% to 31% of the realistic squared error and 15% of its cells, so no handful of cases explains the score.

### Level test (`outputs/level_test.csv`)

The hybrid idea H1 lets the Phase 1 LightGBM model set each case's level. Done after the fact, each case's prediction is multiplied by one factor so its open-ground mean equals a chosen level. The LightGBM level for a case is the S06c out-of-fold prediction, from a model that never saw its group.

| Level taken from | Pooled R², seeds 0 to 2 | Pooled R², mean of the 3 seeds | Realistic pooled R², mean of the 3 seeds |
| --- | --- | --- | --- |
| U-Net (unchanged) | 0.7092, 0.7122, 0.7111 | 0.7194 | 0.6874 |
| LightGBM | 0.6980, 0.6997, 0.7109 | 0.7111 | 0.6779 |
| Average of the two | 0.7089, 0.7114, 0.7173 | 0.7207 | 0.6887 |
| True mean (upper bound) | 0.7435, 0.7452, 0.7553 | 0.7568 | 0.7297 |

- The U-Net already predicts each case's mean speed as well as LightGBM does: R² of the case means 0.874 to 0.896 for the three seeds, against 0.876 for LightGBM.
- Replacing the level with LightGBM's lowers the score; averaging the two changes it by -0.001 to +0.006.
- Even the true level, which no model has, lifts pooled R² only to about 0.74 to 0.76.

So a level-anchored hybrid cannot add much, and the remaining gains must come from the pattern within cases.

## 2. Candidates for a better network

Each candidate trains fold 0 with seed 0 under the S11 protocol and tuned settings. It counts as better only if it beats the best of the three S11 tuned seeds on fold 0's 97 cases (pooled R² 0.789 to 0.802, median per-case R² 0.567 to 0.633); only then does it get all five folds.

- **`unet_tuned_e200`: the S11 tuned U-Net for 200 epochs instead of 100.** In 14 of the 15 S11 tuned trainings the best inner-validation error came at the last epoch, and it fell by 0.005 RMSE on average over the last 20 epochs (`pipeline/S11_unet_baseline/outputs/runs/s11_unet_tuned_seed*/fold*.csv`).
- **`convnext_unet`: the U-Net layout built from ConvNeXt blocks** (`ConvNeXtUNet`, 5,461,281 parameters against the tuned U-Net's 4,860,313):
  - each block has a 7×7 depthwise convolution with circular padding, a per-cell LayerNorm, a 1×1 expansion to four times the channels with GELU, a 1×1 projection back, and a residual connection with a layer scale;
  - the larger filters see more of the street around each cell at every level;
  - it has no positional or coordinate input, and a float64 test checks that shifts by multiples of 16 cells commute with the network exactly.

Results: pending (`outputs/fold0_screen.csv`).

## Running it

```bash
uv run python pipeline/S13_hybrid/s13_level_pattern.py      # needs the S11 tuned predictions
uv run python -m urbanwind.dl.train --config pipeline/S13_hybrid/configs/convnext_unet.yaml --seed 0 --folds 0
uv run python -m pipeline.S13_hybrid.s13_fold0               # scores every finished candidate on fold 0
```

The candidates were trained on Kaggle T4 GPUs with `pipeline/S11_unet_baseline/kaggle_runner.py`, and their fold-0 predictions rebuilt locally with `--repredict`.

## Limitations

- The level test rescales finished predictions; a network trained to predict U/U_mean could behave differently, but the true-mean upper bound limits what any level correction can add.
- The fold-0 screen uses one seed and 97 cases. The S11 seed spread on those cases is wide for the median per-case R² (0.567 to 0.633), so only a clear win moves on to all five folds.
