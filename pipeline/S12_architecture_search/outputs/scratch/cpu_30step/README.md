# Preliminary CPU 30-step smoke test results (SCRATCH — do not use as final validation)

These JSON files were produced by local 30-step CPU smoke tests. They exist solely
to confirm basic gradient flow and verify that models do not collapse before submitting
full 2,000-step jobs to Kaggle GPUs.

## Summary of 30-step smoke test checks

| File | Steps | Device | Initial Loss | Final Loss | Mean Target | Relative Loss | Status |
|---|---|---|---|---|---|---|---|
| `overfit_check_res_unet.json` | 30 | cpu | 0.4404 | 0.2509 | 1.0002 | 0.2508 | Fixed initialisation; learns stably without collapse |
| `overfit_check_attn_unet.json` | 30 | cpu | 1.5342 | 0.2500 | 1.0002 | 0.2500 | Learns stably |
| `overfit_check_fno.json` | 30 | cpu | 0.5673 | 0.2376 | 1.0002 | 0.2376 | Learns stably |

## Note on 2,000-step validation

Authoritative 2,000-step overfit validation results (matching the S11 baseline protocol)
must be executed on GPU (e.g. Kaggle) via:
```bash
uv run python pipeline/S12_architecture_search/overfit_check.py --model all --steps 2000
```
and saved directly to `pipeline/S12_architecture_search/outputs/`.
Do NOT cite these 30-step preliminary results as evidence of full convergence.
