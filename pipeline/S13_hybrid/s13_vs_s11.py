"""
S13: a candidate that passed the fold-0 screen, against the S11 tuned U-Net on all five folds.

Scores the candidate and the three S11 tuned seeds on the 482 development cases, and compares the
candidate with the S11 run of the same training seed case by case, with 95% intervals from resampling
whole cities and layout families (2,000 times, as in S11). Writes outputs/<candidate>_summary.csv and
outputs/<candidate>_vs_s11.csv.

Usage: uv run python -m pipeline.S13_hybrid.s13_vs_s11 [candidate run, default s13_unet_tuned_e200_seed0]
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from pipeline.S11_unet_baseline.s11_results import bootstrap_vs_s07, score_run, summarise
from urbanwind.dl.train import fold_table

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
S11_SEEDS = [f"s11_unet_tuned_seed{s}" for s in range(3)]


def main(candidate: str = "s13_unet_tuned_e200_seed0") -> None:
    dev = fold_table("grouped")
    runs = list(dict.fromkeys([candidate, *S11_SEEDS]))
    per_case = pd.concat([score_run(run, dev).assign(split="grouped") for run in runs], ignore_index=True)
    counts = per_case.groupby("run")["case"].nunique()
    if (counts != len(dev)).any():
        raise SystemExit(f"every run needs all {len(dev)} development cases: {counts.to_dict()}")
    ref = f"s11_unet_tuned_seed{candidate.rsplit('seed', 1)[1]}"  # the S11 run with the same training seed
    summary = summarise(per_case)
    comparison = bootstrap_vs_s07(per_case, candidate, ref=ref).assign(ref_run=ref)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / f"{candidate}_summary.csv", index=False, float_format="%.4f")
    comparison.to_csv(OUT / f"{candidate}_vs_s11.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(summary.to_string(index=False), "\n")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
