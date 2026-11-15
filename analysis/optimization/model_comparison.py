'''
model_comparison_all.py

Fit the 511 keV peak of every spectrum with EVERY model in MODELS
(all signals × all backgrounds × step on/off), and produce:

    - full table (model × file) of D/ndof and R
    - median D/ndof per model (horizontal bar, sorted)
    - heatmap D/ndof (model × file) — most informative view
    - R spread per file across all models

This is an EXPLORATORY script: it includes ALL models, even physically
implausible ones (flat, poly3, exppoly2) and step variants that may not
be needed. For the systematic uncertainty in the report, restrict to
the plausible list (see systematics_study.py).

Expected runtime: 60 models × 11 files × ~0.3-1 s = 3-10 minutes.

Workflow
--------
    $ uv run model_comparison_all.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lab_compton.config import parse_filename
from lab_compton.maestro_reader import read_dir
from lab_compton.spectrum_fit import fit_peak, _find_peak_window, MODELS


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "Optimization" / "Scintillator1" / "Spe"
TAB_DIR      = PROJECT_ROOT / "results" / "tables" / "optimization"
FIG_DIR      = PROJECT_ROOT / "results" / "figures" / "optimization"

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ALL models from the registry (5 signals × 6 backgrounds × 2 step = 60)
MODELS_TO_TRY = tuple(MODELS.keys())

# 511 keV selection
R_MIN_511 = 0.055
R_MAX_511 = 0.100
D_NDOF_MAX = 5.0        # very permissive: keep almost everything
MIN_CHANNEL = 50

FIG_FORMAT = "pdf"
SAVE_KW = dict(bbox_inches="tight", dpi=200)
VERBOSE_FITS = False    # True to print each fit


# =========================================================================
# Fit every model on every spectrum (same window per file)
# =========================================================================
def fit_all_combinations():
    spectra = read_dir(DATA_DIR, pattern="*.Spe")
    print(f"Found {len(spectra)} spectra, {len(MODELS_TO_TRY)} models")
    print(f"Total fits: {len(spectra) * len(MODELS_TO_TRY)}")

    rows = []
    for name, s in spectra.items():
        key =  parse_filename(name)
        if key is None:
            print(f"[skip] {name}: unparseable")
            continue
        V, G, S = key

        w, found, fwhm_est = _find_peak_window(
            s, mu_ref=None, min_channel=MIN_CHANNEL, fit_half_fwhm=2.5,
        )
        if not found:
            print(f"[skip] {name}: no peak found")
            continue

        print(f"\n{name}  (V={V}, G={G}, S={S})  window={w}")

        for model in MODELS_TO_TRY:
            try:
                fit = fit_peak(s.channels, s.counts, window=w, model=model)
            except Exception as e:
                if VERBOSE_FITS:
                    print(f"  [{model:24s}] FAIL: {e}")
                continue

            R, R_err = fit.peak_resolution(zero_channel=0.0)
            d_ndof = fit.deviance / fit.ndof

            # Keep only 511 keV (R in range); keep all D/ndof for exploration
            if not (R_MIN_511 < R < R_MAX_511):
                if VERBOSE_FITS:
                    print(f"  [{model:24s}] R={100*R:.2f}%  [not 511]")
                continue

            mu, mu_err = fit.centroid()
            rows.append({
                "name": name, "V": V, "G": G, "S": S,
                "model": model,
                "mu": mu, "mu_err": mu_err,
                "R": R, "R_err": R_err,
                "deviance": fit.deviance, "ndof": fit.ndof,
                "D_ndof": d_ndof,
                "valid": fit.valid,
            })
            if VERBOSE_FITS:
                print(f"  [{model:24s}] D/ndof={d_ndof:.2f}  R={100*R:.3f}%")

    return pd.DataFrame(rows)


# =========================================================================
# Summary
# =========================================================================
def print_summary(df):
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    n_files  = df["name"].nunique()
    n_models = df["model"].nunique()
    print(f"\n{n_files} files, {n_models} models with valid 511 keV fit")
    print(f"Total rows: {len(df)}")

    # How many models per file
    counts = df.groupby("name")["model"].nunique()
    print("\nModels with valid 511 keV fit, per file:")
    for name, n in counts.items():
        print(f"  {name:22s}: {n:3d} models")

    # Median D/ndof per model
    print("\n=== Median D/ndof per model (sorted) ===")
    med = df.groupby("model")["D_ndof"].agg(["median", "min", "max", "count"])
    med = med.sort_values("median")
    print(med.round(3).to_string())

    # Median R per model
    print("\n=== Median R (%) and std per model ===")
    med_r = df.groupby("model")["R"].agg(["median", "std", "count"]).copy()
    med_r["median"] *= 100
    med_r["std"] *= 100
    med_r.columns = ["median_%", "std_%", "count"]
    print(med_r.round(3).to_string())

    # Best model per file
    print("\n=== Best model per file (lowest D/ndof) ===")
    best = df.loc[df.groupby("name")["D_ndof"].idxmin()]
    for _, r in best.iterrows():
        print(f"  {r['name']:22s}  best: {r['model']:24s}  "
              f"D/ndof={r['D_ndof']:.3f}  R={100*r['R']:.3f}%")

    # R spread per file
    print("\n=== R spread per file (max-min)/2 across ALL models ===")
    spread = df.groupby("name").agg(
        R_mean=("R", "mean"),
        R_spread=("R", lambda s: (s.max() - s.min())/2 if len(s) > 1 else np.nan),
        n_models=("R", "count"),
    ).reset_index()
    spread["R_mean_pct"]   = spread["R_mean"] * 100
    spread["R_spread_pct"] = spread["R_spread"] * 100
    spread["rel_%"] = (spread["R_spread"] / spread["R_mean"]) * 100
    print(spread[["name", "R_mean_pct", "R_spread_pct", "rel_%", "n_models"]]
          .to_string(index=False))

    print(f"\nMedian R_spread: {spread['R_spread_pct'].median():.4f}% "
          f"(relative: {spread['rel_%'].median():.3f}%)")

    return spread


# =========================================================================
# Plots
# =========================================================================
def make_plots(df, spread):
    # --- 1. Median D/ndof bar chart (all models, sorted) ---
    med = df.groupby("model")["D_ndof"].median().sort_values()
    n = len(med)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * n + 2)))
    y = np.arange(n)
    colors = ["green" if v < 1.1 else ("orange" if v < 1.5 else "red")
              for v in med.values]
    ax.barh(y, med.values, color=colors, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(med.index, fontsize=7)
    ax.axvline(1.0, color="black", ls="--", alpha=0.5, label="D/ndof = 1")
    ax.set_xlabel("median D/ndof")
    ax.set_title("Median D/ndof per model (sorted, all models)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"model_comparison_D_ndof_median.{FIG_FORMAT}", **SAVE_KW)
    plt.show()

    # --- 2. Heatmap: D/ndof per (model, file) ---
    pv = df.pivot_table(index="model", columns="name", values="D_ndof",
                        aggfunc="mean")
    pv = pv.loc[med.index]   # sort rows by median D/ndof

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(pv.columns) + 3),
                                     max(6, 0.22 * len(pv.index) + 2)))
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.8, vmax=2.0)

    ax.set_yticks(np.arange(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=6)
    ax.set_xticks(np.arange(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=45, ha="right", fontsize=7)

    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5)

    ax.set_title("D/ndof per (model, file) — green better, red worse")
    fig.colorbar(im, ax=ax, label="D/ndof")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"model_comparison_D_ndof_heatmap.{FIG_FORMAT}", **SAVE_KW)
    plt.show()

    # --- 3. R spread per file ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(spread))
    ax.bar(x, spread["R_spread_pct"], color="steelblue", label="per-file spread")
    ax.axhline(spread["R_spread_pct"].median(), color="red", ls="--",
               label=f"median = {spread['R_spread_pct'].median():.3f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(spread["name"], rotation=45, ha="right")
    ax.set_ylabel("R spread (%) — (max-min)/2 across all models")
    ax.set_title("Model systematics per file (all 60 models)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"model_comparison_R_spread.{FIG_FORMAT}", **SAVE_KW)
    plt.show()


# =========================================================================
# Main
# =========================================================================
def main():
    print(f"Data dir: {DATA_DIR}")
    print(f"Models ({len(MODELS_TO_TRY)}): {', '.join(MODELS_TO_TRY)}")
    print()

    df = fit_all_combinations()

    if df.empty:
        print("\nNo fits succeeded.")
        return

    df.to_csv(TAB_DIR / "model_comparison_all.csv", index=False)
    print(f"\nFull table saved to {TAB_DIR / 'model_comparison_all.csv'}")

    spread = print_summary(df)
    spread.to_csv(TAB_DIR / "model_comparison_all_R_spread.csv", index=False)

    make_plots(df, spread)


if __name__ == "__main__":
    main()
