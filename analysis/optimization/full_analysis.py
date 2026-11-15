'''
full_analysis.py

Read the CSV produced by optimization.py, filter the 511 keV peak, fit
R(V) for every (G, S), find the best setting, and produce plots.

Two optional steps:
  - calibration (from calibration.py) to convert R from channels to
    energy. Not required for finding V_opt (the absolute scale is
    irrelevant), but useful if you want to report absolute R in the
    final tables.
  - inspect the fit of a single (G, S) group in detail via plot_scan.

Workflow
--------
    # 1. produce the CSV (once)
    $ uv run optimization.py

    # 2. analyze it
    $ uv run full_analysis.py
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab_compton.config import parse_filename
from lab_compton.resolution_scan import (
    fit_groups, best_setting,
)
from lab_compton.scan_plotting import (
    plot_groups, plot_optimum_map, plot_scan,
    print_group_summary, print_summary,
)

import lab_compton.parameters as par


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH     = PROJECT_ROOT / "results" / "tables" / "optimization" / "all_fits.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures" / "optimization"
TAB_DIR      = PROJECT_ROOT / "results" / "tables" / "optimization"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Acceptable R range for the 511 keV peak (with zero_channel=0).
# Typical: 0.07-0.08. Excludes the 1274 keV (~0.05) and the Compton
# edge (~0.32).
R_MIN = 0.055
R_MAX = 0.100

# Best-setting method: "pmt" (physical model, needs a well-defined
# minimum) or "data" (lowest measured R, works even with a flat curve).
BEST_METHOD = "data"

# If True, print and save a detailed diagnostic of the group with the
# most points (typically G=20 at fixed S).
SHOW_BEST_GROUP = True


# =========================================================================
# 1. Load and clean
# =========================================================================
df = pd.read_csv(CSV_PATH)
print(f"Rows in CSV: {len(df)}")

# Filter by R
df_clean = df[(df["R"] > 0.055) & (df["R"] < 0.100) & (df["dead_pct"] < 20)]

# Deduplicate: keep the peak with the lowest reduced deviance per file
def _parse_d_ndof(s):
    try:
        return float(str(s).split("=")[-1].strip())
    except (ValueError, AttributeError):
        return np.nan

df_clean["D_ndof_val"] = df_clean["D_ndof"].apply(_parse_d_ndof)
df_clean = (df_clean
            .sort_values("D_ndof_val")
            .drop_duplicates(subset=["file"], keep="first"))

# Extract V, G, S from the filename using the canonical parser
parsed = df_clean["file"].apply(
    lambda n: pd.Series(parse_filename(n), index=["V", "G", "S"])
)
df_clean[["V", "G", "S"]] = parsed

# Mark as valid (filter already did the job)
df_clean["valid"] = True

print(f"Rows after filter (R in [{R_MIN}, {R_MAX}], 1 peak per file): {len(df_clean)}")
print(df_clean[["file", "mu", "R", "R_err"]].to_string(index=False))


# =========================================================================
# 2. (optional) Energy calibration
# =========================================================================
# Uncomment and fill with your calibration points if you have standard
# sources (Cs-137, Co-60, ...). Skip if you only want the channel R.
#
# from lab_compton.calibration import calibrate, apply_calibration, print_calibration
#
# calib = calibrate([
#     (mu_cs,   err_cs,   661.7),
#     (mu_co1,  err_co1,  1173.2),
#     (mu_co2,  err_co2,  1332.5),
# ], degree=1)
# print_calibration(calib)
# df_clean = apply_calibration(df_clean, calib)


# =========================================================================
# 3. Fit R(V) per (G, S)
# =========================================================================
print("\nFitting R(V) for every (G, S)...")
results, summary = fit_groups(df_clean, by=("G", "S"), minos=True)
print_group_summary(summary)

summary.to_csv(TAB_DIR / "fit_summary.csv", index=False)
print(f"\nFit summary saved to {TAB_DIR / 'fit_summary.csv'}")


# =========================================================================
# 4. Best setting
# =========================================================================
print(f"\n=== Best setting (method: {BEST_METHOD}) ===")
best_series = None
try:
    best = best_setting(summary, method=BEST_METHOD, n_sigma=1.0)
    best_series = best["best"]
    b = best_series
    v_opt_str = (f"V_opt = {b['pmt_V_opt']:.0f} V, R_opt = {100*b['pmt_R_opt']:.3f}%"
                 if BEST_METHOD == "pmt" else
                 f"R_min = {100*b['R_data_min']:.3f}% at V = {b['V_data_min']:.0f} V")
    print(f"Best: G={b['G']}, S={b['S']}  ->  {v_opt_str}")
    print("\nCompatible within 1 sigma:")
    cols = ["G", "S",
            "pmt_V_opt" if BEST_METHOD == "pmt" else "V_data_min",
            "pmt_R_opt" if BEST_METHOD == "pmt" else "R_data_min",
            "pmt_R_opt_err" if BEST_METHOD == "pmt" else "R_data_min_err"]
    cols = [c for c in cols if c in best["compatible"]]
    print(best["compatible"][cols].to_string(index=False))
except ValueError as e:
    print(f"Cannot determine best setting: {e}")
    print("Suggestion: the R(V) curve is probably flat (no clear minimum).")
    print("Try BEST_METHOD = 'data' to pick the lowest measured R, or")
    print("add HV points at lower voltage to constrain the minimum.")


# =========================================================================
# 5. Plots: R(V) for each (G, S) + optimum map
# =========================================================================
print("\nGenerating plots...")

# 5a. R(V) panels per (G, S)
try:
    fig = plot_groups(df_clean, results, ncols=3)
    fig.suptitle("R(V) — 511 keV peak only", fontsize=14)
    fig.savefig(FIG_DIR / "R_vs_HV_groups.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"plot_groups error: {e}")

# 5b. Optimum map (best_series may be None, plot_optimum_map handles it)
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_optimum_map(summary, method=BEST_METHOD, value="R",
                     best=best_series, ax=axes[0])
    plot_optimum_map(summary, method=BEST_METHOD, value="V",
                     best=best_series, ax=axes[1])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "optimum_map.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"optimum map error: {e}")




# =========================================================================
# 6. (optional) Detailed diagnostic of the most-populated group
# =========================================================================
if SHOW_BEST_GROUP and results:
    key_max = max(results, key=lambda k: int(results[k]["df"]["valid"].sum()))
    n_pts   = int(results[key_max]["df"]["valid"].sum())
    if n_pts >= 2:
        r = results[key_max]
        title = ", ".join(f"{n} = {v:g}" for n, v in zip(("G", "S"), key_max))
        print(f"\n=== Detailed diagnostic: {title} ({n_pts} points) ===")

        # Text summary
        print_summary(r["df"], r["parabola"], r["pmt"])

        # Plot: data + both fits
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_scan(r["df"], r["parabola"], r["pmt"], ax=ax, title=title)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "best_group_R_vs_HV.png", dpi=150, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print(f"best-group plot error: {e}")
            
from lab_compton.scan_plotting import (
    plot_R_vs_G, plot_R_vs_G_grid, print_summary_R_vs_G,
)

# =========================================================================
# 7. R vs G (dependence on the amplifier gain)
# =========================================================================
print("\nR vs G analysis...")

# Use df_clean (which has V, G, S after parsing the filename)
fig, ax = plt.subplots(figsize=(9, 5))
plot_R_vs_G(df_clean, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "R_vs_G.pdf", bbox_inches="tight")
plt.show()

fig = plot_R_vs_G_grid(df_clean, ncols=3)
if fig is not None:
    fig.savefig(FIG_DIR / "R_vs_G_grid.pdf", bbox_inches="tight")
    plt.show()

print_summary_R_vs_G(df_clean)
