'''
plot_right_peak.py

R vs HV, keeping only the spectra where the 511 keV peak is in the
"right" part of the spectrum (mu >= MU_MIN). This is required for the
Compton measurement: the scattered peak at 170 keV (theta = 180 deg)
falls at ch(511)/3, which must be above the threshold noise and above
the Compton edge of the 511 (at ~ch(511)*0.67).

Produces:
    - scatter of R vs V (filtered), one color per G
    - fit R(V) per (G, S) where enough points survive (n >= 3)
    - summary table with the filter outcome

Workflow
--------
    $ uv run plot_right_peak.py
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab_compton.config import parse_filename
from lab_compton.resolution_scan import fit_parabola_R_V, fit_pmt_R_V


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH     = PROJECT_ROOT / "results" / "tables" / "optimization" / "all_fits.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures" / "optimization"
TAB_DIR      = PROJECT_ROOT / "results" / "tables" / "optimization"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# 511 keV selection (same as full_analysis.py)
R_MIN = 0.055
R_MAX = 0.100

# Dead time cutoff
DEAD_MAX = 20.0

# "Right side" threshold on the centroid
MU_MIN = 1000          # ch(170) = MU_MIN / 3 ~ 333, comfortable

# Fit method: "parabola" (robust with few points), "pmt" (needs >= 4)
FIT_METHOD = "parabola"

# Min points per group to attempt a fit
MIN_POINTS_FIT = 3

# Plot format
FIG_FORMAT = "pdf"
SAVE_KW    = dict(bbox_inches="tight", dpi=200)


# =========================================================================
# Load and filter
# =========================================================================
def load_right_peak():
    df = pd.read_csv(CSV_PATH)
    print(f"Rows in CSV: {len(df)}")

    # 511 keV + dead time
    df = df[(df["R"] > R_MIN) & (df["R"] < R_MAX) & (df["dead_pct"] < DEAD_MAX)].copy()

    # Deduplicate: keep the peak with the lowest D/ndof per file
    def _d(s):
        try:
            return float(str(s).split("=")[-1].strip())
        except (ValueError, AttributeError):
            return np.nan
    df["D_ndof_val"] = df["D_ndof"].apply(_d)
    df = df.sort_values("D_ndof_val").drop_duplicates(subset=["file"], keep="first")

    # Parse V, G, S
    parsed = df["file"].apply(
        lambda n: pd.Series(parse_filename(n), index=["V", "G", "S"])
    )
    df[["V", "G", "S"]] = parsed
    df["valid"] = True

    # Right-side filter
    n_before = len(df)
    df = df[df["mu"] >= MU_MIN].copy()
    n_after = len(df)
    print(f"Before mu >= {MU_MIN}: {n_before} points")
    print(f"After  mu >= {MU_MIN}: {n_after} points")
    print("\nPoints passing the filter:")
    print(df[["file", "V", "G", "S", "mu", "R", "R_err"]]
          .sort_values(["V", "G"]).to_string(index=False))

    return df


# =========================================================================
# Scatter plot: R vs V, colored by G
# =========================================================================
def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    Gs = sorted(df["G"].unique())
    cmap = plt.cm.viridis
    colors = {G: cmap(i / max(len(Gs) - 1, 1)) for i, G in enumerate(Gs)}

    for G in Gs:
        sub = df[df["G"] == G].sort_values("V")
        c = colors[G]
        ax.errorbar(sub["V"], sub["R"] * 100, yerr=sub["R_err"] * 100,
                    fmt="o", capsize=3, color=c, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"G = {G:g}", zorder=5)
        # Annotate each point with its mu value
        for _, r in sub.iterrows():
            ax.annotate(f"ch={r['mu']:.0f}",
                        (r["V"], r["R"] * 100), xytext=(6, 6),
                        textcoords="offset points", fontsize=7,
                        color=c)

    # Reference line: best R from full analysis (7.24%)
    ax.axhline(7.24, color="gray", ls=":", lw=1, alpha=0.6,
               label="R=7.24% (best from full scan)")

    ax.set_xlabel("HV (V)")
    ax.set_ylabel("R (%) at 511 keV")
    ax.set_title(f"R vs HV — only spectra with 511 keV at ch >= {MU_MIN}\n"
                 f"(Compton: 170 keV at ch >= {MU_MIN/3:.0f})")
    ax.grid(alpha=0.3)
    ax.legend(title="Gain", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"R_vs_HV_right_peak.{FIG_FORMAT}", **SAVE_KW)
    plt.show()


# =========================================================================
# Fit R(V) per (G, S) where enough points survive
# =========================================================================
def fit_and_plot_groups(df):
    print("\n=== Fits per (G, S) with >= 3 points ===")
    fig, ax = plt.subplots(figsize=(9, 5))
    Gs = sorted(df["G"].unique())
    cmap = plt.cm.viridis
    colors = {G: cmap(i / max(len(Gs) - 1, 1)) for i, G in enumerate(Gs)}

    any_fit = False
    for (G, S), sub in df.groupby(["G", "S"]):
        sub = sub.sort_values("V")
        c = colors.get(G, "gray")

        # Scatter
        ax.errorbar(sub["V"], sub["R"] * 100, yerr=sub["R_err"] * 100,
                    fmt="o", capsize=3, color=c, markersize=8,
                    markeredgecolor="black", label=f"G={G:g}, S={S:g}")

        if len(sub) < MIN_POINTS_FIT:
            print(f"  G={G:g}, S={S:g}: {len(sub)} points → no fit")
            continue

        try:
            if FIT_METHOD == "pmt":
                res = fit_pmt_R_V(sub)
                V_opt, R_opt = res["V_opt"], res["R_opt"]
                # Draw PMT curve
                V_fit = np.linspace(sub["V"].min(), sub["V"].max(), 200)
                from lab_compton.resolution_scan import _pmt_model
                ax.plot(V_fit, _pmt_model(V_fit, *res["popt"]) * 100,
                        "-", color=c, alpha=0.6)
                print(f"  G={G:g}, S={S:g}: n={len(sub)}, "
                      f"V_opt={V_opt:.0f} V, R_opt={100*R_opt:.3f}%, "
                      f"D/ndof={res['chi2']/res['ndof']:.2f}")
            else:
                res = fit_parabola_R_V(sub)
                V0, R0 = res["V0"], res["R0"]
                # Draw parabola
                V_fit = np.linspace(sub["V"].min(), sub["V"].max(), 200)
                ax.plot(V_fit, (R0 + res["k"] * (V_fit - V0) ** 2) * 100,
                        "-", color=c, alpha=0.6)
                print(f"  G={G:g}, S={S:g}: n={len(sub)}, "
                      f"V0={V0:.0f} V, R0={100*R0:.3f}%, "
                      f"D/ndof={res['chi2']/res['ndof']:.2f}")
            any_fit = True
        except (ValueError, RuntimeError) as e:
            print(f"  G={G:g}, S={S:g}: {len(sub)} points → fit failed: {e}")

    ax.set_xlabel("HV (V)")
    ax.set_ylabel("R (%) at 511 keV")
    ax.set_title(f"R(V) with {FIT_METHOD} fit — 511 keV at ch >= {MU_MIN}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"R_vs_HV_right_peak_fit.{FIG_FORMAT}", **SAVE_KW)
    plt.show()

    if not any_fit:
        print("\nNo group had >= 3 points with the filter. "
              "Consider lowering MU_MIN or adding more points.")


# =========================================================================
# Main
# =========================================================================
def main():
    df = load_right_peak()
    if df.empty:
        print(f"\nNo points with mu >= {MU_MIN}. Lower the threshold.")
        return

    df.to_csv(TAB_DIR / "right_peak_points.csv", index=False)
    print(f"\nFiltered table saved to {TAB_DIR / 'right_peak_points.csv'}")

    plot_scatter(df)
    fit_and_plot_groups(df)


if __name__ == "__main__":
    main()
