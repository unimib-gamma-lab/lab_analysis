'''
best_for_compton.py

Find the best (V, G) configuration for the Compton measurement.

The objective is:
    minimize R(V, G)
    subject to  mu(V, G) >= MU_MIN

Working assumption: mu is proportional to V^alpha * G, with alpha = PMT_ALPHA
(typical 5-7 for a PMT). Then the "effective gain"
    A_eff = V^alpha * G
characterizes the signal size. Configurations with the same A_eff have
the same mu, so comparing R among them shows where the noise/saturation
balance is best.

Produces:
    - scatter of R vs A_eff (only points with mu >= MU_MIN)
    - a sorted table: best configuration, alternatives, gain_eff
    - a chart with the current best marked

Workflow
--------
    $ uv run best_for_compton.py
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab_compton.config import parse_filename


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH     = PROJECT_ROOT / "results" / "tables" / "optimization" / "all_fits.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures" / "optimization"
TAB_DIR      = PROJECT_ROOT / "results" / "tables" / "optimization"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

R_MIN      = 0.055
R_MAX      = 0.100
DEAD_MAX   = 20.0
MU_MIN     = 1000      # 511 keV at channel >= this

# Exponent of the PMT gain (mu ~ V^alpha * G). Typical: 5-7.
# Set to None to *fit* alpha from the data (needs >= 3 points with
# the same G and different V, or vice versa).
PMT_ALPHA = 6.0

# Plot
FIG_FORMAT = "pdf"
SAVE_KW    = dict(bbox_inches="tight", dpi=200)


# =========================================================================
# Load
# =========================================================================
def load():
    df = pd.read_csv(CSV_PATH)
    print(f"Rows in CSV: {len(df)}")

    # Basic filter (511 keV, low dead time)
    df = df[(df["R"] > R_MIN) & (df["R"] < R_MAX) & (df["dead_pct"] < DEAD_MAX)].copy()

    # Deduplicate: one row per file
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

    # Keep only points with 511 keV at high channel
    n_before = len(df)
    df = df[df["mu"] >= MU_MIN].copy()
    n_after = len(df)

    print(f"Before mu >= {MU_MIN}: {n_before} points")
    print(f"After  mu >= {MU_MIN}: {n_after} points")

    # Effective gain
    df["A_eff"] = df["V"] ** PMT_ALPHA * df["G"]

    # Sort by R
    df = df.sort_values("R").reset_index(drop=True)

    print("\nPoints passing the filter, sorted by R:")
    print(df[["file", "V", "G", "S", "mu", "R", "R_err", "A_eff"]]
          .to_string(index=False))

    return df


# =========================================================================
# Best configuration
# =========================================================================
def best_point(df):
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOR COMPTON (mu >= %.0f)" % MU_MIN)
    print("="*80)
    if df.empty:
        print("No points pass the filter. Lower MU_MIN.")
        return

    b = df.iloc[0]
    print(f"  V     = {b['V']:.0f} V")
    print(f"  G     = {b['G']:.0f}")
    print(f"  S     = {b['S']:.2f} us")
    print(f"  mu    = {b['mu']:.1f} ch  (511 keV)")
    print(f"  R     = {100*b['R']:.3f}% ± {100*b['R_err']:.3f}%")
    print(f"  A_eff = {b['A_eff']:.2e}")

    # Alternatives within 2 sigma
    tol = 2 * b["R_err"]
    alt = df[df["R"] - b["R"] <= tol]
    if len(alt) > 1:
        print(f"\n  Alternatives within 2 sigma ({2*tol*100:.2f}%):")
        print(alt[["V", "G", "S", "mu", "R", "R_err"]]
              .to_string(index=False))


# =========================================================================
# Plot: R vs A_eff (only points passing the filter)
# =========================================================================
def plot_R_vs_Aeff(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = df["A_eff"].values
    y = df["R"].values * 100
    yerr = df["R_err"].values * 100

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color="steelblue",
                markersize=10, markeredgecolor="black")

    # Annotate each point with (V, G)
    for xi, yi, (_, r) in zip(x, y, df.iterrows()):
        ax.annotate(f"({r['V']:.0f}, {r['G']:.0f})",
                    (xi, yi), xytext=(8, 5), textcoords="offset points",
                    fontsize=8, color="darkblue")

    # Highlight best
    b = df.iloc[0]
    ax.scatter([b["A_eff"]], [b["R"] * 100],
               marker="*", s=400, facecolor="gold",
               edgecolor="red", linewidth=2, zorder=10,
               label=f"best: V={b['V']:.0f}, G={b['G']:.0f}, R={b['R']*100:.2f}%")

    ax.set_xscale("log")
    ax.set_xlabel(f"A_eff = V^{PMT_ALPHA:.0f} · G (arbitrary units, log scale)")
    ax.set_ylabel("R (%)")
    ax.set_title(f"R vs A_eff — 511 keV at ch >= {MU_MIN}\n"
                 f"(iso-A_eff: same signal size; lower R is better)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"best_for_compton.{FIG_FORMAT}", **SAVE_KW)
    plt.show()


# =========================================================================
# Plot: R vs mu (to verify the "right side" condition)
# =========================================================================
def plot_R_vs_mu(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = df["mu"].values
    y = df["R"].values * 100
    yerr = df["R_err"].values * 100

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color="steelblue",
                markersize=10, markeredgecolor="black")

    # Annotate each point with (V, G)
    for xi, yi, (_, r) in zip(x, y, df.iterrows()):
        ax.annotate(f"({r['V']:.0f}, {r['G']:.0f})",
                    (xi, yi), xytext=(8, 5), textcoords="offset points",
                    fontsize=8, color="darkblue")

    b = df.iloc[0]
    ax.scatter([b["mu"]], [b["R"] * 100],
               marker="*", s=400, facecolor="gold",
               edgecolor="red", linewidth=2, zorder=10,
               label=f"best: R={b['R']*100:.2f}%")

    ax.axvline(MU_MIN, color="gray", ls=":", label=f"mu_min = {MU_MIN}")

    ax.set_xlabel("mu (channel of 511 keV peak)")
    ax.set_ylabel("R (%)")
    ax.set_title(f"R vs mu — all points with mu >= {MU_MIN}\n"
                 f"(higher mu = more room for scattered Compton)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"R_vs_mu_right_peak.{FIG_FORMAT}", **SAVE_KW)
    plt.show()


# =========================================================================
# Suggested additional measurements (informational)
# =========================================================================
def suggest_measurements(df):
    print("\n" + "="*80)
    print("SUGGESTIONS FOR ADDITIONAL MEASUREMENTS")
    print("="*80)
    if df.empty:
        print("No points. Raise/lower MU_MIN first.")
        return

    b = df.iloc[0]
    target_Aeff = b["A_eff"]
    print(f"Current best is (V={b['V']:.0f}, G={b['G']:.0f}), "
          f"A_eff = {target_Aeff:.2e}, mu = {b['mu']:.0f}.")
    print(f"To test if there is a better configuration, measure along "
          f"the iso-A_eff curve A_eff = {target_Aeff:.2e}:")
    print()
    print(f"  {'V':>6s}  {'G':>6s}  {'mu_expected':>12s}  "
          f"{'ch(170)':>10s}  {'rate_hint':>10s}")
    print("  " + "-" * 50)
    for V in [450, 475, 500, 525, 550]:
        G = target_Aeff / V ** PMT_ALPHA
        G = round(G)
        mu_expected = b["mu"] * (V ** PMT_ALPHA * G) / target_Aeff
        ch_170 = mu_expected / 3
        rate_hint = "low" if G < 300 else ("mid" if G < 700 else "high (dead risk)")
        print(f"  {V:>6d}  {G:>6d}  {mu_expected:>10.0f}  "
              f"{ch_170:>10.0f}  {rate_hint:>10s}")


# =========================================================================
# Main
# =========================================================================
def main():
    df = load()
    if df.empty:
        return

    df.to_csv(TAB_DIR / "best_for_compton.csv", index=False)
    print(f"\nFiltered table saved to {TAB_DIR / 'best_for_compton.csv'}")

    best_point(df)
    plot_R_vs_Aeff(df)
    plot_R_vs_mu(df)
    suggest_measurements(df)


if __name__ == "__main__":
    main()
