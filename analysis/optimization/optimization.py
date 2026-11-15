'''
optimization.py

Fit every significant peak in every .Spe of DATA_DIR, and save a CSV
with one row per (file, peak). The CSV is then consumed by full_analysis.py,
which filters the 511 keV peak and fits R(V).

Peak finding uses scipy.signal.find_peaks with LOW thresholds (so wide peaks
are also found) plus a WIDTH filter (to discard single-channel noise and
too-wide structures like Compton edges). The fit window is fit_half_fwhm
times the FWHM measured on the data, not a fixed number of channels.

Outputs
-------
- CSV with one row per (file, peak):  results/tables/optimization/all_fits.csv
- One figure per spectrum (data + candidates):   results/figures/optimization/fits/
- One figure per fitted peak:                    results/figures/optimization/fits/
- One figure per spectrum (overlay of all fits): results/figures/optimization/fits/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from pathlib import Path
import lab_compton.parameters as par

from lab_compton.maestro_reader import read_dir
from lab_compton.spectrum_fit import fit_peak, MODELS, ComptonStep


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "Optimization" / "Scintillator1" / "Spe"
OUT_CSV      = PROJECT_ROOT / "results" / "tables" / "optimization" / "all_fits.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures" / "optimization" / "fits"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL        = "gauss_exp"
MIN_CHANNEL  = 50          # ignore threshold noise below this channel
ZERO_CHANNEL = 0.0         

# Peak finding (scipy.signal.find_peaks)
PEAK_PROMINENCE = 100       # minimum prominence (counts)
PEAK_HEIGHT     = 80       # minimum peak height (counts)
PEAK_DISTANCE   = 30       # minimum channel distance between peaks
PEAK_WIDTH_MIN  = 8        # minimum FWHM (channels) — rejects single-channel noise
PEAK_WIDTH_MAX  = 130      # maximum FWHM (channels) — rejects Compton edges

# Fit window
FIT_HALF_FWHM = 2.5        # fit window = ±FIT_HALF_FWHM × FWHM_data
MIN_HALF      = 8          # minimum ±half-window (channels)

SHOW_PLOTS   = False
SAVE_PLOTS   = True
FIG_FORMAT   = "pdf"
SAVE_KW      = dict(bbox_inches="tight", dpi=200)

DO_PLOTS     = SHOW_PLOTS or SAVE_PLOTS


# =========================================================================
# Find all peaks (find_peaks + width filter + adaptive window)
# =========================================================================
def find_all_peaks(spectrum, *, min_channel=MIN_CHANNEL,
                   prominence=PEAK_PROMINENCE, height=PEAK_HEIGHT,
                   distance=PEAK_DISTANCE,
                   width_min=PEAK_WIDTH_MIN, width_max=PEAK_WIDTH_MAX,
                   fit_half_fwhm=FIT_HALF_FWHM, min_half=MIN_HALF):
    '''
    Find all peaks with a physically meaningful width.

    Returns a list of dicts:
        {'center': int, 'window': (lo, hi), 'fwhm_est': float}
    sorted by center.
    '''
    counts = np.asarray(spectrum.counts, dtype=float)

    peaks, props = find_peaks(
        counts,
        prominence=prominence,
        height=height,
        distance=distance,
        width=(width_min, width_max),
    )
    peaks = peaks[peaks >= min_channel]
    if len(peaks) == 0:
        return []

    widths = peak_widths(counts, peaks, rel_height=0.5)[0]

    result = []
    for p, w in zip(peaks, widths):
        half = max(int(round(fit_half_fwhm * w)), min_half)
        first, last = int(spectrum.channels[0]), int(spectrum.channels[-1])
        lo = max(first + 1, int(p) - half, min_channel)
        hi = min(last - 1, int(p) + half)
        if hi - lo < 12:
            continue
        result.append({
            "center":   int(p),
            "window":   (lo, hi),
            "fwhm_est": float(w),
        })
    return sorted(result, key=lambda x: x["center"])


# =========================================================================
# Fit one file
# =========================================================================
def process_file(name, s):
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    mask    = s.channels >= MIN_CHANNEL
    ch_plot = s.channels[mask]
    co_plot = s.counts[mask]

    # Find all significant peaks
    found_peaks = find_all_peaks(s)
    centers = [p["center"] for p in found_peaks]

    print(f"\nFound {len(found_peaks)} peaks (ch >= {MIN_CHANNEL}): {centers}")
    for p in found_peaks:
        print(f"  center={p['center']:5d}  window={p['window']}  "
              f"fwhm_est={p['fwhm_est']:.1f} ch")

    if not found_peaks:
        print("No peak found: skipping.")
        return [], {}

    # 1. Overview plot
    if DO_PLOTS:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.step(ch_plot, co_plot, where="mid", color="steelblue", label="data")
        if centers:
            ax.plot(centers, s.counts[centers], "rv", label="candidates",
                    markersize=9, zorder=5)
        ax.set_xlim(MIN_CHANNEL, s.channels[-1])
        ax.set_xlabel("channel"); ax.set_ylabel("counts")
        ax.set_title(f"{s.path.name} — live {s.live:.1f} s")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        if SAVE_PLOTS:
            fig.savefig(FIG_DIR / f"{s.path.stem}_spectrum.{FIG_FORMAT}", **SAVE_KW)
        if SHOW_PLOTS: plt.show()
        else:          plt.close(fig)

    # 2. Fit each peak
    rows, fits = [], {}
    for pk in found_peaks:
        ch = pk["center"]
        lo, hi = pk["window"]
        try:
            fit = fit_peak(s.channels, s.counts, window=(lo, hi), model=MODEL)
        except Exception as e:
            print(f"[ch {ch}] fit failed: {e}")
            continue

        fits[ch] = fit
        mu, mu_err      = fit.centroid()
        fwhm, fwhm_err  = fit.fwhm()
        n, n_lo, n_up   = fit.n_events_asym()
        R, R_err = fit.peak_resolution(zero_channel=ZERO_CHANNEL)
        if not (par.R_MIN_511 < R < par.R_MAX_511):
            print(f"[ch {ch}] R={R:.3f} out of [{par.R_MIN_511}, {par.R_MAX_511}], skip")
            continue
        if fit.reduced_deviance() > par.D_NDOF_MAX:
            print(f"[ch {ch}] D/ndof={fit.reduced_deviance():.2f} > {par.D_NDOF_MAX}, skip")
            continue

        rows.append({
            "file":     name,
            "channel":  int(ch),
            "valid":    fit.valid,
            "window":   f"{lo}-{hi}",
            "mu":       mu,        "mu_err":   mu_err,
            "fwhm":     fwhm,      "fwhm_err": fwhm_err,
            "R":        R,         "R_err":    R_err,
            "n_sig":    n,         "n_lo":     n_lo,  "n_up": n_up,
            "D_ndof":   f"{fit.deviance:.1f}/{fit.ndof} = {fit.reduced_deviance():.2f}",
            "warnings": "; ".join(fit.warnings()),
            "live": s.live,
            "real": s.real,
            "dead_pct": 100 * (1 - s.live / s.real),
        })
        
    # 3. Diagnostic plots
    if DO_PLOTS:
        def components_on_grid(fit, n_points=400):
            x_fine = np.linspace(fit.channels[0] - 0.5, fit.channels[-1] + 0.5, n_points)
            lo_e, hi_e = fit.channels[0] - 0.5, fit.channels[-1] + 0.5
            m = MODELS[fit.model]()
            v = fit.values
            peak = m.signal.pdf(x_fine, *[v[k] for k in m.signal.params])
            bkg  = m.background.pdf(x_fine, *[v[k] for k in m.background.params], lo_e, hi_e)
            total = peak + bkg
            step = None
            if getattr(m, "step", False):
                step = ComptonStep.pdf(x_fine, v["h"], v["mu"], v["sigma"])
                total = total + step
            return {"x": x_fine, "peak": peak, "bkg": bkg, "step": step, "total": total}

        for ch, fit in fits.items():
            mu, mu_err    = fit.centroid()
            fwhm, _       = fit.fwhm()
            R, _          = fit.peak_resolution(zero_channel=ZERO_CHANNEL)
            c = components_on_grid(fit)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax1.errorbar(fit.channels, fit.counts,
                         yerr=np.sqrt(np.maximum(fit.counts, 1)),
                         fmt="o", ms=4, capsize=2, color="steelblue", label="data")
            ax1.plot(c["x"], c["peak"], "--", color="seagreen", lw=1.6, label="peak")
            ax1.plot(c["x"], c["bkg"], ":", color="firebrick", lw=1.6, label="background")
            if c["step"] is not None:
                ax1.plot(c["x"], c["step"], "-.", color="gray", lw=1.4, label="step")
            ax1.plot(c["x"], c["total"], "-", color="darkorange", lw=2.0, label="total fit")
            ax1.set_ylabel("counts")
            ax1.set_title(
                f"peak at ch≈{ch} | mu = {mu:.2f} ± {mu_err:.2f} ch | "
                f"FWHM = {fwhm:.2f} ch | R = {100*R:.2f}% | "
                f"D/ndof = {fit.reduced_deviance():.2f}"
            )
            ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
            ax2.axhline(0, color="k", lw=0.5)
            ax2.bar(fit.channels, fit.residuals(), width=0.9, color="steelblue")
            ax2.set_xlabel("channel"); ax2.set_ylabel("(obs−exp)/√exp")
            ax2.set_ylim(-4, 4); ax2.grid(alpha=0.3)
            fig.tight_layout()
            if SAVE_PLOTS:
                fig.savefig(FIG_DIR / f"{s.path.stem}_peak_ch{ch}.{FIG_FORMAT}", **SAVE_KW)
            if SHOW_PLOTS: plt.show()
            else:          plt.close(fig)

        # Overlay
        if fits:
            lo_plot = max(MIN_CHANNEL, min(p["window"][0] for p in found_peaks) - 30)
            hi_plot = min(len(s.counts) - 1, max(p["window"][1] for p in found_peaks) + 30)

            fig, (ax_main, ax_res) = plt.subplots(
                2, 1, figsize=(11, 7), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_main.step(ch_plot, co_plot, where="mid",
                         color="steelblue", alpha=0.7, label="data")
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(fits), 1)))
            for (ch, fit), color in zip(fits.items(), colors):
                c = components_on_grid(fit)
                mu, _ = fit.centroid()
                ax_main.plot(c["x"], c["total"], "-", color=color, lw=1.8,
                             label=f"total ch{ch} (mu={mu:.1f})")
                ax_res.scatter(fit.channels, fit.residuals(),
                               s=18, color=color, label=f"ch{ch}", zorder=3)
            ax_main.set_xlim(lo_plot, hi_plot)
            ax_main.set_ylabel("counts")
            ax_main.set_title(f"{s.path.name} — all fits overlaid")
            ax_main.legend(fontsize=8, ncol=2); ax_main.grid(alpha=0.3)
            ax_res.axhline(0, color="k", lw=0.5)
            ax_res.set_xlabel("channel"); ax_res.set_ylabel("(obs − exp) / √exp")
            ax_res.set_ylim(-4, 4); ax_res.grid(alpha=0.3)
            fig.tight_layout()
            if SAVE_PLOTS:
                fig.savefig(FIG_DIR / f"{s.path.stem}_overlay.{FIG_FORMAT}", **SAVE_KW)
            if SHOW_PLOTS: plt.show()
            else:          plt.close(fig)

    return rows, fits


# =========================================================================
# Main
# =========================================================================
def main():
    spectra = read_dir(DATA_DIR, pattern="*.Spe")
    print(f"Found {len(spectra)} files in {DATA_DIR}")

    all_rows = []
    for name, s in spectra.items():
        rows, _ = process_file(name, s)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo fit succeeded.")
        return

    df = pd.DataFrame(all_rows).sort_values(["file", "channel"]).reset_index(drop=True)
    print("\n=== Summary ===")
    print(df[["file", "channel", "valid", "mu", "R", "R_err", "D_ndof"]].to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nTable saved to {OUT_CSV}")
    if SAVE_PLOTS:
        print(f"Figures saved to {FIG_DIR} (format: {FIG_FORMAT})")


if __name__ == "__main__":
    main()
