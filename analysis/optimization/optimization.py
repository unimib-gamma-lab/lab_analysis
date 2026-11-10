'''
optimization_all.py

Versione "in loop" del tuo inspect script: stessa struttura,
ma cicla su tutti i file .Spe della cartella e taglia i primi
MIN_CHANNEL canali dal plot.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

from lab_compton.maestro_reader import read_dir
from lab_compton.spectrum_fit import fit_peak, MODELS, ComptonStep

# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "Optimization" / "Scintillator1" / "Spe"

MODEL        = "gauss_lin"
MIN_CHANNEL  = 10
ZERO_CHANNEL = 0
SHOW_PLOTS   = True

# =========================================================================
# Funzione che elabora UN file
# =========================================================================
def process_file(name, s):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    
    # --- Maschera per escludere i primi MIN_CHANNEL canali ---
    mask = s.channels >= MIN_CHANNEL
    ch_plot = s.channels[mask]
    co_plot = s.counts[mask]

    # ---------------------------------------------------------------------
    # 2) Find peaks (filtra il rumore a bassa energia)
    # ---------------------------------------------------------------------
    peaks, props = find_peaks(s.counts, prominence=200, distance=10, height=200)
    peaks = peaks[peaks >= MIN_CHANNEL]

    print(f"\nCandidate peaks (ch >= {MIN_CHANNEL}): {peaks}")
    if len(peaks) == 0:
        print("Nessun picco sopra MIN_CHANNEL: salto.")
        return [], {}

    if SHOW_PLOTS:
        fig, ax = plt.subplots(figsize=(11, 4))
        
        # Usa gli array mascherati
        ax.step(ch_plot, co_plot, where="mid", color="steelblue", label="data")
        ax.plot(peaks, s.counts[peaks], "rv", label="candidates",
                markersize=9, zorder=5)
        
        # Imposta l'asse x per partire da MIN_CHANNEL
        ax.set_xlim(MIN_CHANNEL, s.channels[-1])
        
        ax.set_xlabel("channel"); ax.set_ylabel("counts")
        ax.set_title(f"{s.path.name} — live {s.live:.1f} s")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    # ---------------------------------------------------------------------
    # 3) Fit each peak
    # ---------------------------------------------------------------------
    rows, fits = [], {}
    for ch in peaks:
        fit_half = max(14, int(0.15 * ch))
        lo = max(int(s.channels[0]) + 1, ch - fit_half)
        hi = min(int(s.channels[-1]) - 1, ch + fit_half)
        try:
            fit = fit_peak(s.channels, s.counts, window=(lo, hi), model=MODEL)
        except Exception as e:
            print(f"[ch {ch}] fit failed: {e}")
            continue

        fits[int(ch)] = fit
        mu, mu_err      = fit.centroid()
        fwhm, fwhm_err  = fit.fwhm()
        n, n_lo, n_up   = fit.n_events_asym()
        R, R_err        = fit.peak_resolution(zero_channel=ZERO_CHANNEL)

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
        })

    if SHOW_PLOTS:
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

        # -----------------------------------------------------------------
        # 4) Plot singoli fit
        # -----------------------------------------------------------------
        for ch, fit in fits.items():
            mu, _         = fit.centroid()
            fwhm, _       = fit.fwhm()
            R, _          = fit.peak_resolution(zero_channel=ZERO_CHANNEL)
            n, n_lo, n_up = fit.n_events_asym()
            c = components_on_grid(fit)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]})

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
                f"D/ndof = {fit.reduced_deviance():.2f}")
            ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

            ax2.axhline(0, color="k", lw=0.5)
            ax2.bar(fit.channels, fit.residuals(), width=0.9, color="steelblue")
            ax2.set_xlabel("channel"); ax2.set_ylabel("(obs−exp)/√exp")
            ax2.set_ylim(-4, 4); ax2.grid(alpha=0.3)
            plt.tight_layout(); plt.show()

        # -----------------------------------------------------------------
        # 5) Plot totale (overlay)
        # -----------------------------------------------------------------
        lo_plot = max(MIN_CHANNEL, int(peaks.min()) - 30) # <-- usa MIN_CHANNEL come limite
        hi_plot = min(len(s.counts) - 1, int(peaks.max()) + 30)
        
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
            
        # Usa gli array mascherati
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
        ax_res.set_xlabel("channel"); ax_res.set_ylabel("(obs−exp)/√exp")
        ax_res.set_ylim(-4, 4); ax_res.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    return rows, fits

# =========================================================================
# MAIN
# =========================================================================
def main():
    spectra = read_dir(DATA_DIR, pattern="*.Spe")
    print(f"Trovati {len(spectra)} file in {DATA_DIR}")

    all_rows = []
    for name, s in spectra.items():
        rows, _ = process_file(name, s)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNessun fit riuscito.")
        return

    df = pd.DataFrame(all_rows).sort_values(["file", "channel"]).reset_index(drop=True)
    print("\n=== Summary finale ===")
    print(df[["file", "channel", "valid", "mu", "R", "R_err", "D_ndof"]].to_string(index=False))

    out = PROJECT_ROOT / "results" / "tables" / "optimization" / "all_fits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nTabella salvata in {out}")

if __name__ == "__main__":
    main()
