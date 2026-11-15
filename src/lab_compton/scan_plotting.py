'''
scan_plotting.py

Plotting and text-summary utilities for the HV/gain/shaping scan.
Pure presentation: nothing here affects the fit or the results.

matplotlib is imported lazily inside each plotting function, so this
module can be imported on a headless server without a display.
'''

import numpy as np


# =========================================================================
# Local helpers (avoid circular imports with resolution_scan.py)
# =========================================================================

def _pmt_curve(V, R_intr, A, B):
    '''R(V)^2 = R_intr^2 + (A/V)^2 + (B*V)^2. Same as in resolution_scan.'''
    return np.sqrt(R_intr**2 + (A / V)**2 + (B * V)**2)


# =========================================================================
# plot_scan: one R(V) panel
# =========================================================================

def plot_scan(df, res_par=None, res_pmt=None, *,
              title=None, ax=None, show_bands=True,
              ylabel="Resolution R = FWHM / E"):
    '''
    Plot R(V) with both fits overlaid. Returns the matplotlib Axes.
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    sub = df[df["valid"]].dropna(subset=["V", "R", "R_err"])
    ax.errorbar(sub["V"], sub["R"], yerr=sub["R_err"],
                fmt="o", capsize=3, color="steelblue", label="data", zorder=5)

    if len(sub) > 1:
        V_fit = np.linspace(sub["V"].min(), sub["V"].max(), 300)
    else:
        V_fit = None

    if res_par is not None and V_fit is not None:
        V0, R0, k = res_par["V0"], res_par["R0"], res_par["k"]
        chi2_ndof = res_par["chi2"] / res_par["ndof"] if res_par.get("ndof") else np.nan

        ax.plot(V_fit, R0 + k * (V_fit - V0) ** 2, "-", color="darkorange",
                label=f"parabola (χ²/ndof = {chi2_ndof:.2f})")

        if "V0_minos" in res_par and res_par["V0_minos"]["valid"]:
            m = res_par["V0_minos"]
            V0_lo, V0_up = V0 + m["lower"], V0 + m["upper"]
            v0_label = f"V0 = {V0:.0f} +{m['upper']:.0f}/-{abs(m['lower']):.0f} V"
        else:
            V0_lo, V0_up = V0 - res_par["V0_err"], V0 + res_par["V0_err"]
            v0_label = f"V0 = {V0:.0f} ± {res_par['V0_err']:.0f} V"

        ax.axvline(V0, color="darkorange", ls="--", alpha=0.5, label=v0_label)
        if show_bands and V0_lo < V0_up:
            ax.axvspan(V0_lo, V0_up, color="darkorange", alpha=0.12, zorder=0)
        ax.plot([V0], [R0], "*", color="darkorange", ms=14, zorder=6)

    if res_pmt is not None and res_pmt["valid"] and V_fit is not None:
        V_opt, R_opt = res_pmt["V_opt"], res_pmt["R_opt"]
        chi2_ndof = res_pmt["chi2"] / res_pmt["ndof"] if res_pmt.get("ndof") else np.nan

        ax.plot(V_fit, _pmt_curve(V_fit, *res_pmt["popt"]), "-", color="seagreen",
                label=f"PMT model (χ²/ndof = {chi2_ndof:.2f})")

        if "V_opt_asym" in res_pmt:
            V_lo = res_pmt["V_opt_asym"]["lower"]
            V_up = res_pmt["V_opt_asym"]["upper"]
            v_label = f"V_opt = {V_opt:.0f} +{V_up - V_opt:.0f}/-{V_opt - V_lo:.0f} V"
        else:
            V_lo = V_opt - res_pmt["V_opt_err"]
            V_up = V_opt + res_pmt["V_opt_err"]
            v_label = f"V_opt = {V_opt:.0f} ± {res_pmt['V_opt_err']:.0f} V"

        ax.axvline(V_opt, color="seagreen", ls=":", alpha=0.6, label=v_label)
        if show_bands and np.isfinite(V_lo) and np.isfinite(V_up) and V_lo < V_up:
            ax.axvspan(V_lo, V_up, color="seagreen", alpha=0.12, zorder=0)
        ax.plot([V_opt], [R_opt], "*", color="seagreen", ms=14, zorder=6)

    if len(sub) > 1:
        pad = 0.05 * (sub["V"].max() - sub["V"].min())
        ax.set_xlim(sub["V"].min() - pad, sub["V"].max() + pad)
    ax.set_xlabel("HV (V)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return ax


# =========================================================================
# print_summary: text summary of a single group
# =========================================================================

def print_summary(df, res_par=None, res_pmt=None):
    '''Print a compact summary of the scan and of both fits.'''
    sub = df[df["valid"]].dropna(subset=["V", "R"])
    print("\n=== HV scan summary ===")
    print(f"points used: {len(sub)}")
    for _, r in sub.iterrows():
        print(f"  V = {r['V']:7g}   R = {r['R']:.4f} ± {r['R_err']:.4f}"
              f"   mu = {r['mu']:.2f}")

    if res_par is not None:
        print("\n--- Parabolic fit ---")
        chi2, ndof = res_par.get("chi2"), res_par.get("ndof")
        if ndof:
            print(f"  χ²/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:.2f}")
        if "V0_minos" in res_par and res_par["V0_minos"]["valid"]:
            m = res_par["V0_minos"]
            print(f"  V0    = {res_par['V0']:.1f} +{m['upper']:.1f}/-{abs(m['lower']):.1f} V  [MINOS]")
        else:
            print(f"  V0    = {res_par['V0']:.1f} ± {res_par['V0_err']:.1f} V  [HESSE]")
        print(f"  R(V0) = {res_par['R0']:.4f} ± {res_par['R0_err']:.4f}")
        print(f"  k     = {res_par['k']:.2e} ± {res_par['k_err']:.2e}")

    if res_pmt is not None:
        print("\n--- PMT model fit ---")
        chi2, ndof = res_pmt.get("chi2"), res_pmt.get("ndof")
        if ndof:
            print(f"  χ²/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:.2f}")
        print(f"  R_intr = {res_pmt['R_intr']:.4f} ± {res_pmt['R_intr_err']:.4f}")

        if "A_minos" in res_pmt and res_pmt["A_minos"]["valid"]:
            m = res_pmt["A_minos"]
            print(f"  A      = {res_pmt['A']:.2f} +{m['upper']:.2f}/-{abs(m['lower']):.2f}  [MINOS]")
        else:
            print(f"  A      = {res_pmt['A']:.2f} ± {res_pmt['A_err']:.2f}  [HESSE]")

        if "B_minos" in res_pmt and res_pmt["B_minos"]["valid"]:
            m = res_pmt["B_minos"]
            print(f"  B      = {res_pmt['B']:.2e} +{m['upper']:.2e}/-{abs(m['lower']):.2e}  [MINOS]")
        else:
            print(f"  B      = {res_pmt['B']:.2e} ± {res_pmt['B_err']:.2e}  [HESSE]")

        if np.isfinite(res_pmt["V_opt"]):
            if "V_opt_asym" in res_pmt:
                v = res_pmt["V_opt_asym"]
                print(f"  V_opt  = {res_pmt['V_opt']:.1f} "
                      f"+{v['upper'] - res_pmt['V_opt']:.1f}"
                      f"/-{res_pmt['V_opt'] - v['lower']:.1f} V  [MINOS]")
            else:
                print(f"  V_opt  = {res_pmt['V_opt']:.1f} ± {res_pmt['V_opt_err']:.1f} V  [HESSE]")
            if "R_opt_asym" in res_pmt:
                r = res_pmt["R_opt_asym"]
                print(f"  R(Vopt)= {res_pmt['R_opt']:.4f} "
                      f"+{r['upper'] - res_pmt['R_opt']:.4f}"
                      f"/-{res_pmt['R_opt'] - r['lower']:.4f}  [MINOS]")
            else:
                print(f"  R(Vopt)= {res_pmt['R_opt']:.4f} ± {res_pmt['R_opt_err']:.4f}  [HESSE]")
        else:
            print("  V_opt  = nan (fit did not give A, B > 0)")
        if not res_pmt["valid"]:
            print("  [warn] V_opt is outside the data range: extrapolation")


# =========================================================================
# plot_groups: grid of R(V) panels, one per (G, S)
# =========================================================================

def plot_groups(df, results, *, ncols=3, **plot_kw):
    '''One R(V) panel per (G, S). Returns the matplotlib Figure.'''
    import matplotlib.pyplot as plt

    keys = list(results)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 3.6 * nrows),
                             squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        r = results[key]
        title = ", ".join(f"{n} = {v:g}" for n, v in zip(("G", "S"), key))
        if int(r["df"]["valid"].sum()) == 0:
            ax.set_title(title + " (no valid fit)")
            continue
        if len(r["df"][r["df"]["valid"]]) < 2:
            sub = r["df"][r["df"]["valid"]]
            ax.errorbar(sub["V"], sub["R"], yerr=sub["R_err"],
                        fmt="o", capsize=3, color="steelblue")
            ax.set_title(title + " (1 point)")
            ax.grid(alpha=0.3)
            continue
        plot_scan(r["df"], r["parabola"], r["pmt"], ax=ax, title=title, **plot_kw)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# =========================================================================
# print_group_summary: one line per (G, S)
# =========================================================================

def print_group_summary(summary):
    '''Print one line per (G, S): optimum HV and resolution.'''
    def f(row, key, spec, default="   n/a"):
        v = row.get(key, np.nan)
        return default if v is None or not np.isfinite(v) else format(v, spec)

    print("\n=== (G, S) summary ===")
    print("   G      S   n |  PMT model: V_opt [V]        R_opt [%]            chi2/n | "
          "parabola: V0 [V]   R0 [%]   | lowest point R [%] (V)")
    for _, r in summary.iterrows():
        pmt = (f"{f(r, 'pmt_V_opt', '7.0f')} "
               f"+{f(r, 'pmt_V_opt_up', '.0f', '?')}/-{f(r, 'pmt_V_opt_lo', '.0f', '?').lstrip('-')}"
               f"   {100 * r.get('pmt_R_opt', np.nan):6.3f} ± {100 * r.get('pmt_R_opt_err', np.nan):.3f}"
               f"   {f(r, 'pmt_chi2_ndof', '5.2f')}"
               f"{'' if r.get('pmt_valid', True) else '  [outside range]'}")
        par = (f"{f(r, 'par_V0', '7.0f')}   {100 * r.get('par_R0', np.nan):6.3f}"
               f"{'' if r.get('par_in_range', True) else '  [outside range]'}")
        print(f"  {r['G']:>5g} {r['S']:>5g} {int(r['n_points']):3d} | {pmt} | {par} | "
              f"{100 * r.get('R_data_min', np.nan):.3f} ({f(r, 'V_data_min', '.0f')})")


# =========================================================================
# plot_optimum_map: 2D heatmap on the (G, S) grid
# =========================================================================

def plot_optimum_map(summary, *, method="pmt", value="R", best=None, ax=None):
    '''
    Map over the (G, S) grid of the optimum resolution (value="R", in %)
    or of the optimum HV (value="V"). The best setting is marked with a star.

    Parameters
    ----------
    best : pd.Series, optional
        Output of best_setting()["best"]. If None, computed lazily from
        best_setting(summary, method=method).
    '''
    import matplotlib.pyplot as plt

    cols = {"pmt":      {"R": "pmt_R_opt",   "V": "pmt_V_opt"},
            "parabola": {"R": "par_R0",      "V": "par_V0"},
            "data":     {"R": "R_data_min",  "V": "V_data_min"}}
    col = cols[method][value]
    scale = 100.0 if value == "R" else 1.0
    pv = summary.pivot(index="S", columns="G", values=col) * scale

    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * len(pv.columns) + 3,
                                      0.9 * len(pv.index) + 2.5))
    im = ax.imshow(pv.values, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pv.columns)), [f"{g:g}" for g in pv.columns])
    ax.set_yticks(range(len(pv.index)), [f"{s:g}" for s in pv.index])
    ax.set_xlabel("gain G")
    ax.set_ylabel("shaping S")

    vmin, vmax = np.nanmin(pv.values), np.nanmax(pv.values)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                light = vmax > vmin and (v - vmin) / (vmax - vmin) < 0.5
                ax.text(j, i, f"{v:.2f}" if value == "R" else f"{v:.0f}",
                        ha="center", va="center",
                        color="black" if light else "white", fontsize=9)

    if value == "R":
        if best is None:
            from lab_compton.resolution_scan import best_setting
            try:
                best = best_setting(summary, method=method)["best"]
            except ValueError:
                best = None
        if best is not None:
            ax.plot([list(pv.columns).index(best["G"]) + 0.3],
                    [list(pv.index).index(best["S"]) + 0.3],
                    "*", color="red", ms=14, markeredgecolor="white")

    ax.set_title(f"{'R at the optimum HV (%)' if value == 'R' else 'optimum HV (V)'}  [{method}]")
    plt.colorbar(im, ax=ax)
    return ax
    
# =========================================================================
# plot_R_vs_G: R as a function of the amplifier gain, one series per V
# =========================================================================

def plot_R_vs_G(df, *, title=None, ax=None, log_x=True, marker_size=7):
    '''
    Plot R vs G (gain) at fixed HV.

    One series per V (HV) setting:
        - if V has >= 2 G values, the points are connected by a line
          (you can see how R changes with G at fixed HV);
        - if V has only 1 G value, the point is an isolated marker.

    The G axis is log-scale when G spans more than a factor of 3, which
    is typically the case (G from 20 to 1000).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_scan or the filtered all_fits.csv. Must contain
        columns V, G, S, R, R_err, valid.
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    log_x : bool
        If True, log-scale on G (recommended).
    marker_size : float

    Returns
    -------
    matplotlib.axes.Axes
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sub = df[df["valid"]].dropna(subset=["G", "R", "R_err"]).copy()
    if sub.empty:
        ax.set_title("R vs G: no valid data")
        return ax

    # One color per V (sorted). plt.cm.viridis is the colormap object.
    Vs = sorted(sub["V"].unique())
    cmap = plt.cm.viridis
    V_to_color = {V: cmap(i / max(len(Vs) - 1, 1)) for i, V in enumerate(Vs)}

    S_set = sorted(sub["S"].unique())
    multi_S = len(S_set) > 1

    for V in Vs:
        g_v = sub[sub["V"] == V].sort_values("G")
        color = V_to_color[V]
        n_G = g_v["G"].nunique()
        label = f"V={V:g}" + (f", S={g_v['S'].iloc[0]:g}" if multi_S else "")

        if n_G >= 2:
            ax.errorbar(g_v["G"], g_v["R"] * 100, yerr=g_v["R_err"] * 100,
                        fmt="o-", capsize=3, color=color,
                        markersize=marker_size, label=label, zorder=5)
        else:
            ax.errorbar(g_v["G"], g_v["R"] * 100, yerr=g_v["R_err"] * 100,
                        fmt="o", capsize=2, color=color, alpha=0.6,
                        markersize=marker_size - 1, label=label, zorder=3)

    ax.set_xlabel("Gain G")
    ax.set_ylabel("R (%)")
    if log_x and sub["G"].max() / sub["G"].min() > 3:
        ax.set_xscale("log")
    ax.set_title(title or "R vs G (gain) — one series per HV setting")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    return ax


# =========================================================================
# plot_R_vs_G_grid: one panel per V that has >= 2 G values
# =========================================================================

def plot_R_vs_G_grid(df, *, ncols=3, log_x=True, marker_size=6):
    '''
    Grid of R vs G panels, one per V (HV) that has at least 2 G values.

    Useful to zoom in on the V settings where the G dependence is visible.
    V settings with only one G value are excluded (no curve to show).

    Returns
    -------
    matplotlib.figure.Figure, or None if no V has >= 2 G.
    '''
    import matplotlib.pyplot as plt

    sub = df[df["valid"]].dropna(subset=["G", "R", "R_err"]).copy()
    if sub.empty:
        print("[plot_R_vs_G_grid] no valid data")
        return None

    V_counts = sub.groupby("V")["G"].nunique()
    V_curves = sorted(V_counts[V_counts >= 2].index.tolist())
    if not V_curves:
        print("[plot_R_vs_G_grid] no V with >= 2 G values")
        return None

    ncols = min(ncols, len(V_curves))
    nrows = int(np.ceil(len(V_curves) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, V in zip(axes, V_curves):
        g_v = sub[sub["V"] == V].sort_values("G")
        ax.errorbar(g_v["G"], g_v["R"] * 100, yerr=g_v["R_err"] * 100,
                    fmt="o-", capsize=3, color="steelblue",
                    markersize=marker_size)
        ax.set_xlabel("Gain G")
        ax.set_ylabel("R (%)")
        ax.set_title(f"V = {V:g} V  (n_G = {len(g_v)})")
        if log_x and g_v["G"].max() / g_v["G"].min() > 3:
            ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")

    for ax in axes[len(V_curves):]:
        ax.set_visible(False)

    fig.suptitle("R vs G — one panel per HV setting (only V with >= 2 G)",
                 fontsize=12)
    fig.tight_layout()
    return fig


# =========================================================================
# print_summary_R_vs_G: text summary of R vs G
# =========================================================================

def print_summary_R_vs_G(df):
    '''
    Text summary: for each V that has >= 2 G values, print:
        - min and max R across G
        - the relative spread (R_max - R_min) / R_min
        - which G gives the minimum, and the min with its error

    Useful to answer "does the gain really matter at this HV?".
    '''
    sub = df[df["valid"]].dropna(subset=["G", "R", "R_err"]).copy()
    if sub.empty:
        print("\n[print_summary_R_vs_G] no valid data")
        return

    print("\n=== R vs G summary (per HV setting, only V with >= 2 G) ===")
    print(f"  {'V':>6s}  {'n_G':>4s}  {'R_min (%)':>10s}  {'R_max (%)':>10s}  "
          f"{'spread rel':>11s}  {'G@min':>7s}  {'R@min ± err':>18s}")
    print("  " + "-" * 82)

    for V, g in sub.groupby("V"):
        n_G = g["G"].nunique()
        if n_G < 2:
            continue
        i_min = g["R"].idxmin()
        r_min = g.loc[i_min, "R"]
        r_err = g.loc[i_min, "R_err"]
        r_max = g["R"].max()
        spread_rel = 100.0 * (r_max - r_min) / r_min if r_min > 0 else np.nan
        print(f"  {V:>6g}  {n_G:>4d}  {100*r_min:>10.3f}  {100*r_max:>10.3f}  "
              f"{spread_rel:>10.2f}%  {g.loc[i_min, 'G']:>7g}  "
              f"{100*r_min:>8.3f} ± {100*r_err:>4.3f}")
