'''
systematics.py

Systematic uncertainties for the HV/gain/shaping scan:

    - window systematics: re-fit the scan with different fit_half_fwhm
      values (the half-width of the fit window, in units of the measured
      FWHM) and compute the per-file spread of R.
    - model systematics: re-fit the scan with different peak models
      (gauss_lin vs cb_lin) and compute the per-file spread of R.

The two effects are treated as independent and combined in quadrature.
The spread is reported as the semi-dispersion (max - min) / 2 per file,
which is a conservative estimate of the systematic uncertainty (it
covers the whole range of plausible values, not just one standard
deviation).
'''

import numpy as np
import pandas as pd


# =========================================================================
# Per-file spread helper
# =========================================================================

def _per_file_spread(df_all, *, key_cols=("name",)):
    '''
    Given a DataFrame of fits with different 'variants' (identified by
    an extra column added by the caller, e.g. 'fit_half_fwhm' or 'model'),
    compute for each file:
        R_mean   = mean of R over the variants
        R_spread = (max(R) - min(R)) / 2
    Only valid fits are used. Files with < 2 valid variants get NaN spread.
    '''
    ok = df_all[df_all["valid"]].dropna(subset=["R", "R_err"])
    if ok.empty:
        return pd.DataFrame(columns=list(key_cols) + ["V", "G", "S",
                                                     "R_mean", "R_spread",
                                                     "n_variants"])

    rows = []
    for key, grp in ok.groupby(list(key_cols)):
        r = grp["R"].values
        rows.append({
            **dict(zip(key_cols, key if isinstance(key, tuple) else (key,))),
            "V":         float(grp["V"].iloc[0]),
            "G":         float(grp["G"].iloc[0]),
            "S":         float(grp["S"].iloc[0]),
            "R_mean":    float(np.mean(r)),
            "R_spread":  float((r.max() - r.min()) / 2.0) if len(r) > 1 else np.nan,
            "n_variants": int(len(r)),
        })
    return pd.DataFrame(rows)


# =========================================================================
# Window systematics
# =========================================================================

def window_systematics(folder, *,
                       variants=(1.5, 2.0, 2.5, 3.0, 4.0),
                       verbose=True, **load_kw):
    '''
    Re-fit the scan with several fit_half_fwhm values.

    Parameters
    ----------
    folder : str | Path
        Folder of .Spe files.
    variants : iterable of float
        Values of fit_half_fwhm to try. The default 1.5-4.0 brackets the
        typical optimal window (2.5-3 FWHM) with a safety margin.
    verbose : bool
        Print progress.
    **load_kw :
        Other keyword arguments for load_scan (e.g. min_channel=30,
        gain=..., shaping=..., zero_channel=...).

    Returns
    -------
    df_all : pd.DataFrame
        Concatenated fits, one row per (file, variant).
    df_spread : pd.DataFrame
        One row per file with R_mean, R_spread (semi-dispersion over
        variants), n_variants.
    '''
    from lab_compton.resolution_scan import load_scan

    frames = []
    for fh in variants:
        if verbose:
            print(f"[window] fit_half_fwhm = {fh}")
        df = load_scan(folder, fit_half_fwhm=fh, verbose=False, **load_kw)
        df["fit_half_fwhm"] = fh
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_spread = _per_file_spread(df_all)
    return df_all, df_spread


# =========================================================================
# Model systematics
# =========================================================================

def model_systematics(folder, *,
                      models=None, exclude=(),
                      verbose=True, **load_kw):
    '''
    Re-fit the scan with several peak models.

    Parameters
    ----------
    folder : str | Path
        Folder of .Spe files.
    models : iterable of str or None
        Keys of MODELS in spectrum_fit. If None (default), use ALL models
        registered in MODELS, minus those in `exclude`. If you add a new
        model to MODELS, it is automatically included here on the next run.
    exclude : iterable of str
        Model keys to skip even if present in MODELS. Useful to drop
        models that are known to be inappropriate for the current setup
        (e.g. exclude=("gauss_lin_step",) if you don't believe in the
        Compton step).
    verbose : bool
        Print progress.
    **load_kw :
        Other keyword arguments for load_scan (min_channel, gain, shaping, ...).

    Returns
    -------
    df_all, df_spread
        Same structure as window_systematics.
    '''
    from lab_compton.resolution_scan import load_scan
    from lab_compton.spectrum_fit import MODELS

    # Default: all registered models, minus the excluded ones
    if models is None:
        models = list(MODELS.keys())
    models = [m for m in models if m not in exclude]
    if not models:
        raise ValueError("no model left after excluding %s" % (exclude,))

    frames = []
    for model in models:
        if verbose:
            print(f"[model] model = {model}")
        df = load_scan(folder, model=model, verbose=False, **load_kw)
        df["model"] = model
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_spread = _per_file_spread(df_all)
    return df_all, df_spread


# =========================================================================
# Print a summary
# =========================================================================

def print_systematics(spread, label="systematic"):
    '''
    Print a compact summary of the per-file spread.
    Uses only rows where R_spread is finite and n_variants >= 2.
    '''
    if spread.empty:
        print(f"\n[{label}] no rows (empty or all invalid).")
        return

    ok = spread.dropna(subset=["R_spread"])
    ok = ok[ok["n_variants"] >= 2]
    if ok.empty:
        print(f"\n[{label}] no file with >= 2 valid variants.")
        return

    med   = float(np.median(ok["R_spread"]))
    r_med = float(np.median(ok["R_mean"]))
    i_max = ok["R_spread"].idxmax()
    r_max = float(ok.loc[i_max, "R_spread"])
    name_max = ok.loc[i_max, "name"]

    print(f"\n=== {label} ===")
    print(f"  files with >= 2 valid variants: {len(ok)} / {len(spread)}")
    print(f"  median R_spread:  {100*med:.4f}%  "
          f"(relative: {100*med/r_med:.3f}%)")
    print(f"  max R_spread:     {100*r_max:.4f}%  "
          f"(file: {name_max})")

    # Quick table
    cols = ["name", "V", "G", "S", "R_mean", "R_spread", "n_variants"]
    cols = [c for c in cols if c in ok.columns]
    print(ok[cols].sort_values("R_spread", ascending=False).to_string(index=False))


# =========================================================================
# Combine the two systematics
# =========================================================================

def combine_systematics(spread_window, spread_model):
    '''
    Combine window and model systematics in quadrature, per file.

    Files missing in one of the two tables get NaN for the corresponding
    systematic (and the total uses only the available one).

    Returns
    -------
    pd.DataFrame with columns
        name, V, G, S, R_mean, syst_window, syst_model, syst_tot
    '''
    def _rename(df, suffix):
        if df.empty:
            return df.assign(**{f"R_spread_{suffix}": np.nan,
                                f"R_mean_{suffix}":   np.nan,
                                f"n_var_{suffix}":    0})
        return df.rename(columns={
            "R_spread":   f"R_spread_{suffix}",
            "R_mean":     f"R_mean_{suffix}",
            "n_variants": f"n_var_{suffix}",
        })

    w = _rename(spread_window, "w")
    m = _rename(spread_model,  "m")

    merged = pd.merge(w, m, on="name", how="outer",
                      suffixes=("", "_m"))
    # After merge, columns V/G/S may be duplicated (V and V_m etc.)
    def _first(df, base):
        cols = [c for c in df.columns if c == base or c.startswith(base + "_")]
        return df[cols].bfill(axis=1).iloc[:, 0] if cols else np.nan

    out = pd.DataFrame({
        "name":        merged["name"],
        "V":           _first(merged, "V"),
        "G":           _first(merged, "G"),
        "S":           _first(merged, "S"),
        "R_mean":      merged[["R_mean_w", "R_mean_m"]].mean(axis=1),
        "syst_window": merged["R_spread_w"],
        "syst_model":  merged["R_spread_m"],
    })
    out["syst_tot"] = np.sqrt(out["syst_window"].fillna(0.0)**2
                              + out["syst_model"].fillna(0.0)**2)
    return out
