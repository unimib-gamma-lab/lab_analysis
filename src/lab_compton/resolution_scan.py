'''
resolution_scan.py

Fit R(V) for each (G, S) combination of a scan.

Workflow
--------
    df = load_scan("data/", mu_ref=None, min_channel=30)
    # optional: df = apply_calibration(df, calib)   # from calibration.py
    results, summary = fit_groups(df)
    best = best_setting(summary, method="pmt")
    # plotting lives in scan_plotting.py, re-exported here for convenience

Resolution
----------
R = FWHM / (mu - zero_channel), adimensional. With zero_channel = 0 the
result is correct only if the ADC zero is at channel 0. Pass the value
from calibration once known. For the fit R(V) the absolute scale is
irrelevant, so zero_channel = 0 is fine for finding V_opt.

The model fit on R(V) is
    R(V)^2 = R_intr^2 + (A / V)^2 + (B * V)^2
R_intr is the intrinsic resolution of the crystal (asymptotic), A/V
the electronic noise (decreases with HV), B*V the multiplication
noise / saturation (increases with HV).
The optimum is at V_opt = sqrt(A/B).
'''

import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

from lab_compton.config import parse_filename
from lab_compton.maestro_reader import read_dir
from lab_compton.spectrum_fit import _find_peak_window, fit_peak


_SCAN_COLUMNS = ["name", "V", "G", "S", "valid", "error", "win_lo", "win_hi",
                 "mu", "mu_err", "fwhm", "fwhm_err", "R", "R_err",
                 "zero_channel",
                 "n_sig", "n_lo", "n_up", "deviance", "ndof", "warnings"]


# =========================================================================
# Load & fit
# =========================================================================

def load_scan(folder, mu_ref=None, *, gain=None, shaping=None,
              model="gauss_exp",
              zero_channel=0.0, zero_channel_err=0.0,
              search_half=None, fit_half_fwhm=2.5, window_kw=None,
              min_channel=0,
              parse=parse_filename, recursive=False, verbose=True):
    '''
    Load every .Spe in `folder`, fit the main peak, return a DataFrame
    sorted by (G, S, V). R is computed as FWHM / (mu - zero_channel).

    Parameters
    ----------
    folder : str | Path
        Folder of .Spe files.
    mu_ref : None | float | callable
        Where to look for the peak. None: most prominent peak of the
        whole spectrum. A number: search around that channel. A callable
        (V, G, S) -> channel.
    gain, shaping : optional
        If given, keep only the spectra with that gain / shaping.
    model : str
        Key of MODELS in spectrum_fit. Default "gauss_lin".
    zero_channel, zero_channel_err : float
        Channel of E = 0 and its error. Default 0.0 (no correction).
        For finding V_opt the absolute scale is irrelevant, but if you
        want absolute R (e.g. for a report) pass the calibrated value.
    search_half, fit_half_fwhm : see _find_peak_window.
    window_kw : dict, optional
        Extra kwargs for _find_peak_window.
    min_channel : int
        Ignore peaks below this channel (threshold noise).
    parse, recursive : see read_dir.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    pd.DataFrame
    '''
    window_kw = dict(window_kw or {})
    raw = read_dir(folder, pattern="*.Spe", recursive=recursive)

    rows = []
    for name, s in raw.items():
        key = parse(name)
        if key is None:
            if verbose:
                print(f"[skip] unparseable name: {name}")
            continue
        V, G, S = key
        if (gain is not None and G != gain) or (shaping is not None and S != shaping):
            continue

        row = {"name": name, "V": V, "G": G, "S": S, "valid": False}
        rows.append(row)
        try:
            ref = mu_ref(V, G, S) if callable(mu_ref) else mu_ref
            w, found, _ = _find_peak_window(
                s, ref, search_half=search_half,
                fit_half_fwhm=fit_half_fwhm,
                min_channel=min_channel, **window_kw)
            if not found:
                row["error"] = "peak not found"
                continue

            fit = fit_peak(s.channels, s.counts, window=w, model=model)

            mu, mu_err      = fit.centroid()
            fwhm, fwhm_err  = fit.fwhm()
            n, n_lo, n_up   = fit.n_events_asym()
            R, R_err        = fit.peak_resolution(zero_channel, zero_channel_err)

            row.update({
                "win_lo": w[0], "win_hi": w[1],
                "mu": mu, "mu_err": mu_err,
                "fwhm": fwhm, "fwhm_err": fwhm_err,
                "R": R, "R_err": R_err,
                "zero_channel": zero_channel,
                "n_sig": n, "n_lo": n_lo, "n_up": n_up,
                "deviance": fit.deviance, "ndof": fit.ndof,
                "valid": bool(fit.valid),
                "warnings": "; ".join(fit.warnings()),
            })
        except ValueError as e:
            row["error"] = str(e)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"no spectra"
            f"{'' if gain is None else f' with G={gain}'}"
            f"{'' if shaping is None else f' with S={shaping}'} in '{folder}'"
        )
    df = df.reindex(columns=_SCAN_COLUMNS)
    df["valid"] = df["valid"].fillna(False).astype(bool)
    df = df.sort_values(["G", "S", "V"]).reset_index(drop=True)

    if verbose:
        print(f"[ok] {int(df['valid'].sum())}/{len(df)} valid fits")
        for (g, sh), sub in df.groupby(["G", "S"]):
            print(f"     G={g}, S={sh}: {int(sub['valid'].sum())}/{len(sub)} valid, "
                  f"V={sub['V'].min()}-{sub['V'].max()}")
    return df


# =========================================================================
# Fit R(V): parabola
# =========================================================================

def fit_parabola_R_V(df, *, V_guess=None, minos=True):
    '''
    Fit R(V) = R0 + k * (V - V0)^2.

    Returns a dict with R0, V0, k, their errors, chi2/ndof, and (if
    MINOS is available) V0_minos.
    '''
    sub = df[df["valid"]].dropna(subset=["V", "R", "R_err"])
    if len(sub) < 3:
        raise ValueError(f"need at least 3 valid points, got {len(sub)}")

    V     = sub["V"].values.astype(float)
    R     = sub["R"].values.astype(float)
    R_err = sub["R_err"].values.astype(float)

    def parabola(v, R0, V0, k):
        return R0 + k * (v - V0) ** 2

    if V_guess is None:
        V_guess = V[np.argmin(R)]

    cost = LeastSquares(V, R, R_err, parabola)
    m = Minuit(cost, R0=R.min(), V0=V_guess, k=1e-6)
    m.limits["R0"] = (0, None)
    m.limits["k"]  = (0, None)
    span = V.max() - V.min()
    m.limits["V0"] = (V.min() - 2 * span, V.max() + 2 * span)
    m.strategy = 2
    m.migrad()
    m.hesse()

    if not m.valid:
        raise RuntimeError(f"parabola fit did not converge: {m.fmin}")
    if minos:
        m.minos("V0")

    out = {
        "R0": float(m.values["R0"]), "R0_err": float(m.errors["R0"]),
        "V0": float(m.values["V0"]), "V0_err": float(m.errors["V0"]),
        "k":  float(m.values["k"]),  "k_err":  float(m.errors["k"]),
        "popt": np.array([m.values["R0"], m.values["V0"], m.values["k"]]),
        "pcov": np.array(m.covariance),
        "V": V, "R": R, "R_err": R_err,
        "model": "parabola",
        "chi2": float(m.fval),
        "ndof": len(V) - 3,
    }
    if minos and "V0" in m.merrors and m.merrors["V0"].is_valid:
        me = m.merrors["V0"]
        out["V0_minos"] = {
            "lower": float(me.lower),
            "upper": float(me.upper),
            "valid": True,
        }
    return out


# =========================================================================
# Fit R(V): PMT model (three parametrisations)
# =========================================================================

def _pmt_model(v, R_intr, A, B):
    '''R(V)^2 = R_intr^2 + (A/V)^2 + (B*V)^2'''
    return np.sqrt(R_intr**2 + (A / v)**2 + (B * v)**2)


def _pmt_model_vopt(v, R_intr, V_opt, B):
    '''Same model, with A = B * V_opt^2 (V_opt is a fit parameter).'''
    return np.sqrt(R_intr**2 + (B * V_opt**2 / v)**2 + (B * v)**2)


def _pmt_model_ropt(v, R_opt, V_opt, r):
    '''Same model, parametrised by R_opt, V_opt and r = R_intr / R_opt in (0, 1).'''
    B = R_opt * np.sqrt((1.0 - r**2) / 2.0) / V_opt
    return np.sqrt((r * R_opt)**2 + (B * V_opt**2 / v)**2 + (B * v)**2)


def _minos_dict(m, name):
    '''Extract a single MINOS result from a Minuit object.'''
    if name not in m.merrors:
        return {"lower": np.nan, "upper": np.nan, "valid": False}
    me = m.merrors[name]
    return {
        "lower": float(me.lower),
        "upper": float(me.upper),
        "valid": bool(me.is_valid),
    }


def _ropt_minos(V, R, R_err, R_opt, V_opt, R_intr):
    '''MINOS on R_opt, in the parametrisation (R_opt, V_opt, r = R_intr / R_opt).'''
    r0 = float(np.clip(R_intr / R_opt, 0.02, 0.98))
    m2 = Minuit(LeastSquares(V, R, R_err, _pmt_model_ropt),
                R_opt=R_opt, V_opt=V_opt, r=r0)
    m2.limits["R_opt"] = (0, None)
    m2.limits["V_opt"] = (V.min() / 2, V.max() * 2)
    m2.limits["r"]     = (0, 1)
    m2.strategy = 2
    m2.migrad()
    m2.hesse()
    if not m2.valid:
        return {"lower": np.nan, "upper": np.nan, "valid": False}
    m2.minos("R_opt")
    out = _minos_dict(m2, "R_opt")
    out["valid"] = out["valid"] and out["lower"] < 0 < out["upper"]
    return out


def fit_pmt_R_V(df, *, p0=None, minos=True):
    '''
    Fit R(V) with the PMT model R(V)^2 = R_intr^2 + (A/V)^2 + (B*V)^2.

    Fits (R_intr, V_opt, B) with A = B * V_opt^2, so V_opt is a fit
    parameter and MINOS gives its profile-likelihood interval directly.
    R_opt gets its own MINOS in the parametrisation (R_opt, V_opt, r).

    Returns a dict with parameters, errors, chi2/ndof, and MINOS intervals.
    '''
    sub = df[df["valid"]].dropna(subset=["V", "R", "R_err"])
    if len(sub) < 4:
        raise ValueError(f"need at least 4 valid points, got {len(sub)}")

    V     = sub["V"].values.astype(float)
    R     = sub["R"].values.astype(float)
    R_err = sub["R_err"].values.astype(float)

    if p0 is None:
        i_min = int(np.argmin(R))
        V_opt0 = V[i_min]
        R_intr0 = 0.8 * R[i_min]
        B0 = np.sqrt(max((R[i_min]**2 - R_intr0**2) / 2, 1e-12)) / V_opt0
    else:
        R_intr0, A0, B0 = p0
        V_opt0 = np.sqrt(A0 / B0)

    cost = LeastSquares(V, R, R_err, _pmt_model_vopt)
    m = Minuit(cost, R_intr=R_intr0, V_opt=V_opt0, B=B0)
    m.limits["R_intr"] = (0, None)
    m.limits["V_opt"]  = (V.min() / 2, V.max() * 2)
    m.limits["B"]      = (0, None)
    m.strategy = 2
    m.migrad()
    m.hesse()
    if not m.valid:
        raise RuntimeError(f"PMT fit did not converge: {m.fmin}")
    if minos:
        m.minos("V_opt", "B")

    R_intr, V_opt, B = (float(m.values[k]) for k in ("R_intr", "V_opt", "B"))
    A = B * V_opt**2

    # Covariance of (R_intr, A, B) from (R_intr, V_opt, B)
    J = np.array([[1.0, 0.0,            0.0],
                  [0.0, 2 * B * V_opt,  V_opt**2],
                  [0.0, 0.0,            1.0]])
    pcov = J @ np.array(m.covariance) @ J.T
    popt = np.array([R_intr, A, B])
    perr = np.sqrt(np.diag(pcov))

    R_opt = float(_pmt_model(V_opt, *popt))
    if R_opt > 0:
        Jr = np.array([R_intr / R_opt,
                       A / (V_opt**2 * R_opt),
                       B * V_opt**2 / R_opt])
        R_opt_err = float(np.sqrt(max(Jr @ pcov @ Jr, 0.0)))
    else:
        R_opt_err = np.nan

    valid = bool(V.min() <= V_opt <= V.max() and A > 0 and B > 0)

    out = {
        "R_intr": R_intr, "R_intr_err": perr[0],
        "A": A, "A_err": perr[1],
        "B": B, "B_err": perr[2],
        "V_opt": V_opt, "V_opt_err": float(m.errors["V_opt"]),
        "R_opt": R_opt, "R_opt_err": R_opt_err,
        "popt": popt, "pcov": pcov,
        "V": V, "R": R, "R_err": R_err,
        "valid": valid,
        "model": "pmt",
        "chi2": float(m.fval),
        "ndof": len(V) - 3,
    }

    if minos:
        out["V_opt_minos"] = _minos_dict(m, "V_opt")
        out["B_minos"]     = _minos_dict(m, "B")
        if out["V_opt_minos"]["valid"]:
            out["V_opt_asym"] = {"lower": V_opt + out["V_opt_minos"]["lower"],
                                 "upper": V_opt + out["V_opt_minos"]["upper"]}
        out["R_opt_minos"] = _ropt_minos(V, R, R_err, R_opt, V_opt, R_intr)
        if out["R_opt_minos"]["valid"]:
            out["R_opt_asym"] = {"lower": R_opt + out["R_opt_minos"]["lower"],
                                 "upper": R_opt + out["R_opt_minos"]["upper"]}
    return out


# =========================================================================
# Fit all groups
# =========================================================================

def fit_groups(df, *, by=("G", "S"), minos=True):
    '''
    Fit R(V) separately for every (G, S) combination.

    Returns (results, summary).
    '''
    results, rows = {}, []
    for key, sub in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        sub = sub.sort_values("V").reset_index(drop=True)
        ok = sub[sub["valid"]].dropna(subset=["V", "R", "R_err"])

        res = {"df": sub, "parabola": None, "pmt": None, "errors": {}}
        for name, fn in (("parabola", fit_parabola_R_V), ("pmt", fit_pmt_R_V)):
            try:
                res[name] = fn(ok, minos=minos)
            except (ValueError, RuntimeError) as e:
                res["errors"][name] = str(e)
        results[key] = res

        row = dict(zip(by, key))
        row["n_points"] = len(ok)
        if len(ok):
            i = int(ok["R"].values.argmin())
            row.update(V_data_min=float(ok["V"].iloc[i]),
                       R_data_min=float(ok["R"].iloc[i]),
                       R_data_min_err=float(ok["R_err"].iloc[i]))

        p = res["parabola"]
        if p is not None:
            m = p.get("V0_minos")
            lo, up = (m["lower"], m["upper"]) if m and m["valid"] else (-p["V0_err"], p["V0_err"])
            row.update(par_V0=p["V0"], par_V0_lo=lo, par_V0_up=up,
                       par_R0=p["R0"], par_R0_err=p["R0_err"],
                       par_chi2_ndof=p["chi2"] / p["ndof"] if p["ndof"] > 0 else np.nan,
                       par_in_range=bool(ok["V"].min() <= p["V0"] <= ok["V"].max()))

        q = res["pmt"]
        if q is not None:
            a = q.get("V_opt_asym")
            lo, up = ((a["lower"] - q["V_opt"], a["upper"] - q["V_opt"]) if a
                      else (-q["V_opt_err"], q["V_opt_err"]))
            ra = q.get("R_opt_asym")
            rlo, rup = ((ra["lower"] - q["R_opt"], ra["upper"] - q["R_opt"]) if ra
                        else (-q["R_opt_err"], q["R_opt_err"]))
            row.update(pmt_V_opt=q["V_opt"], pmt_V_opt_lo=lo, pmt_V_opt_up=up,
                       pmt_R_opt=q["R_opt"], pmt_R_opt_err=q["R_opt_err"],
                       pmt_R_opt_lo=rlo, pmt_R_opt_up=rup,
                       pmt_chi2_ndof=q["chi2"] / q["ndof"] if q["ndof"] > 0 else np.nan,
                       pmt_valid=bool(q["valid"]))
        rows.append(row)

    return results, pd.DataFrame(rows)


# =========================================================================
# Pick the best setting
# =========================================================================

_METHODS = {
    "pmt":      ("pmt_R_opt", "pmt_R_opt_err", "pmt_valid"),
    "parabola": ("par_R0",    "par_R0_err",    "par_in_range"),
    "data":     ("R_data_min", "R_data_min_err", None),
}


def best_setting(summary, *, method="pmt", n_sigma=1.0):
    '''
    Pick the (G, S) with the lowest resolution, and list the ones that
    are statistically compatible with it.
    '''
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}")
    col, err, flag = _METHODS[method]
    if col not in summary:
        raise ValueError(f"no fit results for method '{method}'")
    s = summary[np.isfinite(summary[col])]
    if flag is not None:
        s = s[s[flag].fillna(False).astype(bool)]
    if s.empty:
        raise ValueError(f"no usable '{method}' result")
    best = s.loc[s[col].idxmin()]
    tol = n_sigma * np.hypot(s[err], best[err])
    compatible = s[(s[col] - best[col]).abs() <= tol].sort_values(col)
    return {"best": best, "compatible": compatible, "method": method}


# =========================================================================
# Re-exports for backward compatibility
# =========================================================================
# These used to live in this module; now they are in scan_plotting.py.
# Importing them here keeps `from lab_compton.resolution_scan import
# plot_groups` working. The imports must come after the function
# definitions above to avoid circular imports.
# noqa: E402

from lab_compton.scan_plotting import (      
    plot_scan, plot_groups, plot_optimum_map,
    print_summary, print_group_summary,
)
