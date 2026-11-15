'''
calibration.py

Energy calibration utilities: convert channel positions into energies
and compute the energy resolution of a peak.

The calibration is a polynomial map E(ch) = sum_k a_k * ch^k, where
the coefficients a_k are fitted from pairs (ch, E_known) taken from
standard sources (Cs-137, Co-60, Na-22, ...).

For a linear map E = a_1 * ch + a_0, ch_0 = -a_0/a_1 is the channel
of E = 0 (extrapolated). For a generic polynomial, the "zero channel"
is E(ch_0) = 0, but the R calculation does not need it explicitly:

    R = FWHM_ch * (dE/dch)(mu) / E(mu)

which is adimensional and independent of the absolute energy scale.
'''

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

import lab_compton.parameters as par


# =========================================================================
# Polynomial helpers (coefficients in ASCENDING order: a_0, a_1, a_2, ...)
# =========================================================================

def _poly_eval(coeffs, ch):
    '''Evaluate sum_k a_k * ch^k (scalar or array).'''
    ch = np.asarray(ch, dtype=float)
    out = np.zeros_like(ch)
    for k, a in enumerate(coeffs):
        out = out + a * ch**k
    return out


def _poly_derivative(coeffs, ch):
    '''d/dch of the polynomial at ch (scalar or array).'''
    ch = np.asarray(ch, dtype=float)
    out = np.zeros_like(ch)
    for k, a in enumerate(coeffs):
        if k == 0:
            continue
        out = out + k * a * ch**(k - 1)
    return out


def _poly_eval_model(ch, *coeffs):
    '''Signature for iminuit LeastSquares.'''
    return _poly_eval(coeffs, ch)


# =========================================================================
# Calibration fit
# =========================================================================

def calibrate(points, *, degree=1, n_iter=3, minos=False):
    '''
    Fit an energy calibration E(ch) = sum_k a_k * ch^k with iminuit.

    Parameters
    ----------
    points : iterable of (ch, ch_err, E, E_err?)
        Each tuple gives the channel position of a known line, its
        error, the known energy in keV, and optionally its error
        (default 0). You need at least `degree + 1` points.
    degree : int
        Polynomial degree. 1 = linear (most common), 2 = quadratic, ...
    n_iter : int
        Number of iterations to include the ch uncertainty in the
        weights. Usually converges in 2-3 iterations.
    minos : bool
        If True, also run MINOS on the coefficients (asymmetric errors
        from the profile likelihood). Not usually needed for a
        well-behaved polynomial calibration; HESSE is fine.

    Returns
    -------
    dict
        {'coeffs':  array of coefficients (ascending order: a_0, a_1, ...),
         'cov':     covariance matrix of the coefficients,
         'degree':  int,
         'chi2':    chi-square at the minimum,
         'ndof':    degrees of freedom (n_points - degree - 1),
         'minuit':  the Minuit object (for MINOS, diagnostics, etc.)}
    '''
    if len(points) < degree + 1:
        raise ValueError(f"need at least {degree + 1} points for degree {degree}")

    chs, dchs, Es, dEs = [], [], [], []
    for p in points:
        chs.append(float(p[0]))
        dchs.append(float(p[1]))
        Es.append(float(p[2]))
        dEs.append(float(p[3]) if len(p) > 3 else 0.0)
    chs  = np.array(chs)
    dchs = np.array(dchs)
    Es   = np.array(Es)
    dEs  = np.array(dEs)

    # Parameter names for Minuit, one per coefficient
    names = [f"a{k}" for k in range(degree + 1)]

    # Initial guess from an unweighted polynomial fit
    coeffs_desc = np.polyfit(chs, Es, degree)
    p0 = coeffs_desc[::-1]  # ascending order

    m = None
    popt, pcov = None, None
    for _ in range(n_iter):
        # Effective y-error: the ch uncertainty converted to keV at the
        # current slope, combined in quadrature with the known E error.
        dE_dch = _poly_derivative(p0, chs)
        sigma_eff = np.sqrt((dE_dch * dchs)**2 + dEs**2)
        sigma_eff = np.where(sigma_eff > 0, sigma_eff, 1.0)

        cost = LeastSquares(chs, Es, sigma_eff, _poly_eval_model, name=names)
        m = Minuit(cost, dict(zip(names, p0)))
        m.strategy = 2
        m.migrad()
        m.hesse()

        if not m.valid:
            raise RuntimeError(f"calibration fit did not converge: {m.fmin}")

        popt = np.array([m.values[n] for n in names])
        pcov = np.array(m.covariance)
        p0   = popt

    if minos:
        m.minos()

    return {
        "coeffs": popt,
        "cov":    pcov,
        "degree": degree,
        "chi2":   float(m.fval),
        "ndof":   len(chs) - len(names),
        "minuit": m,
    }


# =========================================================================
# Apply calibration to a DataFrame of fits
# =========================================================================

def apply_calibration(df, calib, *, keep_channel_R=True):
    '''
    Add energy columns to a scan DataFrame and replace R, R_err with
    the energy-based resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_scan. Must contain columns:
        mu, mu_err, fwhm, fwhm_err, R, R_err.
    calib : dict
        Output of calibrate().
    keep_channel_R : bool
        If True (default), keep the channel-based R as 'R_ch'/'R_ch_err'
        before overwriting R/R_err.

    Returns
    -------
    pd.DataFrame
        A copy of df with added columns 'E', 'E_err' and (if
        keep_channel_R) 'R_ch', 'R_ch_err'. R, R_err are replaced with
        the calibrated values.
    '''
    coeffs = np.asarray(calib["coeffs"])
    cov    = np.asarray(calib["cov"])

    df = df.copy()
    if keep_channel_R and "R_ch" not in df.columns:
        df["R_ch"]     = df["R"]
        df["R_ch_err"] = df["R_err"]

    E_out, E_err_out = [], []
    R_out, R_err_out = [], []

    for _, row in df.iterrows():
        mu      = row.get("mu")
        mu_err  = row.get("mu_err", 0.0) or 0.0
        fwhm    = row.get("fwhm")
        fwhm_err = row.get("fwhm_err", 0.0) or 0.0

        if not (np.isfinite(mu) and np.isfinite(fwhm)):
            E_out.append(np.nan); E_err_out.append(np.nan)
            R_out.append(np.nan); R_err_out.append(np.nan)
            continue

        E   = float(_poly_eval(coeffs, mu))
        dE  = float(_poly_derivative(coeffs, mu))

        if not np.isfinite(E) or E <= 0 or not np.isfinite(dE):
            E_out.append(np.nan); E_err_out.append(np.nan)
            R_out.append(np.nan); R_err_out.append(np.nan)
            continue

        R = fwhm * dE / E

        # ---- gradient of E w.r.t. (coeffs, mu) ----
        grad_E_a  = np.array([mu**k for k in range(len(coeffs))])
        grad_E_mu = dE

        # ---- gradient of R w.r.t. (coeffs, mu) via numerical diff ----
        def R_of(mu_=mu, fwhm_=fwhm, coeffs_=coeffs):
            E_  = float(_poly_eval(coeffs_, mu_))
            dE_ = float(_poly_derivative(coeffs_, mu_))
            if E_ <= 0 or not np.isfinite(E_):
                return np.nan
            return fwhm_ * dE_ / E_

        eps_a = 1e-7
        grad_R_a = np.zeros(len(coeffs))
        for k in range(len(coeffs)):
            cp = coeffs.copy(); cp[k] += eps_a
            R_p = R_of(coeffs_=cp)
            grad_R_a[k] = (R_p - R) / eps_a if np.isfinite(R_p) else 0.0

        eps_mu = 1e-7 * max(abs(mu), 1.0)
        R_p = R_of(mu_=mu + eps_mu)
        R_m = R_of(mu_=mu - eps_mu)
        grad_R_mu = ((R_p - R_m) / (2 * eps_mu)
                     if np.isfinite(R_p) and np.isfinite(R_m) else 0.0)

        grad_R_fwhm = dE / E

        # ---- error propagation ----
        var_E = grad_E_a @ cov @ grad_E_a + (grad_E_mu * mu_err)**2
        var_R = (grad_R_a @ cov @ grad_R_a
                 + (grad_R_mu * mu_err)**2
                 + (grad_R_fwhm * fwhm_err)**2)
        # Note: the fit correlation between mu and FWHM is ignored
        # (it lives in FitResult.cov, not in the DataFrame). Effect is
        # typically small compared to the other terms.

        E_out.append(E)
        E_err_out.append(float(np.sqrt(max(var_E, 0.0))))
        R_out.append(float(R))
        R_err_out.append(float(np.sqrt(max(var_R, 0.0))))

    df["E"]     = E_out
    df["E_err"] = E_err_out
    df["R"]     = R_out
    df["R_err"] = R_err_out
    return df


# =========================================================================
# Convenience: estimate zero channel from a 22Na spectrum (both lines)
# =========================================================================

def estimate_zero_channel(fit_511, fit_1274, E1=None, E2=None):
    '''
    Estimate the ADC zero channel from two fits of known lines.

    Useful as a one-off calibration check: run it once on a low-gain
    spectrum where both lines are visible, get ch0, then pass it to
    FitResult.peak_resolution(zero_channel=ch0) for other spectra.

    Parameters
    ----------
    fit_511, fit_1274 : FitResult
        Fits of the two lines (must have 'mu' among the fit names).
    E1, E2 : float, optional
        Known energies of the two lines. Default from parameters.py.

    Returns
    -------
    (ch0, ch0_err) : float
        Zero channel and its error.
    '''
    if E1 is None: E1 = par.E1_KEV
    if E2 is None: E2 = par.E2_KEV

    mu1 = fit_511.values["mu"]
    mu2 = fit_1274.values["mu"]
    err_mu1 = fit_511.errors["mu"]
    err_mu2 = fit_1274.errors["mu"]

    a = (E2 - E1) / (mu2 - mu1)
    ch0 = mu1 - E1 / a

    # d ch0 / d mu1 = 1 + E1/(E2-E1),  d ch0 / d mu2 = -E1/(E2-E1)
    k = E1 / (E2 - E1)
    var_ch0 = (1 + k)**2 * err_mu1**2 + k**2 * err_mu2**2
    return float(ch0), float(np.sqrt(max(var_ch0, 0.0)))

# =========================================================================
# Print
# =========================================================================

def print_calibration(calib):
    '''Print coefficients, errors and chi2/ndof of a calibration fit.'''
    coeffs = calib["coeffs"]
    cov    = calib["cov"]
    errs   = np.sqrt(np.diag(cov))
    print("\n=== Energy calibration ===")
    if calib.get("ndof"):
        print(f"  degree = {calib['degree']}   "
              f"chi2/ndof = {calib['chi2']:.2f}/{calib['ndof']} = "
              f"{calib['chi2'] / calib['ndof']:.2f}")
    else:
        print(f"  degree = {calib['degree']}")
    for k, (a, e) in enumerate(zip(coeffs, errs)):
        print(f"  a_{k} = {a:.6g} ± {e:.6g}")

    m = calib.get("minuit")
    if m is not None and hasattr(m, "merrors") and len(m.merrors) > 0:
        print("  MINOS intervals:")
        for name, me in m.merrors.items():
            print(f"    {name}: +{me.upper:.6g} / -{abs(me.lower):.6g}"
                  f"  {'[valid]' if me.is_valid else '[invalid]'}")
