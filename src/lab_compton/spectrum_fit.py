'''
spectrum_fit.py

Library to define all the utilities regarding fit, fwhm and resolution.
Everything is done in channels, not in energy, to avoid to propagate the errors.

The idea for the fitting is the following.

- The peak is modelled as a Gaussian or a Crystall ball (account for asymetries;
  for the resolution, I think the gaussian is enough, also 'cause it's easier to
  compute the FWHM
  
- The bkg is modelled as a linear model + erf function (optional); other ideas 
  are polynomials and exp
  
All the fit normalization parameters are bounded to be > 0, the fit is an ExtendedBinnedNLL computed using minuit (both migrad and minos for asymettric errors)

The idea is to create a class for the signal and one for the bkg
'''

import warnings
import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
import lab_compton.parameters as par
from scipy.special import ndtr
from scipy.stats import norm, crystalball
from dataclasses import dataclass, field

### SIGNAL MODELS ###

class GaussianSignal:
    '''
    gaussian peak
    '''
    
    params = ("n_sig", "mu", "sigma")
    
    @staticmethod
    def cdf(x, n_sig, mu, sigma):
        return n_sig * ndtr((x - mu) / sigma)

    @staticmethod
    def pdf(x, n_sig, mu, sigma):
        return n_sig * norm.pdf(x, loc=mu, scale=sigma)
    
    #TO FIX
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        k = max(1, len(y) // 10)
        a0 = np.median(y)
        A0 = max(float(y.max() - a0), 1.0)
        i_max = int(np.argmax(y))
        # stima larghezza: quanti canali sopra metà altezza
        above = np.count_nonzero(y > a0 + 0.5 * A0)
        sigma0 = max(above / par.FWHM_FACTOR, 0.7)
        return {"n_sig": A0 * sigma0 * np.sqrt(2 * np.pi),
                "mu": float(x[i_max]), "sigma": sigma0}
    
    #TO FIX
    @staticmethod
    def limits(lo_edge, hi_edge):
        return {
            "n_sig": (0, None),
            "mu": (lo_edge, hi_edge),
            "sigma": (0.3, (hi_edge - lo_edge) / 2)
        }


class CrystalBallSignal:
    '''
    One sided crystall ball (tail at low energy).
    Parameters: n_sig, mu, sigma, alpha, n
    '''    
    
    params = ("n_sig", "mu", "sigma", "alpha", "n")
    
    @staticmethod
    def cdf(x, n_sig, mu, sigma, alpha, n):
        return n_sig * crystalball.cdf(x, alpha, n, loc=mu, scale=sigma)

    @staticmethod
    def pdf(x, n_sig, mu, sigma, alpha, n):
        return n_sig * crystalball.pdf(x, alpha, n, loc=mu, scale=sigma)
        
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        g = GaussianSignal.initial_guess(x, y, lo_edge, hi_edge)
        g["alpha"] = 1.5 #How many sigmas far the tail starts
        g["n"] = 5.0 #How fast the tail decreases
        return g

    @staticmethod
    def limits(lo_edge, hi_edge):
        lim = GaussianSignal.limits(lo_edge, hi_edge)
        lim["alpha"] = (0.1, 20.0)
        lim["n"] = (1.1, 50.0)
        return lim

### BACKGROUND MODELS

class LinearBackground:
    '''
    Linear bkg outside the window
    Parameters: b_left, b_right
    '''
    
    params = ("b_left", "b_right")
    
    #w = window
    #t = track from lo_edge
    
    #TO UNDERSTAND BETTER
    @staticmethod
    def cdf(x, b_left, b_right, lo_edge, hi_edge):
        t = x - lo_edge
        w = hi_edge - lo_edge
        return b_left * t + 0.5 * (b_right - b_left) * t * t / w

    @staticmethod
    def pdf(x, b_left, b_right, lo_edge, hi_edge):
        w = hi_edge - lo_edge
        return b_left + (b_right - b_left) * (x - lo_edge) / w
    
    #TO FIX
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        k = max(1, len(y) // 10)
        return {"b_left": max(float(y[:k].mean()), 1e-3),
                "b_right": max(float(y[-k:].mean()), 1e-3)}

    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"b_left": (0, None), "b_right": (0, None)}
    
class ComptonStep:
    '''Step function. Parametri: h.'''
    params = ("h",)
    
    #TO SEE BETTER ALL
    @staticmethod
    def cdf(x, h, mu, sigma):
        z = (x - mu) / sigma
        den = 1 / (np.sqrt(2*np.pi))
        return h * sigma * (z * ndtr(-z) - den * np.exp(-0.5 * z * z))

    @staticmethod
    def pdf(x, h, mu, sigma):
        return h * norm.sf((x - mu) / sigma)   # sf = 1 - cdf

    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        return {"h": 0.05 * max(float(np.median(y)), 1.0) + 0.1}

    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"h": (0, None)}   
    
class CompositeModel:
    '''
    CDF sum of more components. The shared parameters are used only once

    '''
    def __init__(self, signal_cls, background_cls, *, step=False):
        self.signal = signal_cls
        self.background = background_cls
        self.step = step
        self.params = tuple(signal_cls.params) + tuple(background_cls.params)
        if step:
            self.params += ("h",)

    def cdf(self, x, lo_edge, hi_edge, **p):
        total = np.zeros_like(x)
        # signal: n_sig, mu, sigma [, alpha, n]
        sig_args = [p[k] for k in self.signal.params]
        total += self.signal.cdf(x, *sig_args)
        # bkh: b_left, b_right
        bkg_args = [p[k] for k in self.background.params]
        total += self.background.cdf(x, *bkg_args, lo_edge, hi_edge)
        # gradino: h (usa mu, sigma del segnale)
        if self.step:
            total += ComptonStep.cdf(x  , p["h"], p["mu"], p["sigma"])
        return total

    def initial_guess(self, x, y, lo_edge, hi_edge):
        guess = {}
        guess.update(self.signal.initial_guess(x, y, lo_edge, hi_edge))
        guess.update(self.background.initial_guess(x, y, lo_edge, hi_edge))
        if self.step:
            guess.update(ComptonStep.initial_guess(x, y, lo_edge, hi_edge))
        return guess

    def limits(self, lo_edge, hi_edge):
        lim = {}
        lim.update(self.signal.limits(lo_edge, hi_edge))
        lim.update(self.background.limits(lo_edge, hi_edge))
        if self.step:
            lim.update(ComptonStep.limits(lo_edge, hi_edge))
        return lim

MODELS = {
    "gauss_lin":      lambda: CompositeModel(GaussianSignal, LinearBackground),
    "gauss_lin_step": lambda: CompositeModel(GaussianSignal, LinearBackground,
                                step=True),
    "cb_lin":         lambda: CompositeModel(CrystalBallSignal, 
                                LinearBackground),
    "cb_lin_step":    lambda: CompositeModel(CrystalBallSignal, 
                                LinearBackground, step=True),
}

#TO UNDERSTAND
def _cb_fwhm(sigma, alpha, n):
    '''
    FWHM of the one-sided Crystal Ball pdf (scipy parametrisation: beta = alpha, m = n).

    The maximum is at z = 0 (z = (x - mu)/sigma). On the right the half maximum is the Gaussian one,
    z = +sqrt(2 ln 2). On the left it is the Gaussian one too, unless alpha < sqrt(2 ln 2): then the half
    maximum falls in the power-law tail, pdf ~ A (B - z)^-n, with A = (n/alpha)^n exp(-alpha^2/2) and
    B = n/alpha - alpha, so z = B - (2A)^(1/n).
    '''
    z_r = np.sqrt(2 * np.log(2))
    if alpha >= z_r:
        z_l = -z_r
    else:
        A = (n / alpha) ** n * np.exp(-alpha ** 2 / 2)
        B = n / alpha - alpha
        z_l = B - (2 * A) ** (1.0 / n)
    return sigma * (z_r - z_l)

### FIT ###

@dataclass
class FitResult:
    '''Container for the result of a single peak fit.'''
    model:    str
    names:    tuple
    values:   dict
    errors:   dict
    cov:      np.ndarray
    valid:    bool
    accurate: bool
    at_limit: bool
    above_max_edm: bool
    deviance: float
    ndof:     int
    window:   tuple           
    channels: np.ndarray
    counts:   np.ndarray
    expected: np.ndarray      
    minos:    dict = field(default_factory=dict)
    
    #v = value, e = error
    
    def param(self, name, minos=True):
        '''(value, lower_err, upper_err),  lower<=0<=upper.'''
        if minos and name in self.minos and self.minos[name]["valid"]:
            m = self.minos[name]
            return self.values[name], m["lower"], m["upper"]
        e = self.errors[name]
        return self.values[name], -e, e
        
    def centroid(self):
        '''Symmetric HESSE error on the mean.'''
        return self.values["mu"], self.errors["mu"]
    
    ### TO UNDERSTAND BETTER
    def _fwhm_and_gradient(self):
        '''FWHM and its gradient with respect to the fit parameters (same order as self.names).'''
        g = np.zeros(len(self.names))
        if "alpha" in self.names:
            # Crystal Ball: FWHM from the pdf (it is not 2.3548*sigma when the tail starts close to the core)
            pv = {k: self.values[k] for k in ("sigma", "alpha", "n")}
            for k in pv:
                h = 1e-6 * max(abs(pv[k]), 1.0)
                up, lo = dict(pv), dict(pv)
                up[k] += h
                lo[k] -= h
                g[self.names.index(k)] = (_cb_fwhm(**up) - _cb_fwhm(**lo)) / (2 * h)
            return _cb_fwhm(**pv), g
        g[self.names.index("sigma")] = par.FWHM_FACTOR
        return par.FWHM_FACTOR * self.values["sigma"], g

    def fwhm(self):
        '''
        FWHM (channels) with a symmetric error from the full covariance matrix.
        Gaussian: 2.3548 * sigma. Crystal Ball: computed from the pdf.
        '''
        fw, g = self._fwhm_and_gradient()
        return fw, float(np.sqrt(max(g @ self.cov @ g, 0.0)))

    def fwhm_asym(self):
        '''(value, lower_err, upper_err). MINOS on sigma for Gaussian models; symmetric for Crystal Ball.'''
        if "alpha" in self.names:
            v, e = self.fwhm()
            return v, -e, e
        v = par.FWHM_FACTOR * self.values["sigma"]
        _, lo, up = self.param("sigma")
        return v, par.FWHM_FACTOR * lo, par.FWHM_FACTOR * up
    
    def n_events(self):       
        return self.values["n_sig"], self.errors["n_sig"]   # HESSE

    def n_events_asym(self):
        v, lo, up = self.param("n_sig")
        return v, lo, up
    
    
    ### TO UNDERSTAND
    def peak_resolution(self, zero_channel=0.0, zero_channel_err=0.0):
        '''
        Resolution FWHM / (mu - zero_channel) and its error.

        FIX: the resolution is FWHM / E, and E is proportional to (mu - zero_channel), where zero_channel is
        the channel that corresponds to E = 0 (zero_channel = -c0/c1 for E = c0 + c1*ch). With zero_channel = 0
        (default, old behaviour) the result is correct only if the ADC zero is at channel 0; otherwise it is
        biased (on test2.Spe: 7.30 % instead of 7.64 %).

        zero_channel_err : error on zero_channel, added in quadrature; it ignores the correlation between
        zero_channel and the fit parameters (not negligible if this peak was used for the calibration).
        '''
        fw, g = self._fwhm_and_gradient()
        idx_mu = self.names.index("mu")
        m = self.values["mu"] - zero_channel

        var_fw = g @ self.cov @ g
        cov_fw_mu = g @ self.cov[:, idx_mu]
        var_mu = self.cov[idx_mu, idx_mu]

        R = fw / m
        var_R = (var_fw / m**2
                 + fw**2 * var_mu / m**4
                 - 2 * fw * cov_fw_mu / m**3
                 + (R / m)**2 * zero_channel_err**2)
        return R, np.sqrt(max(var_R, 0.0))
        
    def residuals(self):
        return (self.counts - self.expected) / np.sqrt(
            np.maximum(self.expected, 1e-12)
        )
        
    def reduced_deviance(self):
        return self.deviance / self.ndof

    def warnings(self):
        out = []
        if not self.valid:    out.append("MINIMUM NOT VALID")
        if not self.accurate: out.append("Covariance not accurate")
        if self.at_limit:     out.append("Parameter at boundary")
        if self.above_max_edm: out.append("EDM above max") 
        bad = [k for k, v in self.minos.items() if not v["valid"]]
        if bad:               out.append("MINOS invalid on " + ", ".join(bad))
        return out

    def summary(self):
        lo, hi = self.window
        fw, lo_fw, up_fw = self.fwhm_asym()
        n, n_lo, n_up = self.param("n_sig")
        mu, mu_lo, mu_up = self.param("mu")
        warn = self.warnings()
        s = (f"[{self.model}] win {lo}-{hi} | "
             f"mu = {mu:.3f} +{mu_up:.3f}/-{abs(mu_lo):.3f} | "
             f"FWHM = {fw:.3f} +{up_fw:.3f}/-{abs(lo_fw):.3f} ch | "
             f"N = {n:.0f} +{n_up:.0f}/-{abs(n_lo):.0f} | "
             f"D/ndof = {self.deviance:.1f}/{self.ndof} = "
             f"{self.reduced_deviance():.2f}")
        if warn:
            s += " | " + "; ".join(warn)
        return s
        
def _make_cumulative(model, lo_edge, hi_edge):
    '''
    Wrap CompositeModel.cdf so that ExtendedBinnedNLL can call it as
    f(x, p1, p2, ...). Minuit passes positional args in the order
    given by `name=`; we map them back to keyword arguments.
    '''
    def cumulative(x, *args):
        p = dict(zip(model.params, args))
        return model.cdf(x, lo_edge, hi_edge, **p)
    return cumulative


def fit_peak(channels, counts, window, *,
             model="gauss_lin",
             p0=None,
             strategy=2,
             minos=("n_sig", "mu", "sigma")):
    '''
    Fit a single peak with the chosen composite model.

    Parameters
    ----------
    channels, counts : np.ndarray
        Full spectrum arrays (same length).
    window : (lo, hi)
        Channel window, extrema included.
    model : str
        Key of MODELS. Examples: "gauss_lin", "cb_lin", "gauss_lin_step".
    p0 : dict, optional
        Override initial guess for some parameters.
    strategy : int
        Minuit strategy (0, 1, 2). Default 2.
    minos : tuple, True or None
        Parameters for MINOS errors. True = all. None = none.

    Returns
    -------
    FitResult
    '''
    channels = np.asarray(channels)
    counts = np.asarray(counts)
    lo, hi = window

    sel = (channels >= lo) & (channels <= hi)
    x, y = channels[sel].astype(float), counts[sel].astype(float)
    if len(x) < 12:
        raise ValueError("window too small: at least ~12 channels needed")
    if not np.all(np.diff(x) == 1):
        raise ValueError("window channels must be consecutive")

    # bin edges for ExtendedBinnedNLL: [x0-0.5, x0+0.5, x1+0.5, ...]
    edges = np.concatenate([x - 0.5, [x[-1] + 0.5]])
    lo_edge, hi_edge = edges[0], edges[-1]

    # instantiate model + initial guess
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}', choose from {list(MODELS)}")
    m_model = MODELS[model]()
    start = m_model.initial_guess(x, y, lo_edge, hi_edge)
    if p0:
        start.update(p0)

    # cost function: binned Poisson NLL
    cum = _make_cumulative(m_model, lo_edge, hi_edge)
    cost = ExtendedBinnedNLL(y, edges, cum, name=list(m_model.params))

    # Minuit
    m = Minuit(cost, **{k: start[k] for k in m_model.params})
    for name, lim in m_model.limits(lo_edge, hi_edge).items():
        m.limits[name] = lim
    m.strategy = strategy
    m.migrad()
    m.hesse()

    # optional MINOS
    minos_out = {}
    if minos and m.valid:
        to_run = list(m_model.params) if minos is True else \
                 [k for k in minos if k in m_model.params]
        m.minos(*to_run)
        for k in to_run:
            me = m.merrors[k]
            minos_out[k] = {
                "lower": float(me.lower),
                "upper": float(me.upper),
                "valid": bool(me.is_valid),
            }

    # reconstruct the expected counts per bin
    vals = {k: float(m.values[k]) for k in m_model.params}
    cum_at_edges = cum(edges, *[vals[k] for k in m_model.params])
    expected = np.diff(cum_at_edges)

    return FitResult(
        model=model,
        names=tuple(m_model.params),
        values=vals,
        errors={k: float(m.errors[k]) for k in m_model.params},
        cov=np.array(m.covariance),
        valid=bool(m.valid),
        accurate=bool(m.accurate),
        at_limit=bool(m.fmin.has_parameters_at_limit),
        above_max_edm=bool(m.fmin.is_above_max_edm),
        deviance=float(m.fval),
        ndof=len(y) - len(m_model.params),
        window=(int(x[0]), int(x[-1])),
        channels=x.astype(int),
        counts=y,
        expected=expected,
        minos=minos_out,
    )
    
def _make_row(name, window, fit, zero_channel=0.0, zero_channel_err=0.0):
    '''Flatten a FitResult into a dict suitable for a DataFrame.'''
    mu, mu_err        = fit.centroid()
    fwhm, fwhm_err    = fit.fwhm()
    n, n_lo, n_up     = fit.n_events_asym()  
    R, R_err          = fit.peak_resolution(zero_channel, zero_channel_err)
    return {
        "name":       name,
        "valid":      fit.valid,
        "accurate":   fit.accurate,
        "window_lo":  window[0],
        "window_hi":  window[1],
        "mu":         mu,        "mu_err":     mu_err,
        "fwhm":       fwhm,      "fwhm_err":   fwhm_err,
        "R":          R,         "R_err":      R_err,
        "zero_channel": zero_channel,
        "n_sig":      n,         "n_lo":       n_lo,  "n_up": n_up,
        "deviance":   fit.deviance,  "ndof":   fit.ndof,
        "warnings":   fit.warnings(),
    }


def fit_scan(spectra, window, *, model="gauss_lin", p0=None,
             zero_channel=0.0, zero_channel_err=0.0, **fit_kw):
    '''
    Fit the same peak on many spectra, one after another.

    Parameters
    ----------
    spectra : dict {str: Spectrum}
        Output of `read_dir`, or any dict of Spectrum objects.
    window : tuple (lo, hi) OR dict {name: (lo, hi)}
        If a tuple, the same channel window is used for all spectra
        (fine when the gain does not change, e.g. same HV).
        If a dict, each spectrum uses its own window (needed for
        voltage scans, where the peak shifts with gain — combine
        with `auto_windows`). A dict must contain every spectrum name.
    model : str
        Key of MODELS.
    p0 : dict, optional
        Initial-guess override forwarded to every fit.
    zero_channel, zero_channel_err : float
        Channel of E = 0 (and its error) used for the resolution R = FWHM/(mu - zero_channel).
        The default 0.0 is correct only if the ADC zero is at channel 0 (see FitResult.peak_resolution).
    **fit_kw :
        Forwarded to `fit_peak` (e.g. strategy=1, minos=None for faster scans).

    Returns
    -------
    list of dict
        One row per spectrum, in the order of `spectra`. Rows for
        failed fits contain "valid": False and an "error" message.
    '''
    names = list(spectra)
    rows = []
    for name in names:
        s = spectra[name]
        w = window[name] if isinstance(window, dict) else window
        try:
            fit = fit_peak(s.channels, s.counts, window=w,
                           model=model, p0=p0,  **fit_kw)
            rows.append(_make_row(name, w, fit, zero_channel, zero_channel_err))
        except ValueError as e:
            rows.append({"name": name, "valid": False, "error": str(e)})
    return rows
    
def auto_windows(spectra, mu_ref, *, search_half=60, fit_half=14,
                 prominence_frac=0.25, height_frac=0.2, n_sigma=5.0, distance=10, smooth=3,
                 return_found=False):
    '''
    For each spectrum, find the peak closest to `mu_ref` and return a
    per-spectrum fit window centered on it.

    FIX: the thresholds are now relative to the maximum of the search range (they were absolute counts,
    so they failed on short runs), the counts are lightly smoothed, and the fallback is no longer silent.

    Parameters
    ----------
    spectra : dict {str: Spectrum}
    mu_ref : float
        Reference peak position (channel), e.g. from the nominal-HV run.
    search_half : int
        How far from `mu_ref` to search for the shifted peak (channels).
    fit_half : int
        Half-width of the fit window around the found peak.
    prominence_frac, height_frac : float
        Passed to scipy.signal.find_peaks as fractions of the maximum of the (smoothed) counts
        in the search range.
    n_sigma : float
        Minimum significance: a peak must stand out of the baseline (median of the counts in the search
        range) by at least n_sigma times the Poisson noise of the smoothed counts. Without this, a spectrum
        with no peak would still return a noise bump, because the other thresholds are relative.
    distance : int
        Minimum distance between peaks, in channels.
    smooth : int
        Width of the moving average applied before the search (1 = none).
    return_found : bool
        If True, return (windows, found) with found = {name: bool}.

    Returns
    -------
    dict {name: (lo, hi)}
        If no peak is found for a spectrum, the window is centered on `mu_ref` and a warning is issued.
    '''
    from scipy.signal import find_peaks

    out, found = {}, {}
    for name, s in spectra.items():
        lo_search = max(0, int(mu_ref - search_half))
        hi_search = min(len(s.counts) - 1, int(mu_ref + search_half))
        c = np.asarray(s.counts[lo_search:hi_search + 1], dtype=float)
        if smooth > 1:
            c = np.convolve(c, np.ones(smooth) / smooth, mode="same")
        top = float(c.max()) if len(c) else 0.0
        peaks = np.array([], dtype=int)
        if top > 0:
            base = max(float(np.median(s.counts[lo_search:hi_search + 1])), 1.0)
            noise = np.sqrt(base / max(smooth, 1))
            peaks, _ = find_peaks(c,
                                  prominence=max(prominence_frac * top, n_sigma * noise),
                                  distance=distance,
                                  height=max(height_frac * top, base + n_sigma * noise))
        if len(peaks) == 0:
            warnings.warn(f"auto_windows: no peak found for '{name}' within "
                          f"+-{search_half} channels of {mu_ref}; using a window around mu_ref",
                          stacklevel=2)
            out[name] = (int(mu_ref - fit_half), int(mu_ref + fit_half))
            found[name] = False
            continue
        peaks_global = peaks + lo_search
        best = int(peaks_global[np.argmin(np.abs(peaks_global - mu_ref))])
        out[name] = (best - fit_half, best + fit_half)
        found[name] = True
    return (out, found) if return_found else out
