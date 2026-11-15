'''
spectrum_fit.py

Library to define all the utilities regarding fit, fwhm and resolution.
Everything is done in channels, not in energy, to avoid to propagate the errors.

The idea for the fitting is the following.

- The peak is modelled as a Gaussian or a Crystall ball (account for asymetries); for the resolution, I think the gaussian is enough, also 'cause it's easier to compute the FWHM; other possible models are asymettric gaussian, EMG, Johnson Su... 
  
- The bkg is modelled as a linear model + erf function (optional); other ideas 
  are polynomials, flat, exp or exp(polynomials)
  
All the fit normalization parameters are bounded to be > 0, the fit is an ExtendedBinnedNLL computed using minuit (both migrad and minos for asymettric errors)

The idea is to create a class for the signal and one for the bkg
'''

import warnings
import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
import lab_compton.parameters as par
from dataclasses import dataclass, field
from scipy.signal import find_peaks, peak_widths
from scipy.special import ndtr, log_ndtr, exprel
from scipy.stats import norm, crystalball, exponnorm
from scipy.optimize import root_scalar, minimize_scalar, brentq
import math

_LN2 = math.log(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


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
    
    @staticmethod
    def logpdf_z(z):
        '''
        Note: for GaussianSignal the fast path in _fwhm_and_gradient never
        calls this method (it uses FWHM = 2.3548 * sigma analytically), but
        having it here keeps the interface uniform across signal classes and
        allows the generic solver to work if the fast path is ever disabled.
        '''
        return -0.5 * z * z

    @staticmethod
    def mode_z(sigma_unused=None):
        '''
        Mode of the standard normal pdf, at z = 0.
        Same note as logpdf_z: not used by the fast path, but kept for
        interface uniformity.
        '''
        return 0.0 

    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        a0 = np.median(y)  #guess of the bkg using the median of counts
        A0 = max(float(y.max() - a0), 1.0) #peak height
        
        # estimee of the mean (use baricenter)
        mask = y > a0 + 0.5 * A0
        if np.any(mask):
            mu0 = float(np.sum(x[mask] * y[mask]) / np.sum(y[mask]))
        else:
            mu0 = float(x[np.argmax(y)]) 
        
        # width: how much channels above half maximum
        above = np.count_nonzero(y > a0 + 0.5 * A0)
        sigma0 = max(above / par.FWHM_FACTOR, 0.7)
        return {"n_sig": A0 * sigma0 / INV_SQRT_2PI,
                "mu": mu0, "sigma": sigma0}
    
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
    def logpdf_z(z, alpha, n):
        '''
        Log of the (unnormalised) pdf in the standardised variable
        z = (x - mu)/sigma, with mode at z = 0. Scalar-only, used by
        FitResult._fwhm_and_gradient via fwhm_from_logpdf.
        '''
        log_A = n * math.log(n / alpha) - 0.5 * alpha * alpha
        B = n / alpha - alpha
        if z > -alpha:
            return -0.5 * z * z
        else:
            return log_A - n * math.log(B - z)
    
    @staticmethod
    def mode_z(alpha, n):
        '''
        Position of the maximum of the standardised CB pdf. By construction,
        the Gaussian core is centered at z = 0, so the mode is at z = 0
        for any alpha > 0 and n > 1.
        '''
        return 0.0
                
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

class AsymGaussianSignal:
    '''
    Two-piece (split) Gaussian, mode at mu:
        sigma_L = sigma * (1 - asym)   (x < mu)
        sigma_R = sigma * (1 + asym)   (x >= mu)
    With this parametrisation FWHM = 2.3548 * sigma for EVERY asym (exact):
    asym moves area between the two sides but does not change the FWHM.
    asym = 0 is the Gaussian, and it is an interior point of the allowed range.
    Parameters: n_sig, mu, sigma, asym
    '''
 
    params = ("n_sig", "mu", "sigma", "asym")
 
    @staticmethod
    def cdf(x, n_sig, mu, sigma, asym):
        s_l, s_r = sigma * (1.0 - asym), sigma * (1.0 + asym)
        d = x - mu
        left = 2.0 * s_l * ndtr(d / s_l)
        right = s_l + 2.0 * s_r * (ndtr(d / s_r) - 0.5)
        return n_sig * np.where(d < 0, left, right) / (s_l + s_r)
 
    @staticmethod
    def pdf(x, n_sig, mu, sigma, asym):
        s_l, s_r = sigma * (1.0 - asym), sigma * (1.0 + asym)
        d = x - mu
        s = np.where(d < 0, s_l, s_r)
        return n_sig * math.sqrt(2.0 / math.pi) / (s_l + s_r) * np.exp(-0.5 * (d / s) ** 2)
 
    @staticmethod
    def logpdf_z(z, asym):
        s = (1.0 - asym) if z < 0 else (1.0 + asym)
        return -0.5 * (z / s) ** 2
 
    @staticmethod
    def mode_z(asym):
        return 0.0
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        g = GaussianSignal.initial_guess(x, y, lo_edge, hi_edge)
        g["asym"] = 0.0
        return g
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        lim = GaussianSignal.limits(lo_edge, hi_edge)
        lim["asym"] = (-0.9, 0.9)
        return lim
 
 
class EMGSignal:
    '''
    Exponentially modified Gaussian with the tail at LOW energy (scipy's exponnorm has
    the tail on the right, so x is mirrored: z -> -z). Gaussian core N(mu, sigma) convolved
    with an exponential of scale tau = K * sigma.
 
    mu is the centre of the Gaussian COMPONENT, not the peak position: the mode is at
    mu - mode_z(K)*sigma, i.e. between 0.05 and 1.5 sigma below mu for K in [0.05, 5].
    (FitResult.centroid() and peak_resolution() still use mu.)
    K -> 0 is the Gaussian limit, at the BOUNDARY of the parameter space.
    Parameters: n_sig, mu, sigma, K
    '''
 
    params = ("n_sig", "mu", "sigma", "K")
 
    @staticmethod
    def cdf(x, n_sig, mu, sigma, K):
        return n_sig * exponnorm.sf(-(x - mu) / sigma, K)
 
    @staticmethod
    def pdf(x, n_sig, mu, sigma, K):
        return n_sig * exponnorm.pdf(-(x - mu) / sigma, K) / sigma
 
    @staticmethod
    def logpdf_z(z, K):
        # log of exponnorm's pdf at -z, written with log_ndtr (no overflow, no cancellation)
        return -math.log(K) + 0.5 / K ** 2 + z / K + log_ndtr(-z - 1.0 / K)
 
    @staticmethod
    def mode_z(K):
        # d/dz log f = 0  <=>  phi(y)/Phi(y) = 1/K with y = -z - 1/K; monotone in y, so one root.
        # Bracket (-100, 100) is valid for K >= 0.02 (the class limits start at 0.05).
        g = lambda y: -0.5 * y * y - 0.5 * math.log(2.0 * math.pi) - log_ndtr(y) + math.log(K)
        y = brentq(g, -100.0, 100.0, xtol=1e-14, rtol=1e-14)
        return -(y + 1.0 / K)
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        g = GaussianSignal.initial_guess(x, y, lo_edge, hi_edge)
        g["K"] = 0.5
        return g
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        lim = GaussianSignal.limits(lo_edge, hi_edge)
        lim["K"] = (0.05, 5.0)
        return lim
 
class JohnsonSUSignal:
    '''
    Johnson SU, parametrised by MEAN (mu) and STANDARD DEVIATION (sigma) plus two shape parameters.
    Standard form: U = sinh((N - gamma)/delta) with N ~ N(0,1), i.e. gamma + delta*asinh(U) is standard normal
    (scipy.stats.johnsonsu with loc=0, scale=1). U has mean m1 and variance s1^2 in closed form, and
    z = (x - mu)/sigma  <->  U = m1 + s1*z, so z has mean 0 and variance 1 for every (gamma, delta).
    gamma > 0: tail at low energy; gamma = 0: symmetric but heavier-tailed than a Gaussian.
    Why not scipy's (loc, scale): near the Gaussian limit (delta -> infinity) the pairs (scale, delta) with fixed
    scale/delta give practically the same pdf (e.g. delta=20, scale=60 vs delta=50, scale=150: they differ by 4e-4
    of the peak height), a flat valley for the fit. With mean/std the Gaussian limit is regular in mu and sigma.
    The Gaussian itself is at the BOUNDARY (delta -> infinity), and mu is the mean, not the mode.
    Parameters: n_sig, mu, sigma, gamma, delta
    '''
 
    params = ("n_sig", "mu", "sigma", "gamma", "delta")
 
    @staticmethod
    def _moments(gamma, delta):
        e = math.exp(1.0 / delta ** 2)
        m1 = -math.exp(0.5 / delta ** 2) * math.sinh(gamma / delta)
        v1 = 0.5 * (e - 1.0) * (e * math.cosh(2.0 * gamma / delta) + 1.0)
        return m1, math.sqrt(v1)
 
    @staticmethod
    def cdf(x, n_sig, mu, sigma, gamma, delta):
        m1, s1 = JohnsonSUSignal._moments(gamma, delta)
        u = m1 + s1 * (x - mu) / sigma
        return n_sig * ndtr(gamma + delta * np.arcsinh(u))
 
    @staticmethod
    def pdf(x, n_sig, mu, sigma, gamma, delta):
        m1, s1 = JohnsonSUSignal._moments(gamma, delta)
        u = m1 + s1 * (x - mu) / sigma
        w = gamma + delta * np.arcsinh(u)
        return n_sig * (s1 / sigma) * delta / np.sqrt(1.0 + u * u) * INV_SQRT_2PI * np.exp(-0.5 * w * w)
 
    @staticmethod
    def logpdf_z(z, gamma, delta):
        m1, s1 = JohnsonSUSignal._moments(gamma, delta)
        u = m1 + s1 * z
        w = gamma + delta * math.asinh(u)
        return math.log(delta * s1) - 0.5 * math.log1p(u * u) - 0.5 * w * w - 0.5 * math.log(2.0 * math.pi)
 
    @staticmethod
    def mode_z(gamma, delta):
        # d/du log f = 0  <=>  u/sqrt(1+u^2) + delta*(gamma + delta*asinh u) = 0: strictly increasing in u, one root,
        # and |u/sqrt(1+u^2)| < 1 gives the bracket below
        g = lambda u: u / math.sqrt(1.0 + u * u) + delta * (gamma + delta * math.asinh(u))
        lo, hi = math.sinh((-1.0 / delta - gamma) / delta), math.sinh((1.0 / delta - gamma) / delta)
        u = brentq(g, lo, hi, xtol=1e-14, rtol=1e-14)
        m1, s1 = JohnsonSUSignal._moments(gamma, delta)
        return (u - m1) / s1
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        g = GaussianSignal.initial_guess(x, y, lo_edge, hi_edge)
        g["gamma"] = 0.0
        g["delta"] = 5.0
        return g
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        lim = GaussianSignal.limits(lo_edge, hi_edge)
        lim["gamma"] = (-3.0, 3.0)
        lim["delta"] = (1.5, 15.0)
        return lim
 
 
SIGNALS = {
    "gauss":  GaussianSignal,
    "cb":     CrystalBallSignal,
    "agauss": AsymGaussianSignal,
    "emg":    EMGSignal,
    "jsu":    JohnsonSUSignal,
}
 

### BACKGROUND MODELS

class LinearBackground:
    '''
    Linear bkg outside the window (instead of y = mx + q, I take the line 
    passing through (x1, y1) and (x2, y2) given by lo_edge and high_edge.
    In this form, y(x) = y1 + (y2-y1) frac{x - x1}{x2 - x1}
    
    y(x) = b_left + (b_right - b_left) * frac{x-lo_edge}{hi_edge - lo_edge}
    
    Parameters: b_left, b_right
    '''
    
    params = ("b_left", "b_right")
    
    #w = window
    #t = track from lo_edge
    
    @staticmethod
    def cdf(x, b_left, b_right, lo_edge, hi_edge):
        #just the integral of the pdf
        t = x - lo_edge
        w = hi_edge - lo_edge
        return b_left * t + 0.5 * (b_right - b_left) * t * t / w

    @staticmethod
    def pdf(x, b_left, b_right, lo_edge, hi_edge):
        w = hi_edge - lo_edge
        return b_left + (b_right - b_left) * (x - lo_edge) / w
    
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        k = max(1, len(y) // 10) #10% of the window
        return {"b_left": max(float(y[:k].mean()), 1e-3),
                "b_right": max(float(y[-k:].mean()), 1e-3)}

    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"b_left": (0, None), "b_right": (0, None)}

class FlatBackground:
    '''Constant bkg, b(x) = b_flat. Parameters: b_flat'''
 
    params = ("b_flat",)
 
    @staticmethod
    def pdf(x, b_flat, lo_edge, hi_edge):
        return b_flat * np.ones_like(x, dtype=float)
 
    @staticmethod
    def cdf(x, b_flat, lo_edge, hi_edge):
        return b_flat * (x - lo_edge)
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        k = max(1, len(y) // 10)      # outer 10% on each side, away from the peak
        return {"b_flat": max(float(np.mean(np.concatenate([y[:k], y[-k:]]))), 1e-3)}
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"b_flat": (0, None)}
        
class BernsteinBackground:
    '''
    Polynomial bkg of a given degree in the Bernstein basis on t = (x - lo_edge)/(hi_edge - lo_edge):
        b(t) = sum_i c_i * C(d,i) t^i (1-t)^(d-i),   c_i >= 0   (so b(x) >= 0 everywhere in the window)
    c_0 = b_left and c_d = b_right are the values at the window edges; the interior c_i are control values,
    not values of b. Degree 1 is exactly the linear bkg. Restriction: non-negative coefficients do not span all
    non-negative polynomials (e.g. a parabola that touches 0 inside the window with a negative c_1 is excluded).
    Subclasses set `degree` and `params`.
    '''
 
    degree = None
    params = None
 
    @staticmethod
    def _basis(j, n, t):
        return math.comb(n, j) * t ** j * (1.0 - t) ** (n - j)
 
    @classmethod
    def pdf(cls, x, *args):
        *c, lo_edge, hi_edge = args
        t = (x - lo_edge) / (hi_edge - lo_edge)
        return sum(ci * cls._basis(i, cls.degree, t) for i, ci in enumerate(c))
 
    @classmethod
    def cdf(cls, x, *args):
        # int_0^t B_{i,d} = (1/(d+1)) * sum_{j=i+1}^{d+1} B_{j,d+1}
        *c, lo_edge, hi_edge = args
        w = hi_edge - lo_edge
        t = (x - lo_edge) / w
        n = cls.degree + 1
        return w / n * sum(ci * sum(cls._basis(j, n, t) for j in range(i + 1, n + 1))
                           for i, ci in enumerate(c))
 
    @classmethod
    def initial_guess(cls, x, y, lo_edge, hi_edge):
        g = LinearBackground.initial_guess(x, y, lo_edge, hi_edge)     # start from the straight line
        left, right = g["b_left"], g["b_right"]
        return {name: left + (right - left) * i / cls.degree for i, name in enumerate(cls.params)}
 
    @classmethod
    def limits(cls, lo_edge, hi_edge):
        return {name: (0, None) for name in cls.params}
 
 
class Poly2Background(BernsteinBackground):
    '''Quadratic (Bernstein). Parameters: b_left, b_c1, b_right'''
    degree = 2
    params = ("b_left", "b_c1", "b_right")
 
 
class Poly3Background(BernsteinBackground):
    '''Cubic (Bernstein). Parameters: b_left, b_c1, b_c2, b_right'''
    degree = 3
    params = ("b_left", "b_c1", "b_c2", "b_right")
 
 
class ExpPoly2Background:
    '''
    exp(quadratic): b(t) = exp(p(t)), t = (x - lo_edge)/(hi_edge - lo_edge), with p the parabola through
    (ln b_left, ln b_mid, ln b_right) at t = 0, 1/2, 1. All three parameters are VALUES of b (at the two edges and at the
    centre of the window), always > 0; b_mid = sqrt(b_left * b_right) is the pure exponential.
    No closed-form cdf: the integral from lo_edge to x is a 24-point Gauss-Legendre rule (smooth integrand).
    Parameters: b_left, b_mid, b_right
    '''
 
    params = ("b_left", "b_mid", "b_right")
    _GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
 
    @staticmethod
    def _logb(t, b_left, b_mid, b_right):
        return (2.0 * (t - 0.5) * (t - 1.0) * math.log(b_left)
                - 4.0 * t * (t - 1.0) * math.log(b_mid)
                + 2.0 * t * (t - 0.5) * math.log(b_right))
 
    @staticmethod
    def pdf(x, b_left, b_mid, b_right, lo_edge, hi_edge):
        t = (x - lo_edge) / (hi_edge - lo_edge)
        return np.exp(ExpPoly2Background._logb(t, b_left, b_mid, b_right))
 
    @staticmethod
    def cdf(x, b_left, b_mid, b_right, lo_edge, hi_edge):
        w = hi_edge - lo_edge
        t = np.asarray((x - lo_edge) / w, dtype=float)[..., None]
        nodes = 0.5 * t * (ExpPoly2Background._GL_X + 1.0)             # nodes in [0, t]
        vals = np.exp(ExpPoly2Background._logb(nodes, b_left, b_mid, b_right))
        return w * 0.5 * t[..., 0] * (vals * ExpPoly2Background._GL_W).sum(axis=-1)
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        g = LinearBackground.initial_guess(x, y, lo_edge, hi_edge)
        return {"b_left": g["b_left"], "b_mid": math.sqrt(g["b_left"] * g["b_right"]), "b_right": g["b_right"]}
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"b_left": (1e-6, None), "b_mid": (1e-6, None), "b_right": (1e-6, None)}
 
class ExpBackground:
    '''
    Exponential bkg parametrised, like the linear one, by its values at the window edges:
        b(x) = b_left * (b_right / b_left) ** t ,   t = (x - lo_edge) / (hi_edge - lo_edge)
    Same parameter names, meaning and initial guess as LinearBackground, so the two are
    directly comparable. Both values must be > 0 (lower limit 1e-6).
    Parameters: b_left, b_right
    '''
 
    params = ("b_left", "b_right")
 
    @staticmethod
    def pdf(x, b_left, b_right, lo_edge, hi_edge):
        t = (x - lo_edge) / (hi_edge - lo_edge)
        return b_left * np.exp(math.log(b_right / b_left) * t)
 
    @staticmethod
    def cdf(x, b_left, b_right, lo_edge, hi_edge):
        # integral of b from lo_edge to x = b_left * w * t * exprel(k t), k = ln(b_right/b_left);
        # exprel(y) = (e^y - 1)/y is exact at y = 0, so b_right == b_left needs no special case
        w = hi_edge - lo_edge
        t = (x - lo_edge) / w
        k = math.log(b_right / b_left)
        return b_left * w * t * exprel(k * t)
 
    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        return LinearBackground.initial_guess(x, y, lo_edge, hi_edge)
 
    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"b_left": (1e-6, None), "b_right": (1e-6, None)}
 
class ComptonStep:
    '''Step function. Parametri: h.'''
    params = ("h",)
    
    @staticmethod
    def cdf(x, h, mu, sigma):
        z = (x - mu) / sigma
        return h * sigma * (z * ndtr(-z) - INV_SQRT_2PI * np.exp(-0.5 * z * z))

    @staticmethod
    def pdf(x, h, mu, sigma):
        return h * norm.sf((x - mu) / sigma)   # sf = 1 - cdf

    @staticmethod
    def initial_guess(x, y, lo_edge, hi_edge):
        return {"h": 0.05 * max(float(np.median(y)), 1.0) + 0.1}

    @staticmethod
    def limits(lo_edge, hi_edge):
        return {"h": (0, None)}   
 
BACKGROUNDS = {
    "flat":     FlatBackground,
    "lin":      LinearBackground,
    "exp":      ExpBackground,
    "poly2":    Poly2Background,
    "poly3":    Poly3Background,
    "exppoly2": ExpPoly2Background,
}    

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
            total += ComptonStep.cdf(x , p["h"], p["mu"], p["sigma"])
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

MODELS = {}
for _s, _S in SIGNALS.items():
    for _b, _B in BACKGROUNDS.items():
        for _step in (False, True):
            MODELS[f"{_s}_{_b}" + ("_step" if _step else "")] = (
                lambda S=_S, B=_B, st=_step: CompositeModel(S, B, step=st))
 
def _crossing(f, z0, step, method, xtol, rtol, max_doublings):
    '''Root of f on [z0, +inf), with f(z0) > 0 and f eventually < 0.'''
    a, b = z0, z0 + step
    for _ in range(max_doublings):
        if f(b) <= 0.0:
            break
        a, step = b, 2.0 * step
        b = a + step
    else:
        raise RuntimeError(
            "half maximum not bracketed: the pdf does not fall below half maximum"
        )
    return root_scalar(f, bracket=(a, b), method=method,
                       xtol=xtol, rtol=rtol).root


def fwhm_from_logpdf(logpdf, mode=None, step=1.0, *,
                     method="brentq", xtol=1e-13, rtol=1e-13,
                     max_doublings=60, mode_search=(-10.0, 10.0)):
    '''
    FWHM of a unimodal density, in the units of the variable of `logpdf`.

    Works in standardised z = (x - mu)/sigma. Scalar in, scalar out.
    Requires logpdf to be the log of the (unnormalised) shape.
    '''
    if mode is None:
        mode = minimize_scalar(lambda z: -logpdf(z), bounds=mode_search,
                               method="bounded",
                               options={"xatol": 1e-12}).x
    l_max = logpdf(mode)
    f_r = lambda z: logpdf(z) - l_max + _LN2
    f_l = lambda z: logpdf(2 * mode - z) - l_max + _LN2
    z_r = _crossing(f_r, mode, step, method, xtol, rtol, max_doublings)
    z_l = 2 * mode - _crossing(f_l, mode, step, method, xtol, rtol, max_doublings)
    return float(z_r - z_l)

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
        '''
        FWHM and its gradient with respect to the fit parameters.

        Fast path: pure Gaussian (analytical, no solver, no FD).

        General path: works in standardised z = (x - mu) / sigma, so
            FWHM = sigma * width_z(shape params)
        and d FWHM / d sigma = width_z (exact, no FD). Only shape parameters
        need numerical derivatives.

        Mode of the standardised pdf:
            - if the signal class exposes `mode_z(*shape_params)`, its
              return value is used (analytical, cheap, exact);
            - otherwise `mode=None` is passed to fwhm_from_logpdf, which
              finds it with minimize_scalar (correct for any unimodal pdf,
              slightly slower, needed for EMG, Johnson SU, ...).

        Requires sig_cls.logpdf_z(z, *shape_params) for non-Gaussian models.
        Shape parameters are identified by excluding n_sig, mu, sigma from
        sig_cls.params.
        '''
        g = np.zeros(len(self.names))
        m_model = MODELS[self.model]()
        sig_cls = m_model.signal

        # ---- Fast path: pure Gaussian ----
        if sig_cls.params == ("n_sig", "mu", "sigma"):
            g[self.names.index("sigma")] = par.FWHM_FACTOR
            return par.FWHM_FACTOR * self.values["sigma"], g

        # ---- General path ----
        sigma = self.values["sigma"]
        shape_names = [k for k in sig_cls.params
                       if k not in ("n_sig", "mu", "sigma")]
        shape_vals = {k: self.values[k] for k in shape_names}

        if not hasattr(sig_cls, "logpdf_z"):
            warnings.warn(
                f"{sig_cls.__name__} has no logpdf_z; using 2.3548*sigma for FWHM",
                stacklevel=2,
            )
            g[self.names.index("sigma")] = par.FWHM_FACTOR
            return par.FWHM_FACTOR * sigma, g

        def _logpdf_for(sv):
            args = [sv[k] for k in shape_names]
            return lambda z: sig_cls.logpdf_z(z, *args)

        def _mode_for(sv):
            '''Mode of the standardised pdf: analytical if available, else None.'''
            if hasattr(sig_cls, "mode_z"):
                return sig_cls.mode_z(*[sv[k] for k in shape_names])
            return None

        # Main width
        try:
            width_z = fwhm_from_logpdf(
                _logpdf_for(shape_vals), mode=_mode_for(shape_vals), step=1.0
            )
        except (RuntimeError, ValueError, OverflowError) as e:
            warnings.warn(
                f"fwhm solver failed ({e}); using 2.3548*sigma", stacklevel=2
            )
            g[self.names.index("sigma")] = par.FWHM_FACTOR
            return par.FWHM_FACTOR * sigma, g

        fw = sigma * width_z
        g[self.names.index("sigma")] = width_z  # exact, no FD

        # Shape parameters by central finite differences.
        # The mode is recomputed for each perturbation, because for models
        # like EMG the mode depends on the shape parameters.
        for k in shape_names:
            h = 1e-6 * max(abs(shape_vals[k]), 1.0)
            up = dict(shape_vals); up[k] += h
            lo = dict(shape_vals); lo[k] -= h
            try:
                w_up = fwhm_from_logpdf(
                    _logpdf_for(up), mode=_mode_for(up), step=1.0
                )
                w_lo = fwhm_from_logpdf(
                    _logpdf_for(lo), mode=_mode_for(lo), step=1.0
                )
                g[self.names.index(k)] = sigma * (w_up - w_lo) / (2 * h)
            except (RuntimeError, ValueError, OverflowError):
                g[self.names.index(k)] = 0.0

        return fw, g
        
    def fwhm(self):
        '''
        FWHM (channels) with a symmetric error from the full covariance matrix.

        Gaussian: 2.3548 * sigma (analytical).
        General: FWHM = sigma * width_z(shape params), computed by
        _fwhm_and_gradient; the error comes from g^T cov g where g is
        the gradient of FWHM w.r.t. all fit parameters.
        '''
        fw, g = self._fwhm_and_gradient()
        return fw, float(np.sqrt(max(g @ self.cov @ g, 0.0)))
	
	#WHY CRYSTALBALL NOT ASYM?
    def fwhm_asym(self):
        '''
        (value, lower_err, upper_err).

        Pure Gaussian: MINOS on sigma propagated linearly (exact, asymmetric).
        Non-Gaussian (Crystal Ball, EMG, Johnson SU, ...): HESSE symmetric
        from the gradient. MINOS on a non-linear function of multiple
        correlated parameters would require a profile-likelihood fit.
        '''
        sig_cls = MODELS[self.model]().signal
        if sig_cls.params == ("n_sig", "mu", "sigma"):
            # Pure Gaussian: MINOS on sigma, exact
            v = par.FWHM_FACTOR * self.values["sigma"]
            _, lo, up = self.param("sigma")
            return v, par.FWHM_FACTOR * lo, par.FWHM_FACTOR * up
        # Non-Gaussian: HESSE symmetric
        v, e = self.fwhm()
        return v, -e, e
        
    def n_events(self):       
        return self.values["n_sig"], self.errors["n_sig"]   # HESSE

    def n_events_asym(self):
        v, lo, up = self.param("n_sig")
        return v, lo, up
    
    
    ### TO UNDERSTAND
    def peak_resolution(self, zero_channel=0.0, zero_channel_err=0.0):
        '''
        Resolution FWHM / (mu - zero_channel) and its error.

        The resolution is FWHM / E, and E is proportional to 
        (mu - zero_channel), where zero_channel is the channel 
        that corresponds to E = 0 (zero_channel = -c0/c1 for E = c0 + c1*ch).
        With zero_channel = 0 (default) the result is correct only if the ADC
        zero is at channel 0; otherwise it is biased.

        zero_channel_err : error on zero_channel, added in quadrature; it
        ignores the correlation between zero_channel and the fit parameters
        (not negligible if this peak was used for the calibration).
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

    # MINOS
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
    
def _find_peak_window(spectrum, mu_ref=None, *, min_channel=0,
                      search_half=None, fit_half_fwhm=2.5, min_half=8,
                      prominence_frac=0.25, height_frac=0.2, n_sigma=5.0,
                      distance=10, smooth=3):
    '''
    Find the most prominent significant peak and return the fit window.

    The search is done in [mu_ref - search_half, mu_ref + search_half]; the whole spectrum is searched
    if mu_ref is None. If search_half is None it is max(50, mu_ref / 2).

    The thresholds are relative to the maximum of the (smoothed) counts in the search range, plus a
    significance cut (the peak must stand n_sigma Poisson sigmas above the median), so that they work for
    short and long runs alike and a spectrum with no peak is reported as such.

    The half-width of the fit window is fit_half_fwhm times the FWHM measured on the data, not a fixed number
    of channels: the peak gets wider (in channels) with gain and HV, and a fixed window that is fine at
    channel 60 covers only +-0.5 FWHM at channel 400.

    Returns
    -------
    (window, found, fwhm_est) : window = (lo, hi), or None if no peak was found; fwhm_est in channels.
    '''
    counts = np.asarray(spectrum.counts, dtype=float)
    first, last = int(spectrum.channels[0]), int(spectrum.channels[-1])
    n = len(counts)
    
    if mu_ref is None:
        # Cerca dall'inizio, ma non sotto min_channel
        lo_s = max(0, int(min_channel - first))
        hi_s = n - 1
    else:
        if search_half is None:
            search_half = max(50, 0.5 * mu_ref)
        lo_s = max(0, int(mu_ref - first - search_half))
        hi_s = min(n - 1, int(mu_ref - first + search_half))
    if hi_s <= lo_s:
        return None, False, np.nan

    raw = counts[lo_s:hi_s + 1]
    c = np.convolve(raw, np.ones(smooth) / smooth, mode="same") if smooth > 1 else raw
    top = float(c.max())
    if top <= 0:
        return None, False, np.nan
    base = max(float(np.median(raw)), 1.0)
    noise = np.sqrt(base / max(smooth, 1))
    peaks, props = find_peaks(
        c,
        prominence=max(prominence_frac * top, n_sigma * noise),
        distance=distance,
        height=max(height_frac * top, base + n_sigma * noise),
    )
    if len(peaks) == 0:
        return None, False, np.nan

    best = int(peaks[int(np.argmax(props["prominences"]))])
    fwhm = float(peak_widths(c, [best], rel_height=0.5)[0][0])
    half = max(int(round(fit_half_fwhm * fwhm)), min_half)
    centre = best + lo_s + first
    # Clampa la finestra per non scendere sotto min_channel
    lo_w = max(centre - half, min_channel, first)
    hi_w = min(centre + half, last)
    return (lo_w, hi_w), True, fwhm
