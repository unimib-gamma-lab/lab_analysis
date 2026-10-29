"""Klein-Nishina differential cross section (example of a shared helper)."""
import numpy as np

R_E = 2.8179403262e-15  # classical electron radius [m]
M_E_C2 = 510.99895      # electron rest energy [keV]


def compton_energy(e_gamma, theta):
    """Scattered photon energy [keV] for incident energy e_gamma [keV] and angle theta [rad]."""
    return e_gamma / (1.0 + (e_gamma / M_E_C2) * (1.0 - np.cos(theta)))


def klein_nishina(e_gamma, theta):
    """dsigma/dOmega [m^2/sr] for unpolarised photons on a free electron at rest."""
    ratio = compton_energy(e_gamma, theta) / e_gamma
    return 0.5 * R_E**2 * ratio**2 * (ratio + 1.0 / ratio - np.sin(theta) ** 2)
