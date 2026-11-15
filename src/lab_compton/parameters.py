'''
parameters.py

We define here all the constant values we need
'''

import numpy as np

ELECTRON_RADIUS = 2.8179403262e-13  # classical electron radius [cm]
ELECTRON_MASS   = 510.99895      # electron rest energy [keV]
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

E1_KEV = ELECTRON_MASS #Na22 511 KeV peak
E2_KEV = 1274.53       #Na22 1274 KeV peak

#Resolution
R_MIN_511 = 0.055   # lower bound of R for the 511 keV peak
R_MAX_511 = 0.200   # upper bound
D_NDOF_MAX = 3.0    # max reduced deviance for a usable fit
