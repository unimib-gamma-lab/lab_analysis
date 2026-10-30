'''
utils.py

I define the main functions we'll need during the analysis.
The main ones are:
1) Compton formula
2) Klein-Nishina Cross Section (excplicitly defined, than we'll see how to adapt it to our datas
'''

import numpy as np

GAMMA_SODIUM    = 1274.53  # Gamma energy for Na-22 decay [keV]
ELECTRON_RADIUS = 2.8179403262e-13  # classical electron radius [cm]
ELECTRON_MASS   = 510.99895      # electron rest energy [keV]

def compton(theta, p0, p1, FIT = False):
    """
    Scattered photon energy [keV] for incident energy e_gamma [keV] 
    and angle theta [rad].
    
    Technically we measure 
    	- E'_gamma (our function) 
    	- Theta (our variable)
    	
    The free parameters are defined as:
    	p0 = E_gamma (init at GAMMA_SODIUM)
    	p1 = E_GAMMA / ELECTRON_MASS 
    
    """
    if FIT == False:
        p0 = GAMMA_SODIUM
        p1 = GAMMA_SODIUM / ELECTRON_MASS
    
    return p0 / (1.0 + (p1) * (1.0 - np.cos(theta)))

def klein_nishina(theta, p0, p1, p2, FIT = False):
    """
    dsigma/dOmega [m^2/sr] for unpolarised photons on a free 
    electron at rest; we have to understand to measure it however...
    I'd say something regarding countings or flux probably
    
    The parameters I choose are the following:
    	- p2 = 0.5 * classical electron radius (init at 0.5 * ELECTRON_RADIUS)
    	- "p3" = E' / E
    		   Actually, the idea is to fix the parameters given by 
    		   the Compton, in this way p3 is no longer a parameter.        
    
    """
    
    if FIT == False:
        p0 = GAMMA_SODIUM
        p1 = GAMMA_SODIUM / ELECTRON_MASS
        p2 = 0.5 * (ELECTRON_RADIUS) ** 2
        
    ratio = compton(theta, p0, p1) / p0
    
    return p2 * ratio**2 * (ratio + 1.0 / ratio - np.sin(theta) ** 2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
