'''
physics.py

I define the main functions we'll need during the analysis regarding the physics or the fit for example.

The main ones are:
1) Compton formula
2) Klein-Nishina Cross Section (excplicitly defined, than we'll see how to adapt it to our datas)
'''

import numpy as np
import lab_compton.parameters as p

def compton(theta, E_gamma=None):
    '''Scattered photon energy [keV] for incident E_gamma and angle theta.'''
    if E_gamma is None:
        E_gamma = p.E1_KEV # 511 keV
    alpha = E_gamma / p.ELECTRON_MASS
    return E_gamma / (1.0 + alpha * (1.0 - np.cos(theta)))

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
        p0 = p.GAMMA_SODIUM
        p1 = p.GAMMA_SODIUM / p.ELECTRON_MASS
        p2 = 0.5 * (p.ELECTRON_RADIUS) ** 2
        
    ratio = compton(theta, p0, p1) / p0
    
    return p2 * ratio**2 * (ratio + 1.0 / ratio - np.sin(theta) ** 2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
