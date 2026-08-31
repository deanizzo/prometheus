
import numpy as np
from math import pi


#===================================================================================================================   
'''
Solve radiosity model. Note that the model assumes all wall elements have the same optical properties and that "ambient" is the last element in the list
Inputs:
    T = vector of wall-element temperature (K)
    Tamb = ambient temperature (K)
    surface_refl = surface reflectivity in the IR spectrum
    view_factors = 2D array of view factors between all elements ("aperture" is the last element in the list)    
    K = coefficient matrix for radiosity model (optional, will be re-calculated here if not provided)
'''
def solve_radiosity(T, Tamb, surface_refl, view_factors, Kinv = None):   
    
    # Calculate emission (~40% of the time in this function)
    n = T.shape[0]
    ev = np.zeros(n+1)
    ev[:-1] = (1.0-surface_refl) * 5.6704e-8 * T**4 
    ev[-1] =  5.6704e-8 * Tamb**4 # Aperture

    if Kinv is not None:
        J = np.matmul(Kinv, ev)
    else:
        K = calculate_se_coeff_matrix(surface_refl, view_factors)        
        J = np.linalg.solve(K, ev)   # Total IR energy leaving each surface
    #qnet = J - (view_factors * J).sum(1)  # Net IR energy leaving each element [W/m2]  (slow)
    qnet = J - np.matmul(view_factors,J)    # Net IR energy leaving each element [W/m2] (faster)
    return qnet


def solve_radiosity_abs_flux(qinc, surface_refl, view_factors, Kinv = None): 
    
    # Calculate "emission" (e.g. reflected solar energy)
    n = qinc.shape[0]
    ev = np.zeros(n+1)
    ev[:-1] = qinc*surface_refl   # Solar reflection from assumed incident flux distribution

    if Kinv is not None:
        J = np.matmul(Kinv, ev)
    else:
        K = calculate_se_coeff_matrix(surface_refl, view_factors) 
        J = np.linalg.solve(K, ev)   # Total energy leaving each surface (not counting first reflection)
    #qnet = J - (view_factors * J).sum(1)  # Net energy leaving each element [W/m2]    (slow)
    qnet = J - np.matmul(view_factors,J)    # Net IR energy leaving each element [W/m2] (faster)
    return qnet




def calculate_se_coeff_matrix(surface_refl, view_factors):
    Nelem = view_factors.shape[0]
    Nt = Nelem-1
    refl = np.zeros(Nelem)
    refl[0:Nt] = surface_refl
    K = np.zeros_like(view_factors)  
    for i in range(Nt+1):
        for j in range(Nt+1):
            K[i,j] = - refl[i] * view_factors[i,j]        
            if i == j:
                K[i,j] += 1.0
    return K


