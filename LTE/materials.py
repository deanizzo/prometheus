
import numpy as np
import numpy.polynomial.polynomial as nppoly


def create_material(name, props = None):
    if name == 'Salt_60NaNO3_40KNO3':
        return Salt_60NaNO3_40KNO3()
    elif name == 'Salt_68KCl_32MgCl2':
        return Salt_68KCl_32MgCl2()
    elif name == 'Salt_68KCl_32MgCl2_2006':
        return Salt_68KCl_32MgCl2_2006()
    elif name == 'Salt_20NaCl_40KCl_40MgCl2':
        return Salt_20NaCl_40KCl_40MgCl2()
    elif name == 'Salt_20NaCl_40KCl_40MgCl2_old':
        return Salt_20NaCl_40KCl_40MgCl2_old()
    elif name == 'Sodium':
        return Sodium()
    elif name == 'Air':
        return Air()
    elif name == 'SS316':
        return SS316()
    elif name == 'Haynes230':
        return Haynes230()
    elif name == 'Inconel740H':
        return Inconel740H()
    elif name == 'Inconel600':
        return Inconel600()
    elif name == 'GenericConstantProp':
        return GenericMaterial(props)
    else:
        print ('Material name %s not recognized'%name)
        return None
        
    
    

# Polynomial expression: A + Bx + Cx^2 + ...
# x can be either a single value (will return a single value) or a numpy array (will return an array)
def polynomial(x, coeffs):
    #n = len(coeffs)
    #x = float(x) if not hasattr(x, '__len__') else x.astype('float')
    #return sum([coeffs[i]*(x**i) for i in range(n)])
    y = np.zeros(x.shape, dtype = float) if hasattr(x,'__shape__') else 0.0
    n = len(coeffs)
    for i in range(n):
        y*=x
        y+=coeffs[n-1-i]
    return y    
    #return nppoly.polyval(x, coeffs)
    



# Piecewise polynomial
def piecewise_polynomial(x, coeffs, bounds):
    fcns = [lambda x, c = c: polynomial(x,c) for c in coeffs]
    condlist = [np.logical_and(x>=bnds[0], x<bnds[1]) for bnds in bounds]
    vals = np.piecewise(x,condlist, fcns)
    return vals




class Fluid:
    def __init__(self):
        self.T = None  # Current temperature (K)
        self.k = None  # Thermal conductivity (W/m/K)
        self.density = None  # Density (kg/m3)
        self.cp = None       # Heat capacity (J/kg/K)
        self.viscosity = None # Viscosity(kg/m/s)
        return    

class Solid:
    def __init__(self):
        self.T = None  # Current temperature (K)
        self.k = None  # Thermal conductivity (W/m/K)
        self.density = None  # Density (kg/m3)
        self.cp = None       # Heat capacity (J/kg/K)
        self.cte = None # Thermal expansion coefficient (m/m-K)
        self.youngs_modulus = None  # Youngs modulus (MPa)
        self.poisson_ratio = None   # Poisson's ratio
        self.max_allowable_stress = None  # Maximum allowable stress (MPa) ASME BPVC Section II, Part D, Table 1B
        return       
    
    

#----------------------------------------------------------------------------
# "Gen2" Nitrate Salt      
class Salt_60NaNO3_40KNO3(Fluid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [0.3922, 0.0002, 3e-8, -1e-11])
    def density(self,T):
        return polynomial(T, [2299.4, -0.7875, 0.0002, -1e-7])
    def cp(self,T):
        return polynomial(T, [1438.7, 5e-3, 2e-4, -1e-7])
    def viscosity(self,T):
        return polynomial(T-273.15, [0.02270616, -1.199514e-4, 2.279989e-7, -1.473302e-10])

#----------------------------------------------------------------------------
# Binary chloride salt    
class Salt_68KCl_32MgCl2(Fluid):
    def __init__(self):
        return  
    def k(self,T):
        return  polynomial(T-273.15, [0.50470, -1e-4])
    def density(self,T):
        return polynomial(T-273.15, [1894.3, -0.50997])
    def cp(self,T):
        return polynomial(T-273.15, [1009.1, -1.2203e-2, 1.9700e-5])
    def viscosity(self,T):
        return polynomial(T-273.15, [0.014965, -2.9100e-5, 1.7840e-8])
    
    
#----------------------------------------------------------------------------
# Binary chloride salt (2006 properties)  
class Salt_68KCl_32MgCl2_2006(Fluid):
    def __init__(self):
        return  
    def k(self,T):
        return  polynomial(T, [0.4]),    # Willams 2006 = 0.4, Janz 1981 = 0.62
    def density(self,T):
        return polynomial(T-273.15, [1894.3, -0.50997])
    def cp(self,T):
        return polynomial(T-273.15, [1009.1, -1.2203e-2, 1.9700e-5])
    def viscosity(self,T):
        return polynomial(T-273.15, [0.012013, -2.7892e-5, 1.8310e-8]) 
    
    
#----------------------------------------------------------------------------
# Ternary chloride salt (Final properties from Gen3 liquid pathway, 1/2020)
class Salt_20NaCl_40KCl_40MgCl2(Fluid):  #Gen3 liquid pathway salt chemistry team measurements, valid 450-700C
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T-273.15, [0.81121, -0.0010656, 7.1507e-7])
    def density(self,T):
        return polynomial(T-273.15, [1966, -0.5788])  
    def cp(self,T):
        return polynomial(T-273.15, [1289.46, -0.448381]) 
    def viscosity(self,T):
        return 1.e-3 * (0.689069*np.exp(1224.729755/T)) 
      
    
#----------------------------------------------------------------------------
# Ternary chloride salt (Old properties)
class Salt_20NaCl_40KCl_40MgCl2_old(Fluid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T-273.15, [0.50820, -1e-4])
    def density(self,T):
        return polynomial(T-273.15, [1882.1, -0.406])   
    def cp(self,T):
        return polynomial(T-273.15, [1394.6, -0.52799])
    def viscosity(self,T):
        return 1.e-3 * (0.3036*np.exp(2137.3/T))
      
#----------------------------------------------------------------------------
# Liquid Sodium
class Sodium(Fluid):  #"Thermodynamic and Transport properties of sodium liquid and vapor" ANL/RE-95/2
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [124.659, -0.113788, 5.52077e-5, -1.18378e-8])
    def density(self,T):
        return polynomial(T, [995.221, -0.182457, -3.08695e-5])
    def cp(self,T):
        return polynomial(T, [1646.59, -0.848284, 4.50490e-4])
    def viscosity(self,T):
        return np.exp(-6.4406 - 0.3958*np.log(T) + 556.835/T)
    
#----------------------------------------------------------------------------
# Air
class Air(Fluid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [0.00145453, 0.0000872152, -2.20614E-08])
    def density(self,T):
        return 101325/(8.314*T) * 0.02884
    def cp(self,T):
        return polynomial(T, [1037.49, -0.305497, 7.49335e-4, -3.39363E-7])
    def viscosity(self,T):
        return polynomial(T, [1.0765e-6,  7.15173E-08, -5.03525E-11, 2.02799E-14])
    
        

#----------------------------------------------------------------------------
# SS316
class SS316(Solid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [7.7765, 0.0177, -8e-6, 3e-9])
    def density(self,T):
        return polynomial(T, [8349.38, -0.341708, -8.65128e-5])
    def cp(self,T):
        return polynomial(T, [368.455, 0.399548, -1.70558e-4])
    def cte(self,T):
        return polynomial(T, [2.0e-5, -6.0e-9, 9.0e-12])
    def youngs_modulus(self,T):
        return polynomial(T, [1.93e5])
    def poisson_ratio(self,T):
        return 0.31
    def max_allowable_stress(self,T):
        print ('Max allowable stress for SS316 not implemented')
        return

#----------------------------------------------------------------------------
# Haynes 230
class Haynes230(Solid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [2.9837, 0.02])
    def density(self,T):
        return polynomial(T, [8970.0])
    def cp(self,T):
        return piecewise_polynomial(T, coeffs = [[324.77, 0.28723, -1.1949e-4], [-7189.4, 20.247, -0.017558, 5.0833e-6], [617]], bounds = [[0, 873], [873, 1273], [1273, 5000]])
    def cte(self,T):
        return polynomial(T, [1.0029e-5, 4.7758e-9])   
    def youngs_modulus(self,T):
        return polynomial(T, [2.2581e5, -4.5995e1, -1.0736e-2])  
    def poisson_ratio(self,T):
        return 0.31
    def max_allowable_stress(self,T):
        Tmax = T.max() if hasattr(T, '__len__') else T
        s = piecewise_polynomial(T, coeffs = [[303.88, -0.4995, 1.0756e-3, -1.9552e-6, 1.3901e-9], [144], [1552.2, -2.4453, 9.1455e-4, 4.0705e-8], [242.12, -0.20031]], bounds = [[0, 699], [699, 866], [866, 1171], [1171, 5000]])
        if Tmax>1171:
            print ('Warning: extrapolating maximum allowable stress to T = %.0fC. Data range is %.0f-%.0fC' % (Tmax-273.15, 310-273, 1171-273))
        return s
        
#----------------------------------------------------------------------------
# Inconel 740H
class Inconel740H(Solid):
    def __init__(self):
        return  
    def k(self,T):
        return polynomial(T, [6.916, 0.011434, 2.4098e-6])
    def density(self,T):
        return polynomial(T, [8360.0])
    def cp(self,T):
        return piecewise_polynomial(T, coeffs = [[271.37, 0.95363, -1.3682e-3,  6.9017e-7], [3123.2, -7.7568, 7.4204e-3, -2.2366e-6], [669]], bounds = [[0, 873], [873, 1423], [1423, 5000]])
    def cte(self,T):  # Note that this is mean (secant) thermal expansion and not intstananeous
        return piecewise_polynomial(T, coeffs = [[9.4696e-6, 9.3799e-9, -4.0714e-12], [15.026e-6, -5.5562e-9, 5.75e-12], [16.4e-6]], bounds = [[0, 873], [873, 1173], [1173, 5000]])
    def youngs_modulus(self,T):
        return polynomial(T, [2.3214e5, -2.9233e1, -2.7439e-2]) 
    def poisson_ratio(self,T):
        return 0.31
    def max_allowable_stress(self,T):
        Tmax = T.max() if hasattr(T, '__len__') else T
        s = piecewise_polynomial(T, coeffs = [[295], [950.96, -2.6504, 2.6e-3], [276], [-5.2504e3, 13.135, -7.8e-3], [3.8001e3, -6.1489, 2.46e-3], [1.4401e3, -2.3185, 9.4e-4]], 
                          bounds = [[0, 423.15], [423.15, 523.15], [523.15, 823.15], [823.15, 973.15], [973.15, 1073.15], [1073.15, 5000]])
        if Tmax>1173:
            print ('Warning: extrapolating maximum allowable stress to T = %.0fC. Data range is %.0f-%.0fC' % (Tmax-273.15,  40, 900))
        return s
    
#----------------------------------------------------------------------------
# Inconel 600
class Inconel600(Solid):
    def __init__(self):
        return  
    def k(self,T):
        return piecewise_polynomial(T, coeffs = [[11.172, 1.1494e-2, 3.4938e-6], [27.5]], bounds = [[0, 1073], [1073, 5000]])
    def density(self,T):
        return polynomial(T, [8470.0])
    def cp(self,T):
        return piecewise_polynomial(T, coeffs = [[240.34, 1.1168, -1.7618e-3, 1.0582e-6], [433.16, 0.166], [628]], bounds = [[0, 873], [873, 1173], [1173, 5000]])
    def cte(self,T):
        print ('CTE for Inconel600 not implemented')
        return None
    def youngs_modulus(self,T):
        print ('Youngs modulus for Inconel600 not implemented')
        return None
    def poisson_ratio(self,T):
        return 0.31    
    def max_allowable_stress(self,T):
        print ('Max allowable stress for Inconel600 not implemented')
        return None   
   
    
   
#-----------------------------------------------------------------------------
# Generic constant-property material
class GenericMaterial(Solid):
    def __init__(self, props):
        self.props = props
        return  
    def k(self,T):
        return polynomial(T, [self.props['k']])
    def density(self,T):
        return polynomial(T, [self.props['density']])
    def cp(self,T):
        return polynomial(T, [self.props['cp']])
    def cte(self,T):
        return polynomial(T, [self.props['cte']])
    def youngs_modulus(self,T):
        return polynomial(T, [self.props['E']])
    def poisson_ratio(self,T):
        return 0.31    
    def max_allowable_stress(self,T):
        print ('Max allowable stress for GenericMaterial not implemented')
        return None   

    