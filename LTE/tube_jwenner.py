
import numpy as np
import tube_view_factors
import materials
import radiosity
import operating_conditions
import settings
import cross_section
import elastic_stress



def solve_tube_time_point(inputs):
    tube = inputs[0]
    tube.solve_time_point(time = inputs[1], update_operating_conditions = inputs[2], calculate_stress = inputs[3], verbosity = inputs[4])
    return tube

def solve_tube_transient(inputs):
    tube = inputs[0]
    tube.solve_transient(initialize_soln = inputs[1], calculate_stress = inputs[2], verbosity = inputs[3])    
    return tube



class Tube:
    def __init__(self):  
        self.tube_material_name = 'Haynes230'            # Tube material (functions defined for 'Haynes230', 'SS316')
        self.tube_material_props = {}                    # Custom constant material properties to use if 'tube_material_name' = 'GenericConstantProp'
        self.HTF_material_name = 'Salt_60NaNO3_40KNO3'   # HTF material 
        
        self.allowable_pressure_stress = None   # Allowable pressure stress (So)
        self.allowable_thermal_stress = None            # Allowable stress (optional), either a constant value or a 2D array with columns of temperature (K) and allowalbe stress
        
        
        #--- Tube sizing
        self.OD = 0.04              # Tube OD (m)
        self.twall = 0.0012         # Tube wall thickness
        self.roughness = 4.6e-5     # Tube wall roughness (m)
        self.tube_bends_90 = 0      # Number of 90 degree bends per tube (assumed to occur on each end)
        self.tube_bends_45 = 0      # Number of 45 degree bends per tube  (assumed to occur on each end)   
        self.length = 1.0           # Tube length (m)
        self.mflow = float('nan')   # Tube mass flow (kg/s)
        
        self.ID = float('nan')      # Tube ID (m) (set in initialize)
        self.ro = float('nan')      # Outer tube radius (m) (set in initialize)
        self.ri = float('nan')      # Inner tube radius (m) (set in initialize)

        
        #--- Heat transfer parameters and conditions
        self.is_solar = True                         # Exposed to solar radiation
        self.solar_abs = 0.94                        # Tube solar absorptivity
        self.emis = 0.88                             # Tube IR emissivity   
        self.hext = 10.0                             # External heat loss coefficient  
        
        
        self.flux_dist = 'cosine' # changed 1/29/2025 was 'cosine'
        self.hext_dist = 'step'
        #self.relative_flux_function = lambda theta: np.cos(np.minimum(np.pi/2, np.abs(theta)))  # Function describing incident flux distribution around the tube circumference
        #self.relative_hext_function = lambda theta: np.heaviside(np.pi/2-theta, 0)              # Function describing hext around tube circumference (Step function with relative hext = 1 at front of tube, 0 at back of tube)

        self.operating_conditions = operating_conditions.OperatingConditions()        # Current single-valued operating conditions (flux profile assumed to be supplied at self.disc.nz resolution)
        self.trans_operating_conditions = operating_conditions.OperatingConditions()  # Transient operating conditions

        self.htc_correlation = 'Gnielinski-smooth'   # Heat transfer correlation ('Sieder-Tate', 'Gnielinski-smooth', 'Gnielinski-rough', 'Petukhov-smooth', 'Petukhov-rough', 'Dittus-Boelter', 'Colburn', 'Garcia-2005-wire', 'Vicente-2004-corrugated', 'Vicente-2002-dimpled')        
        self.use_wall_visc_correction = True         # Use Sieder-Tate Nu correction for viscosity at wall T
        self.flow_against_gravity = False            # Is flow upward (against gravity)?
        self.include_gravity = True                  # Include the effects of gravity in pressure drop calculations (set to False for horizontal flow)


        #---- Model and numerical settings
        self.options = settings.SolutionOptions()
        self.specified_Tfout = None          # Specified tube outlet temperature (if not None, the mass flow will be iteratively adjusted to achieve the specified value)
        self.is_fully_developed = True       # Is flow fully developed at start of tube?
        self.Ttol = 1.e-5                    # Tolerance for iterative solution of wall temperatures
        self.niter = 100                     # Maximum number of iterations allowed for wall temperature solutions
        

        #---- Discretization (element sizes set in initialize)
        self.is_axisymmetric = False       # Axisymmetric tube?
        self.disc = settings.Discretization()
        
        self.cross_section = None


        #--- Transient model parameters
        self.time_step = 0.5        # Time set (seconds)
        self.soln_time = 60         # Solution time (seconds)
        self.Tf_initial_condition = []  # Inital condition for fluid T (K)  (z)
        self.Tw_initial_condition = []  # Initial condition for wall T (K)  (z,theta,r)
        
        #--- Stored view factor matricies
        self.view_factors = []                       # View factors between each circumferential element and the set of circumferential elements on the adjacent tube
        self.Kvfinv = []                             # Inverse of coefficient matrix used in IR exchange model
        self.Kvfinv_solar = [] 
        

        #---- Solution profiles
        self.clear_solution()

        return
    
    
    def clear_solution(self):
        # Profiles saved as a function of time (if transient).  Time is the last dimension for transient calculations
        self.Tf = []         # Fluid T (K) vs (z)
        self.Tw = []         # Wall T (K) vs. (z,theta,r)
        self.P = []          # Internal pressure (MPa) vs (z)
        self.qinc_nom = []   # Incident flux at the tube crown vs (z)
        self.max_thermal_stress_equiv_axial = [] # Maximum thermal stress in any cross section (z)
        self.thermal_stress_fraction_of_allowable = []  # Maximum thermal stress at axial position z divided by allowable stress
        self.pressure_stress_fraction_of_allowable = float('nan')  # Maximum pressure stress in the tube divided by allowable stress (evaluated at the maximum tube temperature)
        
        # Single values saved as a function of time (if transient). 
        self.Qfluid = float('nan')  # Total heat transfer into fluid (W)
        self.Qsinc = float('nan')   # Total incident solar energy (W)
        self.Qsabs = float('nan')   # Total absorbed solar energy (W)
        self.Qconv = float('nan')   # Total convection loss (W)
        self.Qrad = float('nan')    # Total IR radiation loss (W)       
        
        self.max_stress_equiv = float('nan')            # Maximum equivalent total stress
        self.max_thermal_stress_equiv = float('nan')    # Maximum equivalent thermal stress
        self.max_pressure_stress_equiv = float('nan')   # Maximum equivalent pressure stress
        
        #self.max_stress_intensity = float('nan')            # Maximum total stress intensity
        #self.max_thermal_stress_intensity = float('nan')    # Maximum thermal stress intensity
        #self.max_pressure_stress_intensity = float('nan')   # Maximum pressure stress intensity       
        
        
        # Profiles saved only at the most recently solved time point (if transient)
        self.htube = []      # Internal h (W/m2/K) vs. (z,theta)
        self.Re = []         # Re vs. (z)
        self.velocity = []   # velocity (m/s) vs. (z)
        self.qnet = []       # Net radiative heat flux (W/m2) into tube vs (z,theta)
        self.qnetIR = []
        self.qabs_solar = [] # Absorbed solar flux (W/m2) vs (z, theta)
        self.qinc_solar = [] # Incident solar flux (W/m2) vs. (z, theta)
        self.stress_equiv = [] # von Mises equivalent total stress vs (z, theta, r)
        self.stress_intensity = [] # Total stress intensity vs (z, theta, r)
        
        
        
        # Single values saved only at the most recently solved time point (if transient)
        self.tol = float('nan')     # Solution temperature tolerance (highest value over axial nodes)
        self.iter = float('nan')    # Solution iterations (highest value over axial nodes)
        self.converged = False      # Is solution converged for all nodes?

        return

 
    #-------------------------------------------------------------------------
    def initialize(self):
        
        if self.options.wall_detail == '0D':    # Reset inputs for 1 front point, 1 back point, average flux on front side
            self.disc.ntheta = 3
            self.flux_dist = 'step'
            #self.relative_flux_function = lambda theta: (2/np.pi) * np.heaviside(np.pi/2-theta, 0)   # 2/pi = average flux on front of tube, np.heaviside = step function
            self.options.use_full_rad_exchange = False
            
        if self.is_axisymmetric:
            self.options.wall_detail = '1D'
            self.disc.ntheta = 2
            self.flux_dist = 'constant'
            self.hext_dist = 'constant'
            #self.relative_flux_function = lambda theta: np.ones_like(theta)
            #self.relative_hext_function = lambda theta: np.ones_like(theta)
            self.options.use_full_rad_exchange = False
            self.options.is_adjacent_tubes = False

        self.ID = self.OD - 2*self.twall
        self.ro = 0.5*self.OD
        self.ri = self.ro - self.twall
        self.disc.update(self.OD, self.twall, self.length)
        
        self.fluid = materials.create_material(self.HTF_material_name)
        if self.tube_material_name != 'GenericConstantProp':
            self.tube_wall = materials.create_material(self.tube_material_name)
        else:
            self.tube_wall = materials.create_material(self.tube_material_name, self.tube_material_props)

        if self.htc_correlation in ['Sieder-Tate']:
            self.use_wall_visc_correction = True
            
            
        if not self.options.is_adjacent_tubes:
            self.options.use_full_rad_exchange = False
            
        #--- Calculate view factors between elements of adjacent tubes
        if self.options.is_adjacent_tubes:
            Nthetaq = int((self.disc.ntheta+1)/2)  # Number of elements on the front half of the tube
            self.view_factors = tube_view_factors.calculate_view_factors(self.ro, Nthetaq-1, self.disc.dz)  # Last row/column is view factor from/to ambient
            if self.options.use_full_rad_exchange:
                Kvf = radiosity.calculate_se_coeff_matrix(1.0-self.emis, self.view_factors)
                self.Kvfinv = np.linalg.inv(Kvf)
                Kvf_solar = radiosity.calculate_se_coeff_matrix(1.0-self.solar_abs, self.view_factors)
                self.Kvfinv_solar = np.linalg.inv(Kvf_solar)

        #--- Initialize solution temperature profiles
        self.Tf = np.zeros(self.disc.nz)
        self.Tw = np.zeros((self.disc.nz, self.disc.ntheta, self.disc.nr))
        

        #--- Initialize cross-section
        self.cross_section = cross_section.CrossSection(self.tube_wall, self.fluid, self.disc, self.options)
        

        return
    
    
    # Function describing incident flux distribution around the tube circumference
    def relative_flux_function(self, theta):
        if self.flux_dist == 'cosine':
            dist =np.cos(np.minimum(np.pi/2, np.abs(theta))) 
        elif self.flux_dist == 'step':   
            dist = (2/np.pi) * np.heaviside(np.pi/2-theta, 0)   # 2/pi = average flux on front of tube, np.heaviside = step function
        elif self.flux_dist == 'constant':
            dist = np.ones_like(theta)  
        return dist
    
    
    # Function describing incident flux distribution around the tube circumference
    def relative_hext_function(self, theta):
        if self.hext_dist == 'step':
            dist = np.heaviside(np.pi/2-theta, 0)  #(Step function with relative hext = 1 at front of tube, 0 at back of tube)
        elif self.hext_dist == 'constant':
            dist = np.ones_like(theta)   
        return dist
            
            
        
    

    #=========================================================================
    # Update inputs 
    def update_inputs(self, **kwargs):
        for k, val in kwargs.items():
            self.set_value(k, val)
        return
    
    def update_inputs_from_dict(self, D):
        for k, val in D.items():
            self.set_value(k, val)
        return    
    
    
    def get_value(self, name):
        if hasattr(self.operating_conditions, name):
            data = getattr(self.operating_conditions, name)
        elif hasattr(self.disc, name):
            data = getattr(self.disc, name)
        elif hasattr(self.options, name):
            data = getattr(self.options, name)  
        elif hasattr(self, name):
            data = getattr(self, name)    
        else:
            print(name + ' not recognized')
            data = None
        return data
    
    def set_value(self, name, val):
        if hasattr(self.operating_conditions, name):
            setattr(self.operating_conditions, name, val)
        elif hasattr(self.disc, name):
            setattr(self.disc, name, val)
        elif hasattr(self.options, name):
            setattr(self.options, name, val)            
        elif hasattr(self, name):
            setattr(self, name, val)    
        else:
            print(name + ' not recognized')
        return  
    

    #-------------------------------------------------------------------------
    # Friction factor and pressure drop    
    #--- Tube friction factor
    def friction_factor(self, Re, roughness = None):    
        if roughness is None:
            roughness = self.roughness  
        eD = roughness / self.ID  # Relative roughness
        ff = (-1.737*np.log(0.269*eD - 2.185/Re*np.log(0.269*eD+14.5/Re)))**-2  # Fanning friction factor
        fd = 4*ff  # Darcy friction factor
        return fd

    #--- Straight tube frictional pressure drop over length L
    def straight_pressure_drop(self, L, rho = None, visc = None, Tf = None):

        if rho is None and visc is None and Tf is not None:
            rho = self.fluid.density(Tf)
            visc = self.fluid.viscosity(Tf)
        else:
            print ('Error: No valid properties or conditions specified for tube pressure drop calacultion')
            return float('nan')           
        vel = self.mflow / rho /(np.pi*(self.ID/2.0)**2)   # Fluid velocity (m/s)
        if self.mflow < 1.e-6:
            return 0.0*vel

        Re = rho*vel*self.ID/visc         # Reynolds number       
        fd = self.friction_factor(Re)
        dp = 0.5*fd*rho*(vel**2)*(L/self.ID)   # Pressure drop (Pa)
        return dp
    
    #--- Calculate tube pressure profile using current fluid T solution
    def calculate_pressure_profile(self, outlet_pressure = 0.0, time_index = None):
        LDe90 = 30
        LDe45 = 15
        Tf = self.Tf if not self.options.is_transient else self.Tf[:,time_index]
        dens = self.fluid.density(Tf)
        dp_per_m = self.straight_pressure_drop(L = 1.0, Tf = Tf)  
        
        sign = 0
        if self.include_gravity:
            sign = 1 if self.flow_against_gravity else -1
            
        P = outlet_pressure*np.ones(self.disc.nz)
        for j in range(1, self.disc.nz):
            i = self.disc.nz-j
            P[i-1] = P[i] + self.disc.dz*(dp_per_m[i] + sign*dens[i]*9.8) * 1.e-6
            if j == 1 or j == self.disc.nz-1:  # Apply pressure drop from tube bends
                P[i-1] += dp_per_m[i]*self.ID*(LDe90*self.tube_bends_90/2 + LDe45*self.tube_bends_45/2)  * 1.e-6
        
        if time_index is None:
            self.P = P  # MPa
        else:
            self.P[:,time_index] = P #MPa
        return 
    
    #-------------------------------------------------------------------------
    #--- Tube internal convective heat transfer coefficient: 
    # Tf = fluid temperature (single value or array) 
    # wall_visc = include near-wall viscosity correction
    # Tw = wall temperature (single value or array).  If a 2D array, the first dimension must match the dimension of Tf
    # z = axial coordinate relative to flow inlet for entrance length corrections (must be same length as Tf if both are provided as arrays)
    def internal_h(self, Tf, wall_visc = False, Tw = None, fluid_properties = None, z = None, allow_laminar = True, show_warnings = True):
        
        isTfarray = hasattr(Tf, '__len__')
        isTwarray = hasattr(Tw, '__len__')
        isTw2D = False if not isTwarray else hasattr(Tw[0],'__len__')
        
        if self.mflow < 1.e-3:
            h = 0.0 * Tf
            if (not isTfarray and isTwarray) or isTw2D:
                h = np.zeros_like(Tw)
            return h, 0.0*Tf, 0.0*Tf
       
        
        if fluid_properties is None:
            rho = self.fluid.density(Tf)
            cp = self.fluid.cp(Tf)
            k = self.fluid.k(Tf)
            visc = self.fluid.viscosity(Tf)
        else:
            rho = fluid_properties['density']
            cp = fluid_properties['cp']
            k = fluid_properties['k']
            visc = fluid_properties['viscosity']
        
        Pr = cp*visc/k  
        vel = self.mflow / rho /(np.pi*(self.ri**2))   
        Re = rho*vel*self.ID/visc    
        Re_min = Re if not isTfarray else Re.min()
        Re_max = Re if not isTfarray else Re.max()        
    
        Re_limits = {'Colburn': [1e4, 1e10], 'Dittus-Boelter':[1e4, 1e10], 'Sieder-Tate': [1e4, 1e10],
                     'Gnielinski-smooth': [3000, 5e6], 'Gnielinski-rough': [3000, 5e6], 'Petukhov-smooth': [1e4, 5e6], 'Petukhov-rough': [1e4, 5e6], 
                     'Skupinski': [3.6e3, 9.05e5], 'Chen': [0, 1e10]}

        
        #--- Apply laminar flow correlation if required
        if Re_min < 2300:      
            Gz = Re*Pr*self.ID/z
            Nu = 3.66+((0.049+0.02/Pr)*Gz**1.12)/(1+0.065*Gz**0.7)  
            #if Re_min<2300:
                #print ('Warning: Heat transfer correlation for laminar flow being using in place of specified correlation') 
                
        #--- Turbulent flow correlations
        else:
            if self.htc_correlation == 'Colburn':
                Nu = 0.023*(Re**0.8)*Pr**(1./3.)
            elif self.htc_correlation == 'Dittus-Boelter':
                Nu = 0.023*(Re**0.8)*Pr**0.4
            elif self.htc_correlation == 'Sieder-Tate':
                Nu = 0.027 * (Re**0.8) * (Pr**(1./3.))
            elif self.htc_correlation in ['Gnielinski-smooth', 'Gnielinski-rough']:
                use_roughness = 0.0 if self.htc_correlation == 'Gnielinski-smooth' else self.roughness           
                fd = self.friction_factor(Re, use_roughness)
                Nu = (fd/8.)*(Re-1000)*Pr / (1+12.7*((fd/8.)**0.5)*(Pr**(2./3.)-1.0))     
            elif self.htc_correlation in ['Petukhov-smooth', 'Petukhov-rough']:
                use_roughness = 0.0 if self.htc_correlation == 'Petukhov-smooth' else self.roughness     
                fd = self.friction_factor(Re, use_roughness)
                Nu = (fd/8)*Re*Pr / (1.07+12.7*((fd/8.)**0.5)*(Pr**(2./3.)-1.0))    
            elif self.htc_correlation == 'Skupinski':    # Low Pr HTF's (e.g. liquid metals)
                Nu = 4.82 + 0.0185*(Re*Pr)**0.827       
            elif self.htc_correlation == 'Chen': # Chen and Chiou: https://doi.org/10.1016/0017-9310(81)90167-8.  Used in Will Logie's sodium models
                Nu = 5.6 + 0.0165*((Re*Pr)**0.85)*(Pr**0.01)
            else:
                print ('Heat transfer correlation %s not recognized'% self.htc_correlation)

            #--- Entry length correction
            if self.options.entry_length_correction:
                Nu *= (1 + (self.ID/z)**0.7)  
    
            #--- Include Nu correction for wall T (based on Sieder-Tate correlation)
            if wall_visc: #self.use_wall_visc_correction:
                visc_wall = self.fluid.viscosity(Tw) # Fluid viscosity at wall temperature
                if isTfarray and isTw2D:
                    n = len(Tf)
                    Nu = np.reshape(Nu, (n,1)) * ((np.reshape(visc, (n,1))/visc_wall)**0.14) 
                else:
                    Nu *= (visc/visc_wall)**0.14  
        
            if show_warnings:
                if Re_min<Re_limits[self.htc_correlation][0] or Re_max>Re_limits[self.htc_correlation][1]:
                    print ('Warning: Re out of range for %s correlation' %(self.htc_correlation))      
        
        if isTfarray and isTw2D:
            n = len(Tf)
            h = Nu*(np.reshape(k, (n,1))/self.ID) 
        else:
            h = Nu*k/self.ID
        
        return h, Re, vel
    
    def apply_near_wall_viscosity_correction(self, h, Tw, visc = None, Tf = None):
        if visc is None:
            visc = self.fluid.viscosity(Tf)
        visc_wall = self.fluid.viscosity(Tw)
        
        isTfarray = hasattr(Tf, '__len__')
        isTwarray = hasattr(Tw, '__len__')
        isTw2D = False if not isTwarray else hasattr(Tw[0],'__len__')
        if isTfarray and isTw2D:
            n = len(Tf)
            h = np.reshape(h, (n,1)) * ((np.reshape(visc, (n,1))/visc_wall)**0.14) 
        else:
            h *= (visc/visc_wall)**0.14     
        return h
        

    #=========================================================================
    # Calculate axial/circumferential distributions of incident/absorbed solar flux
    def calculate_tube_solar_flux_distributions(self):
        if not self.is_solar:
            return np.zeros((self.disc.nz, self.disc.ntheta)), np.zeros((self.disc.nz, self.disc.ntheta)), 0.0, 0.0
        
        scale = self.relative_flux_function(self.disc.thetapts)                      # Scaling factors for incident flux at each theta point relative to value at theta = 0
        qinc = self.operating_conditions.inc_flux.reshape(-1,1) * scale * 1000  # Reshape incident flux per height position into a single-column array, scale to each theta position, and scale to W/m2
        qabs = qinc * self.solar_abs                                            # Generic definition of absorbed flux not accounting for reflection between adjacent tubes
        if self.options.use_full_rad_exchange:  # Include surface exchange model between adjacent tubes to account for diffuse reflection
            nthetaq = int((self.disc.ntheta+1)/2)
            qnet = radiosity.solve_radiosity_abs_flux(scale[0:nthetaq-1], (1.0-self.solar_abs), self.view_factors, Kinv = self.Kvfinv_solar)  # Net energy leaving each element [W/m2]   Note, last element is energy leaving the aperture
            abs_dist = scale[0:nthetaq-1] - qnet[0:-1]                                      # Distribution of "absorbed" energy at tube circumferential positions per 1W/m2 incident energy  (incident energy - net energy leaving element after exchange with adjacent tube)
            qabs = self.operating_conditions.inc_flux.reshape(-1,1) * abs_dist * 1000       # Reshape inc_flux per height position into a single-column array, scale to each theta position, and scale to W/m2
            qabs = np.append(qabs, np.zeros((self.disc.nz,self.disc.ntheta-(nthetaq-1))), axis = 1)   # Append zeros to define absorbed flux array including back of tube


        #--- Calculate total incident and absorbed solar energy
        elem_area_outer = self.disc.dz * (2*self.ro*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)
        Qsinc = qinc[0:self.disc.nz-1,:].sum() * elem_area_outer  # Incident solar energy (W)
        Qsabs = qabs[0:self.disc.nz-1,:].sum() * elem_area_outer  # Absorbed solar energy (W)           
        return qinc, qabs, Qsinc, Qsabs
    

    #=========================================================================
    # Calculate net radiative flux (solar - net emission loss): 
    # qsabs = solar flux distribution (theta),
    # Twall = external wall temperature (theta),
    def calculate_net_radiative_flux_distribution(self, qsabs, Twall):   
        qnet, qnetIR = [np.zeros(self.disc.ntheta) for v in range(2)]
        qnetIR_aperture = 0.0
        if self.options.is_adjacent_tubes:
            nthetaq = int((self.disc.ntheta+1)/2)  # Number of nodes on front-side of tube
            if self.options.use_full_rad_exchange:
                qnetsoln = radiosity.solve_radiosity(Twall[0:nthetaq-1], self.operating_conditions.Tambrad, 1.0-self.emis, self.view_factors, Kinv = self.Kvfinv)  # Net IR energy leaving each element. Note last element is net IR energy incident on aperture  
                qnetIR[0:nthetaq-1] = qnetsoln[0:nthetaq-1]
                qnetIR_aperture = qnetsoln[-1]
            else:
                vf_to_amb = np.zeros(self.disc.ntheta)
                vf_to_amb[0:nthetaq] = self.view_factors[0:nthetaq,-1]  # View factor to ambient for front tubes surfaces
                qnetIR = vf_to_amb * self.emis * 5.6704e-8 * (Twall**4 - self.operating_conditions.Tambrad**4)   # Net IR energy leaving each element
        else:
            qnetIR = self.emis * 5.6704e-8 * (Twall**4 - self.operating_conditions.Tambrad**4)

        qnet = qsabs - qnetIR  # Local radiation heat flux into tube (solar absorbed - net IR emitted) 
        
        #--- Calculate radiative loss (W) for this element
        if self.options.use_full_rad_exchange: 
            QIR = -qnetIR_aperture * (2*self.ro*self.disc.dz)    # Radiative loss [W] from net energy leaving aperture. Factor of 2 accounts for symmetry
        else:
            elem_area_outer = self.disc.dz * (2*self.ro*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)
            QIR = elem_area_outer * qnetIR[0:-1].sum()

        return qnet, qnetIR, QIR
        

    #=========================================================================
    # Solve for steady state tube fluid/wall temperatures at steady state using the current conditions and a fixed mass flow rate
    def solve_for_steady_state_T_profiles(self, use_soln_as_guess = True, Ttol = None, verbosity = 0):
        nz = self.disc.nz
        ntheta = self.disc.ntheta
        nthetaq = int((ntheta+1)/2)
        nr = self.disc.nr
        dz = self.disc.dz
        thetapts = self.disc.thetapts   
        elem_area_outer = self.disc.dz * (2*self.ro*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)
        elem_area_inner = self.disc.dz * (2*self.ri*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)

        #--- Set external heat transfer coefficient for each theta point
        hext = self.hext * self.relative_hext_function(thetapts)
        
        #--- Set incident and absorbed solar flux for each circumferential (theta) point
        qinc, qabs, Qsinc, Qsabs = self.calculate_tube_solar_flux_distributions()
           
        #--- Intialize solution quantities
        Tf = self.operating_conditions.Tfin * np.ones(nz)
        Twall = np.zeros((nz,ntheta,nr))
        qnet, qnetIR, htube = [np.zeros((nz, ntheta)) for v in range(3)]
        Re, vel, tolT, tolTf, solution_iter = [np.zeros(nz) for v in range(5)] 
        is_converged = np.zeros(nz, dtype = bool)       
        Qtofluid, Qradloss, Qconvloss = [0 for v in range(3)]

        Ttol = self.Ttol if Ttol is None else Ttol
        use_soln_as_guess = use_soln_as_guess and len(self.Tw) > 0 and self.Tw.max() > 0  # Only use current solution as guess if one exists
        if use_soln_as_guess:  
            Twall = self.Tw

        #--- Update properties and heat losses for all axial positions 
        for z in range(nz):
            zpos = (z+0.5)*dz if not self.is_fully_developed else 1.e10
            
            # Calculate internal heat transfer coefficient without near-wall viscosity correction (Tf[z] is fixed during the subsequent iterations)
            fluid_properties = {'density': self.fluid.density(Tf[z]), 'cp':self.fluid.cp(Tf[z]), 'k': self.fluid.k(Tf[z]), 'viscosity':self.fluid.viscosity(Tf[z])}
            hz, Re[z], vel[z] = self.internal_h(Tf[z], wall_visc = False, z = zpos, fluid_properties = fluid_properties, show_warnings = verbosity>1)       # could vectorize

            # Set initial guess for wall temperature at this position (based on solution with linearized radiative loss)
            if not use_soln_as_guess:
                Twall[z,:,:] = Twall[z-1,:,:] + (Tf[z] - Tf[z-1]) if z > 0 else Tf[z] + 50*self.is_solar  # Initial guess
                if self.is_solar:  # Improve initial guess using solution with radiative loss linearized around initial guess for Twall
                    fullvf = np.ones(ntheta) if not self.options.is_adjacent_tubes else np.append(self.view_factors[0:nthetaq,-1], np.zeros(ntheta-nthetaq))  # Full array of view factors for tube
                    htube[z,:] = hz if not self.use_wall_visc_correction else self.apply_near_wall_viscosity_correction(hz, Twall[z,:,0], fluid_properties['viscosity'])
                    Tlin = Twall[z,:,-1]  # Linearization temperature at external wall
                    qext = qabs[z,:] - fullvf*self.emis*5.6704e-8*(Tlin**4-self.operating_conditions.Tambrad**4) + 4*fullvf*self.emis*5.6704e-8*Tlin**3*(Tlin-self.operating_conditions.Tamb)
                    self.cross_section.conditions.populate(Twall[z,:,:], Tf[z], qext, htube[z,:], hext+4*fullvf*self.emis*5.6704e-8*Tlin**3, self.operating_conditions.Tamb, self.mflow)
                    Tnew = self.cross_section.solve_wall_cross_section_steady_state()
                    Twall[z,:,:] = Tnew


            converged = False
            i = 0            
            while not converged and i<=self.niter:  # Iterations to converge wall tempeature solution, IR loss, and temperature-dependent properties and heat-transfer coefficients
                htube[z,:] = hz if not self.use_wall_visc_correction else self.apply_near_wall_viscosity_correction(hz, Twall[z,:,0], fluid_properties['viscosity'])  # Update heat transfer coefficient using near-wall viscosity correction
                qnet[z,:], qnetIR[z,:], QIR_at_z = self.calculate_net_radiative_flux_distribution(qabs[z,:], Twall[z,:,-1])  # Calculate net absorbed radiative flux (solar - IR loss)
                self.cross_section.conditions.populate(Twall[z,:,:], Tf[z], qnet[z,:], htube[z,:], hext, self.operating_conditions.Tamb, self.mflow)
                Tnew = self.cross_section.solve_wall_cross_section_steady_state()    # New wall temperature solution            
                tolT[z] = (np.abs(Tnew - Twall[z,:,:]) / Tnew).max()    # Difference in wall temperature from previous iteration
                
                #--- Update wall temperature
                urf = 1.0 if tolT[z] < 0.1 else (0.4 if tolT[z]<1.0 else 0.1)  # Use a low relaxation factor if solution has large changes from previous guess
                #urf = 1.0 if tolT[z] < 1e-4 else (0.4 if tolT[z]<1e-3 else 0.1)  # Use a low relaxation factor if solution has large changes from previous guess
                
                if Re[z] < 2300:   # Limit urf for laminar flow conditions
                    urf = min(urf, 0.3)
                Twall[z,:,:] = urf*Tnew + (1.-urf)*Twall[z,:,:]
                
                #--- Check convergence*700
                if (tolT[z] < Ttol):
                    converged = True
                    solution_iter[z] = i+1
                    if verbosity > 1:
                        print ('Axial node %d, Iterations = %d, Peak T = %.2fC, Tolerance = %.1e'%(z, i+1, Twall[z,:,:].max()-273.15, tolT[z]))
                i+=1
                
            if not converged and self.niter >10:
                print ('Axial node %d, failed to converge after %d iterations: Peak T = %.2fC, Tolerance = %.1e'%(z, i, Twall[z,:,:].max()-273.15, tolT[z]))
                break  # Stop solution of remaining axial nodes when one failes to converge

            is_converged[z] = converged   

            #--- Update fluid temperature at next node and calculate total heat transfer rate to fluid
            if z < nz-1:
                Qfluid_at_z = elem_area_inner * (htube[z,0:-1] * (Twall[z,0:-1,0]-Tf[z])).sum()  # Heat transfer to fluid from this tube segment [W]
                Tf[z+1] = Tf[z] + Qfluid_at_z / self.mflow / fluid_properties['cp']
                
                #Qtofluid += Qfluid_at_z
                Qradloss += QIR_at_z
                #Qconvloss +=  elem_area_outer * (hext*(Twall[z,:,-1] - self.operating_conditions.Tamb)).sum()

        Qtofluid = elem_area_inner * (htube[:-1,:-1] * ( Twall[:-1,:-1,0]-np.reshape(Tf[:-1], (nz-1,1)) ) ).sum()  
        Qconvloss = elem_area_outer * (np.reshape(hext[:-1], (1,ntheta-1)) * (Twall[:-1,:-1,-1] - self.operating_conditions.Tamb)).sum() 

        #--- Store solution
        self.Tf = Tf       
        self.Tw = Twall       
        self.htube = htube
        self.Re = Re
        self.velocity = vel
        self.qnet = qnet
        self.qnetIR = qnetIR
        self.qabs_solar = qabs
        self.qinc_solar = qinc
        self.qinc_nom = self.operating_conditions.inc_flux

        self.Qfluid = Qtofluid 
        self.Qsinc = Qsinc
        self.Qsabs = Qsabs
        self.Qconv = Qconvloss
        self.Qrad = Qradloss
        self.tol = tolT.max()
        self.iter = solution_iter.max()
        self.converged = is_converged.min()


        return 


    #=========================================================================
    # Solve for steady state tube fluid/wall temperatures at steady state using the current conditions and a fixed mass flow rate
    def solve_for_steady_state_T_profiles_jwenner(self, use_soln_as_guess = True, Ttol = None, verbosity = 0):
        nz = self.disc.nz
        ntheta = self.disc.ntheta
        nthetaq = int((ntheta+1)/2)
        nr = self.disc.nr
        dz = self.disc.dz
        thetapts = self.disc.thetapts   
        elem_area_outer = self.disc.dz * (2*self.ro*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)
        elem_area_inner = self.disc.dz * (2*self.ri*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)

        #--- Set external heat transfer coefficient for each theta point
        hext = self.hext * self.relative_hext_function(thetapts)
        
        #--- Set incident and absorbed solar flux for each circumferential (theta) point
        qinc, qabs, Qsinc, Qsabs = self.calculate_tube_solar_flux_distributions()
           
        #--- Intialize solution quantities
        Tf = self.operating_conditions.Tfin * np.ones(nz)
        Twall = np.zeros((nz,ntheta,nr))
        qnet, qnetIR, htube = [np.zeros((nz, ntheta)) for v in range(3)]
        Re, vel, tolT, tolTf, solution_iter = [np.zeros(nz) for v in range(5)] 
        is_converged = np.zeros(nz, dtype = bool)       
        Qtofluid, Qradloss, Qconvloss = [0 for v in range(3)]

        Ttol = self.Ttol if Ttol is None else Ttol
        use_soln_as_guess = use_soln_as_guess and len(self.Tw) > 0 and self.Tw.max() > 0  # Only use current solution as guess if one exists
        if use_soln_as_guess:  
            Twall = self.Tw

        #--- Update properties and heat losses for all axial positions 
        for z in range(nz):
            zpos = (z+0.5)*dz if not self.is_fully_developed else 1.e10
            
            # Calculate internal heat transfer coefficient without near-wall viscosity correction (Tf[z] is fixed during the subsequent iterations)
            fluid_properties = {'density': self.fluid.density(Tf[z]), 'cp':self.fluid.cp(Tf[z]), 'k': self.fluid.k(Tf[z]), 'viscosity':self.fluid.viscosity(Tf[z])}
            hz, Re[z], vel[z] = self.internal_h(Tf[z], wall_visc = False, z = zpos, fluid_properties = fluid_properties, show_warnings = verbosity>1)       # could vectorize

            # Set initial guess for wall temperature at this position (based on solution with linearized radiative loss)
            if not use_soln_as_guess:
                Twall[z,:,:] = Twall[z-1,:,:] + (Tf[z] - Tf[z-1]) if z > 0 else Tf[z] + 50*self.is_solar  # Initial guess
                if self.is_solar:  # Improve initial guess using solution with radiative loss linearized around initial guess for Twall
                    fullvf = np.ones(ntheta) if not self.options.is_adjacent_tubes else np.append(self.view_factors[0:nthetaq,-1], np.zeros(ntheta-nthetaq))  # Full array of view factors for tube
                    htube[z,:] = hz if not self.use_wall_visc_correction else self.apply_near_wall_viscosity_correction(hz, Twall[z,:,0], fluid_properties['viscosity'])
                    Tlin = Twall[z,:,-1]  # Linearization temperature at external wall
                    qext = qabs[z,:] - fullvf*self.emis*5.6704e-8*(Tlin**4-self.operating_conditions.Tambrad**4) + 4*fullvf*self.emis*5.6704e-8*Tlin**3*(Tlin-self.operating_conditions.Tamb)
                    self.cross_section.conditions.populate(Twall[z,:,:], Tf[z], qext, htube[z,:], hext+4*fullvf*self.emis*5.6704e-8*Tlin**3, self.operating_conditions.Tamb, self.mflow)
                    Tnew = self.cross_section.solve_wall_cross_section_steady_state()
                    Twall[z,:,:] = Tnew


            converged = False
            i = 0            
            while not converged and i<=self.niter:  # Iterations to converge wall tempeature solution, IR loss, and temperature-dependent properties and heat-transfer coefficients
                htube[z,:] = hz if not self.use_wall_visc_correction else self.apply_near_wall_viscosity_correction(hz, Twall[z,:,0], fluid_properties['viscosity'])  # Update heat transfer coefficient using near-wall viscosity correction
                qnet[z,:], qnetIR[z,:], QIR_at_z = self.calculate_net_radiative_flux_distribution(qabs[z,:], Twall[z,:,-1])  # Calculate net absorbed radiative flux (solar - IR loss)
                self.cross_section.conditions.populate(Twall[z,:,:], Tf[z], qnet[z,:], htube[z,:], hext, self.operating_conditions.Tamb, self.mflow)
                Tnew = self.cross_section.solve_wall_cross_section_steady_state()    # New wall temperature solution            
                tolT[z] = (np.abs(Tnew - Twall[z,:,:]) / Tnew).max()    # Difference in wall temperature from previous iteration
                
                #--- Update wall temperature
                urf = 1.0 if tolT[z] < 0.1 else (0.4 if tolT[z]<1.0 else 0.1)  # Use a low relaxation factor if solution has large changes from previous guess
                #urf = 1.0 if tolT[z] < 1e-4 else (0.4 if tolT[z]<1e-3 else 0.1)  # Use a low relaxation factor if solution has large changes from previous guess
                
                if Re[z] < 2300:   # Limit urf for laminar flow conditions
                    urf = min(urf, 0.3)
                Twall[z,:,:] = urf*Tnew + (1.-urf)*Twall[z,:,:]
                
                #--- Check convergence*700
                if (tolT[z] < Ttol):
                    converged = True
                    solution_iter[z] = i+1
                    if verbosity > 1:
                        print ('Axial node %d, Iterations = %d, Peak T = %.2fC, Tolerance = %.1e'%(z, i+1, Twall[z,:,:].max()-273.15, tolT[z]))
                i+=1
                
            if not converged and self.niter >10:
                print ('Axial node %d, failed to converge after %d iterations: Peak T = %.2fC, Tolerance = %.1e'%(z, i, Twall[z,:,:].max()-273.15, tolT[z]))
                break  # Stop solution of remaining axial nodes when one failes to converge

            is_converged[z] = converged   

            #--- Update fluid temperature at next node and calculate total heat transfer rate to fluid
            if z < nz-1:
                Qfluid_at_z = elem_area_inner * (htube[z,0:-1] * (Twall[z,0:-1,0]-Tf[z])).sum()  # Heat transfer to fluid from this tube segment [W]
                Tf[z+1] = Tf[z] + Qfluid_at_z / self.mflow / fluid_properties['cp']
                
                #Qtofluid += Qfluid_at_z
                Qradloss += QIR_at_z
                #Qconvloss +=  elem_area_outer * (hext*(Twall[z,:,-1] - self.operating_conditions.Tamb)).sum()

        Qtofluid = elem_area_inner * (htube[:-1,:-1] * ( Twall[:-1,:-1,0]-np.reshape(Tf[:-1], (nz-1,1)) ) ).sum()  
        Qconvloss = elem_area_outer * (np.reshape(hext[:-1], (1,ntheta-1)) * (Twall[:-1,:-1,-1] - self.operating_conditions.Tamb)).sum() 

        #--- Store solution
        self.Tf = Tf       
        self.Tw = Twall       
        self.htube = htube
        self.Re = Re
        self.velocity = vel
        self.qnet = qnet
        self.qnetIR = qnetIR
        self.qabs_solar = qabs
        self.qinc_solar = qinc
        self.qinc_nom = self.operating_conditions.inc_flux

        self.Qfluid = Qtofluid 
        self.Qsinc = Qsinc
        self.Qsabs = Qsabs
        self.Qconv = Qconvloss
        self.Qrad = Qradloss
        self.tol = tolT.max()
        self.iter = solution_iter.max()
        self.converged = is_converged.min()


        return



    #=========================================================================
    # Solve for steady state tube fluid/wall temperatures using the current conditions, including iteration to set mass flow if user-specified tube outlet temperature is provided
    def solve_steady_state(self, mflow_guess = None, calculate_stress = False, outlet_pressure = 0.0, verbosity = 0):

        niter = 1
        #--- Set initial guess for mass flow rate if this is an iterative solution
        if self.specified_Tfout is not None:
            niter = self.niter
            Tpts = np.linspace(self.operating_conditions.Tfin, self.specified_Tfout, 100)
            HTFcpavg = np.trapz(self.fluid.cp(Tpts), Tpts) / (self.specified_Tfout - self.operating_conditions.Tfin)  
            self.mflow = (0.88 * self.operating_condtions.inc_flux.mean() * 1000 * self.OD * self.length) / HTFcpavg / (self.specified_Tfout - self.operating_conditions.Tfin) if mflow_guess is None else mflow_guess
            
        mtol = 1.0
        Ttol_target = self.Ttol
        for m in range(niter):
            Ttol = 1.e-3 if (niter>1 and (m == 0 or mtol > 0.05)) else min(Ttol_target, 0.1*mtol)   # Use low temperature tolerance for bad mflow guesses but, ensure temperature tolerance is better than mass flow tolerance to avoid oscillatory solutions
            self.solve_for_steady_state_T_profiles(True, Ttol, verbosity)

            if niter>1:
                if verbosity > 0:
                    print ('Iteration %d: Mass flow = %.2f kg/s, Qfluid = %.3f W, Outlet T = %.2f C'%(m, self.mflow, self.Qfluid, self.Tf[-1]-273.15))              
                
                T_err = (self.Tf[-1] - self.specified_Tfout)/self.specified_Tfout
                if abs(T_err) < self.Ttol:  # Mass flow is converged 
                    print ('Mass flow solution converged in %d iterations'%(m+1))
                    break
                elif m == self.niter-1:
                    print ('Mass flow solution did not converge')
                    break                    
                
                mflow_new = self.Qfluid/ HTFcpavg / (self.specified_Tfout-self.operating_conditions.Tfin)  
                mtol = (abs((self.mflow - mflow_new)/mflow_new)).max()
                urf = 1.0 if mtol <0.03 else 0.8
                self.mflow = urf*mflow_new + (1.0-urf)*self.mflow
                
        if calculate_stress:
            self.calculate_pressure_profile(outlet_pressure)
            self.calculate_stress()         
            self.calculate_thermal_stress_relative_to_allowable()
            self.calculate_pressure_stress_relative_to_allowable()
        return
    
    
    #=========================================================================
    def initialize_transient_results(self):
        nstep = int(self.soln_time / self.time_step)  
        
        #--- Intialize solution quantities to be stored vs time
        self.time = np.arange(nstep) * self.time_step
        self.Tf = np.zeros((self.disc.nz, nstep))
        self.P = np.zeros((self.disc.nz, nstep))
        if self.is_solar:
            self.qinc_nom = np.zeros((self.disc.nz, nstep))
        self.Tw = np.zeros((self.disc.nz,self.disc.ntheta,self.disc.nr,nstep))
        self.max_thermal_stress_equiv_axial = np.zeros((self.disc.nz, nstep))
        for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qrad', 'Qconv', 
                  'max_stress_equiv', 'max_thermal_stress_equiv', 'max_pressure_stress_equiv',
                  'max_stress_intensity', 'max_thermal_stress_intensity', 'max_pressure_stress_intensity']:
            setattr(self, k, np.zeros(nstep))

        #--- Intialize solution quantities to be stored only at most recent time point
        self.qnet, self.qnetIR, self.htube, self.hamb = [np.zeros((self.disc.nz, self.disc.ntheta)) for v in range(4)]
        self.Re, self.vel = [np.zeros(self.disc.nz) for v in range(2)] 
        
        #--- Set IC
        self.Tf[:,0] = self.Tf_initial_condition   
        self.Tw[:,:,:,0] = self.Tw_initial_condition   
        return
        
            
    
    #=========================================================================
    def solve_time_point(self, time, update_operating_conditions = False, calculate_stress = False, outlet_pressure = 0.0, verbosity = 0):
        '''
        Solve for solution at next time point 
        '''
        elem_area_outer = self.disc.dz * (2*self.ro*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)
        elem_area_inner = self.disc.dz * (2*self.ri*self.disc.dtheta)   # tube element outer wall surface area (m2). Note dtheta defined for half-tube (pi/Ntheta-1)

        t = round(time/self.time_step)
        if update_operating_conditions:  # Update operating_conditions from trans_operating conditions
            self.operating_conditions = self.trans_operating_conditions.get_operating_conditions_at_time(time) # Update self.operating_conditions at this point in time   
            self.mflow = self.operating_conditions.mass_flow
            hext = self.hext if 'hext' not in vars(self.operating_conditions).keys() else self.operating_conditions.hext  # Use hext in operating_conditions if it exists
        else: # Assume operating_conditions are already populated
            hext = self.hext
            

        hext = hext * self.relative_hext_function(self.disc.thetapts)                   # Update external heat transfer coefficient  
        self.qinc_solar, self.qabs_solar, self.Qsinc[t], self.Qsabs[t] = self.calculate_tube_solar_flux_distributions()  # Set incident/absorbed flux distributions (used inc_flux in operating_conditions)

        #--- Calculate properties and internal heat transfer coefficients at time t (faster to evaluate all at once than per z position)
        wall_properties = {'density': self.tube_wall.density(self.Tw[:,:,:,t]), 'cp':self.tube_wall.cp(self.Tw[:,:,:,t]), 'k': self.tube_wall.k(self.Tw[:,:,:,t])}
        fluid_properties = {'density': self.fluid.density(self.Tf[:,t]), 'cp':self.fluid.cp(self.Tf[:,t]), 'k': self.fluid.k(self.Tf[:,t]), 'viscosity':self.fluid.viscosity(self.Tf[:,t])}
        zpts = (self.disc.zpts + 0.5*self.disc.dz) if not self.is_fully_developed else  1e10*np.ones(self.disc.nz)
        self.htube[:,:], self.Re, self.vel = self.internal_h(self.Tf[:,t], self.use_wall_visc_correction, self.Tw[:,:,0,t], fluid_properties, z = zpts, show_warnings = verbosity>1)     # Update internal h

        #--- Loop over axial positions
        for z in range(self.disc.nz):
            wall_props = {k:wall_properties[k][z,:,:] for k in wall_properties.keys()}  # Wall properties at this cross-section
            fluid_props = {k:fluid_properties[k][z] for k in fluid_properties.keys()}  # fluid properties at this cross-section
            self.qnet[z,:], self.qnetIR[z,:], QIR_at_z = self.calculate_net_radiative_flux_distribution(self.qabs_solar[z,:], self.Tw[z,:,-1,t])  # Calculate net absorbed radiative flux (solar - IR loss)

            # Solve for wall and fluid T
            Tfin = None if z>0 else self.operating_conditions.Tfin
            Tf_prevz = None if z == 0 else self.Tf[z-1,t+1]  # Fluid temperature solution at prevous z position and current time point
            Tf_prevzt = None if z == 0 else self.Tf[z-1,t]   # Fluid temperature at previous z position and previous time point
            self.cross_section.conditions.populate(self.Tw[z,:,:,t], self.Tf[z,t], self.qnet[z,:], self.htube[z,:], hext, self.operating_conditions.Tamb, self.mflow, self.time_step, Tf_prevz, Tf_prevzt)
            Twsoln, Tfsoln = self.cross_section.solve_cross_section_at_time_point(wall_props, fluid_props, Tfin)    # New wall temperature solution        
            self.Tw[z,:,:,t+1] = Twsoln
            self.Tf[z,t+1] = Tfsoln
            
            #print('time = %d, z = %d, Tmax = %.2fC, Tmin = %.2fC'%(t, z, Twsoln.max()-273.15, Twsoln.min()-273.15))

            #--- Collect overall heat transfer rates
            if z<self.disc.nz-1:
                self.Qrad[t] += QIR_at_z     # Radiative loss from this tube segment [W] 
                #self.Qfluid[t] += elem_area_inner * (self.htube[z,0:-1] * (self.Tw[z,0:-1,0,t]-self.Tf[z,t])).sum()       # Heat transfer to fluid from this tube segment [W]                                                                           
                #self.Qconv[t] +=  elem_area_outer * (hext[:-1] * (self.Tw[z,:-1,-1,t] - self.operating_conditions.Tamb)).sum()   # Convective loss from this tube segment [W]  

        self.Qfluid[t] = elem_area_inner * (self.htube[:-1,:-1] * ( self.Tw[:-1,:-1,0,t]-np.reshape(self.Tf[:-1,t], (self.disc.nz-1,1)) ) ).sum()  
        self.Qconv[t] = elem_area_outer * (np.reshape(hext[:-1], (1,self.disc.ntheta-1)) * (self.Tw[:-1,:-1,-1,t] - self.operating_conditions.Tamb)).sum() 
        self.calculate_pressure_profile(outlet_pressure, t)
        if self.is_solar:
            self.qinc_nom[:,t] = self.operating_conditions.inc_flux

        return 
        

    #=========================================================================
    def solve_transient(self, initialize_soln = True, calculate_stress = False, outlet_pressure = 0.0, verbosity = 0):
        nstep = int(self.soln_time / self.time_step)  
        nper10 = int(0.1*self.soln_time / self.time_step)
        if initialize_soln:
            self.initialize_transient_results()
            self.Tf[:,0] = self.Tf_initial_condition  # Fluid IC
            self.Tw[:,:,:,0] = self.Tw_initial_condition     # Wall IC
        for t in range(nstep-1):
            time = t*self.time_step     # Current time relative to first point in self.trans_operating_conditions
            #if t%nper10 == 0:
            #    print('%d %% complete'%((time/self.soln_time)*100))
            self.solve_time_point(time, True, calculate_stress, outlet_pressure, verbosity)
            if calculate_stress:
                self.calculate_pressure_profile(outlet_pressure, t)

        if calculate_stress:   
            self.calculate_total_stress()            
        
        return
     
     
    #=========================================================================
    def set_result(self, name, val, time_index = None):
        if self.options.is_transient:
            attr = getattr(self, name)
            attr[...,time_index] = val
        else:
            setattr(self, name, val)
        return     
    
    def get_result(self, name, time_index = None):
        if time_index is not None:
            return getattr(self, name)[...,time_index]
        else:
            return getattr(self, name)
        return          
 

    #=========================================================================
    # Total stress [MPa].  Note, pressure stress calculation uses pressure profile currently stored in self.P
    def calculate_stress(self, use_property_temperature_variation = True):
        stress = elastic_stress.TubeElasticStress(self.ri, self.ro, self.tube_material_name, self.tube_material_props)
        thermal_equiv, pressure_equiv, self.stress_equiv = stress.calculate_total_stress(self.P, self.Tw, use_property_temperature_variation)
        self.max_thermal_stress_equiv = thermal_equiv.max(0).max(0).max(0)  # Maximum equivalent thermal stress over z, theta, r
        self.max_pressure_stress_equiv = pressure_equiv.max(0).max(0).max(0) 
        self.max_stress_equiv = self.stress_equiv.max(0).max(0).max(0) 
        self.max_thermal_stress_equiv_axial = thermal_equiv.max(1).max(1)  # Maximum equivalent stress over theta,r at each axial position
        
        
        

        #thermal_equiv, pressure_equiv, self.stress_equiv = [np.zeros_like(self.Tw) for v in range(3)]
        #Tw = self.Tw if not self.options.is_transient else self.Tw[..., time_index]
        #for z in range(self.disc.nz):
        #    thermal_equiv[z,:,:], pressure_equiv[z,:,:], self.stress_equiv[z,:,:] = self.cross_section.calculate_total_stress(self.P[z], Tw[z,:,:], use_property_temperature_variation)

        #self.set_result('max_thermal_stress_equiv', thermal_equiv.max(), time_index)  
        #self.set_result('max_pressure_stress_equiv', pressure_equiv.max(), time_index)
        #self.set_result('max_stress_equiv', self.stress_equiv.max(), time_index)
        #self.set_result('max_thermal_stress_equiv_axial', thermal_equiv.max(1).max(1), time_index)
        
        #self.set_result('max_thermal_stress_intensity', thermal_intensity.max(), time_index)
        #self.set_result('max_pressure_stress_intensity', pressure_intensity.max(), time_index)
        #self.set_result('max_stress_intensity', self.stress_intensity.max(), time_index)

        return 
    
    #=========================================================================
    
    def calculate_thermal_stress_relative_to_allowable(self):  # TODO not set up for transient calculations yet          
        if self.allowable_thermal_stress is None: # No provided allowable stress
            self.thermal_stress_fraction_of_allowable = np.zeros_like(self.Tf)
        elif not hasattr(self.allowable_thermal_stress, '__len__'):  # Constant allowable stress
            self.thermal_stress_fraction_of_allowable = self.max_thermal_stress_equiv_axial / self.allowable_thermal_stress  
        else: # Temperature-dependent allowable stress
            allow_stress = np.interp(self.Tw.max(2).max(1), self.allowable_thermal_stress[:,0], self.allowable_thermal_stress[:,1])  # Evaluate allowable stress at maximum temperature in the tube cross-section
            self.thermal_stress_fraction_of_allowable = self.max_thermal_stress_equiv_axial / allow_stress
        return
    
    
    def calculate_pressure_stress_relative_to_allowable(self):  # TODO not set up for transient calculations yet          
        if self.allowable_pressure_stress is None: # No provided allowable stress
            self.pressure_stress_fraction_of_allowable = 0.0
        elif not hasattr(self.allowable_thermal_stress, '__len__'):  # Constant allowable stress
            self.pressure_stress_fraction_of_allowable = self.max_pressure_stress_equiv / self.allowable_pressure_stress  
        else: # Temperature-dependent allowable stress
            allow_stress = np.interp(self.Tw.max(), self.allowable_pressure_stress[:,0], self.allowable_pressure_stress[:,1])  # Evaluate allowable pressure stress at maximum temperature in the tube
            self.pressure_stress_fraction_of_allowable = self.max_pressure_stress_equiv / allow_stress
        return


