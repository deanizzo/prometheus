from math import pi, sin, floor
from copy import deepcopy
import numpy as np
import tube_jwenner as tube
import materials
import operating_conditions
import util
import settings
import timeit
import multiprocessing
import json
import scipy.optimize as scOpt
import pandas as pd
import scipy.interpolate as spInt


class BillboardReceiver:
    def __init__(self):
        
        #--- Receiver design parameters
        self.site = None
        self.Qdes = 565               # Receiver design thermal power (MW)
        self.H = 18.0                 # Receiver height (m)
        self.D = 16.0                 # Receiver diameter (m)
        self.Htower = 170.            # Tower height (m)
        
        self.Npanels = 16             # Number of panels.  Panels are numbered using SolarPILOT convention (counterclockwise starting from south)
        self.Npanels_half = None       # Alternative specification for number of panels - if provided, this supersedes the specification in Npanels
        self.Wpan = float('nan')      # Panel width (set in initialize)
        self.helio_height = 12.2      # Heliostat height (used here only for wind velocity correction)        
        
        #--- Receiver tube parameters
        self.tube_OD = 0.04           # Tube outer diameter (m)
        self.tube_twall = 0.0012      # Tube wall thickness (m)
        self.tube_bends_90 = 4        # Number of 90 degree bends per tube
        self.tube_bends_45 = 0        # Number of 45 degree bends per tube
        self.area = float('nan')                # Total receiver panel area (m2) -> set in initialize()
        self.ntubes_per_panel = float('nan')    # Number of tubes per panel -> set in initialize()

        #--- Fluid and wall properties
        self.tube_material_name = 'Haynes230'            # Tube material (functions defined for 'Haynes230', 'SS316')
        self.tube_material_props = {}                    # Custom constant material properties to use if 'tube_material_name' = 'GenericConstantProp'
        self.HTF_material_name = 'Salt_60NaNO3_40KNO3'   # HTF material
        self.Tfin_design = 290+273.15               # HTF inlet temperature
        self.Tfout_design = 565+273.15              # HTF outlet temperature
        self.mflow_design = float('nan')            # Design point total mass flow rate (kg/s) -> set in initialize
        self.HTFcpavg = float('nan')                # Average HTF Cp (J/kg/K) -> set in initialize
        
        
        self.allowable_pressure_stress = None   # Allowable pressure stress (So)
        self.allowable_thermal_stress = None    # Allowable thermal stress (optional), either a constant value or a 2D array with columns of temperature (K) and allowable stress
        
        
        #--- Heat loss properties
        self.solar_abs = 0.94         # Tube solar absorptivity
        self.emis = 0.88              # Tube IR emissivity          
        self.m_comb = 3.2             # Coefficient for combining natural and forced convection
        self.est_heat_loss = 20.      # Estimated thermal loss (kW/m2) for field layout in SolarPILOT
        

        #---- Operating conditions
        self.operating_conditions = operating_conditions.OperatingConditions()  # Current operating conditions
        self.trans_operating_conditions = operating_conditions.OperatingConditions()  # Full set of operating conditions for transient model

        #--- Operational parameters
        self.flow_control_mode = 0       # Receiver mass flow contol: 0 = separate mass flow per path such that each path achieves the target exit temperature, 
                                         #                            1 = uniform flow per path. Total flow = sum(m1, m2, ..., mN) where m1, m2, etc. are the mass flow rates required to hit the exit temperature in path1, path2, etc. 
                                         #                            2 = uniform flow per path. Total flow = # paths * max(m1, m2, ... mN) where m1, m2, etc. are the mass flow rates required to hit the exit temperature in path1, path2, etc. 
        self.min_turndown = 0.20         # Min turndown fraction
        self.pump_efficiency = 0.85      # Pump efficiency
        self.expected_cycle_eff = 0.412  # Expected cycle efficiency for calculation of equivalent thermal pumping requirement
        self.include_tower_in_Qpump = False  # Include tower pumping requirements in Qpump
        
        #--- Flow path parameters
        self.user_flow_paths = None         # User-defined panel indicies in each flow path.  For example [[0,1], [2,3]] contains two flow paths: one through panels 0 and 1, and a second through panels 2 and 3.  Note, SolarPILOT numbers panels counter-clockwise starting from south                            
        self.npaths = 2                     # Number of receiver flow paths
        self.ncross = 1                     # Number of times the flow is allowed to cross the receiver
        self.is_cross_to_high = True        # True if paths starting at north-most location in the northeast quadrant cross to the north-most location in the southwest quadrant
        self.is_min_before_cross = True     # If paths contain different numbers of panels in the northeast quadrant circuits will alternate between n and n+1 panels before crossing to southwest quadrant 
        self.is_skip_panels = False         # If False receiver flow circuits must consist of adjacent panels (except for when flow crosses the receiver)
        self.is_bottom_inlet = True         # Flow input position position (top or bottom) 
        
        self.npanel_per_path = 0           # Number of receiver panels per flow path 
        self.flow_path_length = []         # Flow path length for each path (m)  
        self.flow_paths = [[]]             # Flow paths. For example [[0,1], [2,3]] contains two flow paths: one through panels 0 and 1, and a second through panels 2 and 3. Panels are numbered counter-clockwise starting from south               
        self.crossover_after_panel = []    # Index of last panel before flow path crosses the receiver (for each flow path)


        #--- Receiver header parameters (only used in transient model)
        self.ignore_interpanel_headers = True  # Ignore headers between panels
        self.header_hext = 0.0         # Header external heat loss coefficient (W/m2/K)   # TODO: Need to update calculations of overall Qfluid, etc. to include header losses if hext > 0
        self.header_ID = 0.30          # Header ID (m)
        self.header_OD = float('nan')
        self.header_twall = 0.01       # Header wall thickness (m)
        self.crossheader_ID = 0.30     # Cross-over header ID (m)
        self.crossheader_OD = float('nan')
        self.crossheader_twall = 0.01  # Cross-over header wall thickness (m)
        self.header_heff = 0.0                  # Wetted heat loss coefficient from header (W/m2/K)
        self.header_length = float('nan')       # Header length (m) (set in intialize())
        self.crossheader_length = float('nan')  # Length of cross-over header (m) (set in initialize())

        '''
        #--- Riser and downcomer parameters (only used in transient model, not implemented yet)
        self.riser_ID = 0.45        # Riser ID (m)
        self.riser_twall = 0.015    # Riser wall thickness (m)
        self.riser_hext  = 0.0       # Wetted heat loss coefficient from riser (W/m2/K)
        self.downcomer_ID = 0.45    # Downcomer ID (m)
        self.downcomer_twall = 0.015 # Downcomer wall thickness (m)
        self.downcomer_hext  = 0.0    # Wetted heat loss coefficient from riser (W/m2/K)
        self.piping_mult = 1.0       # Length of riser or downcomer relative to tower height 
        '''

        

        #--- Solution parameters
        self.options = settings.SolutionOptions()
        self.disc = settings.Discretization()
        
        self.ntubesim = 3           # Number of simulated tubes per panel
        self.Nz_crossheader = 30    # Number of axial discretization points per cross-over header
        self.Nz_header = 6          # Number of axial discretization points per inter-panel header     

        self.Ttol_for_tube = 5e-4   # Temperature tolerance for tube solutions (inner iterations to converge tube temperature profiles w.r.t radiative loss and temperature-dependent internal h and wall k)
        self.Ttol_for_hext = 1e-3   # Temperature tolerance for iterations to converge evaluation temperature for extenral convection coefficients
        self.Ttol_for_mflow = 5e-4  # Temperature tolerance for iterations to converge mass flow iterations to reach target exit temperature
        self.n_flow_iter_max = 20   # Maximum number of iterations for mass flow
        self.n_hext_iter_max = 20   # Maximum number of iterations for external convection coefficients 
        self.tube_weights = []      # Tube weighting factors for total energy balances (set in initialize)

        self.Re_cutoff = 5000              # Cutoff Re, solutions will be stopped if conditions in any tube drop below this value

        
        #--- Transient model formulation
        self.time_step = 0.5        # Time set (seconds)
        self.soln_time = 60         # Solution time (seconds)
        self.initial_condition = 'steady_state'  # 'steady_state' = steady state at first set of operating condition in trans_operating_conditions,                           
                                                 # 'constant' = Constant fluid and wall T
        self.initial_constant_Tf = float('nan')   # Constant fluid T to use if initial_condition = 'constant'
        self.initial_constant_Tw = float('nan')  # Constant wall T to use if initial_condition = 'constant'        
        


        #--- Tubes
        self.base_tube = tube.Tube()   # Base case instance of tube class (properties will be set in initialize)
        self.tubes = []                # List of tubes per panel. Note all tubes are oriented in the solution such that the flow inlet of the tube is at the beginning of the solution arrays. 
        self.headers = []
        self.crossheaders = []
        
        
        #--- Parallelization (only set up for transient model)
        self.parallel_tube_calculation = False  # Parallelize calculations for tubes in a given panel?
        self.nprocess = 3                      # Number of processes to use if parallel_tube_calculation = True

        
        #--- Results
        self.clear_solution(False)
        
        return
    
    def clear_solution(self, is_clear_tubes = True):

        # Single values stored as a function of time (if transient)
        self.Qfluid = float('nan')  # Total heat transfer into fluid (W)
        self.Qsinc = float('nan')   # Total incident solar energy (W)
        self.Qsabs = float('nan')   # Total absorbed solar energy (W)
        self.Qconv = float('nan')   # Total convection loss (W)
        self.Qrad = float('nan')    # Total IR radiation loss (W)
        self.Qpump = float('nan')   # Thermal-equivalent pump power (W)
        self.mflow = float('nan')   # Total mass flow (kg/s)
        self.eta_therm = float('nan')  # Thermal efficiency (Qfluid / Qsabs)
        self.eta = float('nan')     # Receiver efficiency (Qfluid / Qsinc)
        self.eta_pump = float('nan')  # Receiver efficiency including pump loss ((Qfluid - Qpump) / Qsinc)
        self.pressure_drop_with_tower = float('nan')  # PRessure drop across the receiver and riser (MPa)
        self.thermal_stress_fraction_of_allowable_max = float('nan')  # Maximum (thermal stress / allowable stress) occuring at any location on the receiver (MPa)
        self.pressure_stress_fraction_of_allowable_max = float('nan')     # Maximum (pressure stress / allowable pressure stress) occuring at any location on the receiver (MPa)


        # Single values stored only for steady state model, or at most recently solved time point
        self.tol = float('nan')     # Solution temperature tolerance      
        self.mass_flow_converged = False
        self.stopped = False     
        self.soln_code = None  # 0 = successful, 1 = Re under cutoff, 2 = tube temperature solution failed to converge (only applied to steady state solution)
        self.n_flow_iter = 0
        self.n_hext_iter = 0
        
        # Per-path and per-panel values stored as a function of time (if transient)
        self.mflow_per_path = []
        self.Qfluid_per_path = []
        self.Tfout_per_path = []
        self.pressure_drop_per_path = []  # Pressure drop across the receiver (MPa)
        self.Tfout_per_panel = []
        self.hext_per_panel = []
        self.thermal_stress_fraction_of_allowable_max_per_panel = [] # Maximum (stress / allowable stress) occuring for each panel

        
        # Per-tube values stored as a function of time (if transient)
        self.max_stress_per_tube = []  # Maximum von Mises equivalent stress per simulated tube (MPa)

        if is_clear_tubes:
            for p in range(self.Npanels):
                for k in range(self.ntubesim):
                    self.tubes[p][k].clear_solution()            

        return

       


    #=========================================================================
    def load_from_json(self, filename):
        with open('LTE/' +filename + '.json', 'r') as f: 
            params = json.load(f)
        for key,val in params.items():
            setattr(self,key,val)
        self.initialize()
        return

            
    #=========================================================================
    def initialize(self, only_receiver_sizing = False):

        #--- Enforce input parameter data types
        self.H = float(self.H)
        self.D = float(self.D)

        #--- Initialize base receiver tube 
        base_tube = tube.Tube()   # Base case instance of tube class (properties will be set in initialize)
        tube_vars = vars(base_tube).keys()
        for k,val in vars(self).items():   # Set variables in tube class that are an exact match for parameter names in this class
            if k in tube_vars:
                setattr(base_tube, k, val)
        base_tube.update_inputs(length = self.H, OD = self.tube_OD, twall = self.tube_twall)  # Set variables in tube class that are not an exact match for parameter names in this class
        base_tube.initialize()      

        #---set the number of panels, and modify if json input requests it
        self.Npanels = int(self.Npanels) if self.Npanels_half is None else int(2*self.Npanels_half)
        #--- make a base tube for velocity calculations to reference
        if self.modify_Npanels:
            self.mod_Npanels(base_tube)       
        
        self.npaths = int(self.npaths)
        self.ncross = int(self.ncross)
        
        # make an aiming file for calculating flux if none already exists
        if (self.use_aiming_scheme) and (self.aiming_file=="None"):
            txt = "aiming/aiming_file_Qdes_{input1:.0f}_stage1.csv"
            aiming_file=txt.format(input1=self.Qdes)
            print("aiming file name is:", aiming_file)
            self.make_informed_flux_scheme(base_tube,aiming_file)
            self.aiming_file = aiming_file
        #--- Calculate derived parameter inputs
        # self.Wpan = self.D * sin(pi/float(self.Npanels))
        self.Wpan = self.D / float(self.Npanels)                # JWENNER mod
        self.area = self.Npanels * self.Wpan
        self.ntubes_per_panel = int(floor(self.Wpan / self.tube_OD))
        
        self.header_length = 2*self.Wpan if not self.is_skip_panels else 3*self.Wpan
        self.header_OD = self.header_ID + 2*self.header_twall
        self.crossheader_length = self.D        
        self.crossheader_OD = self.crossheader_ID + 2*self.crossheader_twall

        if only_receiver_sizing:    # Will typically be False, unless this function is called after flux profile generation with re-sizing of receiver diameter
            return
        

        #--- Set up receiver flow paths
        ok = self.set_flow_paths()       

        #--- Check consistency of inputs
        if self.ntubesim % 2 !=1:
            self.ntubesim += 1
            print ("Warning: Number of tubes analyzed per panel must be odd. Increasing 'ntubesim' to %d"%self.ntubesim)

        #--- Set tube weighting for total energy balance (tubes at panel extremes represent less of the total panel area)
        self.tube_weights = np.ones(self.ntubesim)
        if self.ntubesim > 1:
            self.tube_weights = 1./(self.ntubesim-1) * np.ones(self.ntubesim)
            self.tube_weights[0] *= 0.5
            self.tube_weights[-1] *= 0.5         
            

        
        #--- initialize all other tube instances
        self.tubes = [[deepcopy(base_tube) for j in range(self.ntubesim)] for k in range(self.Npanels)]  
        

        #--- Calculate design point mass flow
        Tpts = np.linspace(self.Tfin_design, self.Tfout_design, 100)
        self.HTFcpavg = np.trapezoid(self.tubes[0][0].fluid.cp(Tpts), Tpts) / (self.Tfout_design - self.Tfin_design)
        self.mflow_design = self.Qdes*1.e6 / self.HTFcpavg / (self.Tfout_design - self.Tfin_design) 
        
        

        #--- Initialize flow direction in each tube
        for path in range(self.npaths):
            isup = self.is_bottom_inlet 
            for panel in self.flow_paths[path]:
                for j in range(self.ntubesim):
                    self.tubes[panel][j].flow_against_gravity = isup
                if panel not in self.crossover_after_panel:  # Only reverse flow direction with flow doesn't cross receiver
                    isup = not isup  # Reverse flow direction for next panel
                       
        
        #--- Initialize header tube instances
        header = tube.Tube()
        for k,val in vars(self).items():
            if k in vars(header).keys() and k not in ['options', 'disc']:
                setattr(header,k,val)
        header.update_inputs(is_axisymmetric = True, is_solar = False, tube_bends_90 = 0, tube_bends_45 = 0, solar_abs = 0.0, emis = 0.0, hext = self.header_hext, include_gravity = False)  # Inputs applicable to all headers
        header.options = deepcopy(self.options)
        if self.ncross>0:
            header.disc = settings.Discretization(self.disc.nr, self.disc.ntheta, self.Nz_crossheader)
            header.update_inputs(length = self.crossheader_length, OD = self.crossheader_OD, twall = self.crossheader_twall)
            header.initialize()
            self.crossheaders = [deepcopy(header) for j in range(self.npaths)]            
        if not self.ignore_interpanel_headers:
            header.disc = settings.Discretization(self.disc.nr, self.disc.ntheta, self.Nz_header)
            header.update_inputs(length = self.header_length, OD = self.header_OD, twall = self.header_twall)
            header.initialize()
            nheaders = sum([len(p)-1-self.ncross for p in self.flow_paths])
            self.headers = [deepcopy(header) for j in range(nheaders)]   
            
        
        #---- Initialize order of flow path units
        self.flow_path_names = []
        self.flow_path_order = []
        h = 0
        hc = 0
        for p in range(self.npaths):
            order = []
            names = []
            npan = len(self.flow_paths[p])
            for j in range(npan):  # Loop over panels within flow paths
                panel = self.flow_paths[p][j]
                names.append('p%d'%panel)
                order.append(self.tubes[panel])
                if j < npan-1:
                    if panel in self.crossover_after_panel:
                        names.append('hc%d'%hc)
                        order.append([self.crossheaders[hc]])
                        hc += 1
                    elif not self.ignore_interpanel_headers:
                        names.append('h%d'%h)
                        order.append([self.headers[h]])
                        h += 1
            self.flow_path_names.append(names)
            self.flow_path_order.append(order)
               
        #--- Limit number of processes to the number of simulated tubes
        self.nprocess = min(self.nprocess, self.ntubesim) 
        
        return ok
    
    def reinitialize_tubes(self):
        for j in range(len(self.tubes)):
            for k in range(len(self.tubes[j])):
                self.tubes[j][k].initialize()
        for h in self.headers:
            h.initialize()
                
        return
            
    
    #=========================================================================
    # Get and set values
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
    


                
    #=========================================================================
    #--- Set receiver flow paths
    def set_flow_paths(self):
        if self.user_flow_paths:
            self.flow_paths = self.user_flow_paths 
            self.npaths = len(self.flow_paths) 
            self.npanel_per_path = len(self.flow_paths[0]) 
            self.ncross = len(np.where(np.abs(np.diff(self.flow_paths[0]))>1)[0])
        else:
            ok = self.calculate_flow_paths_billboard()
            if not ok:
                return False
        #self.npaths = len(self.flow_paths)        
        self.flow_path_length = self.npanel_per_path*self.H * np.ones(self.npaths)       
        self.is_crossover_allowed = True if self.ncross>=1 else 0
        
        #--- Find panels immediately preceeding the flow crossing the receiver
        if self.ncross == 0 or self.npanel_per_path == 1:           
             self.crossover_after_panel  = [None for p in range(self.npaths)]  
        else:
            self.crossover_after_panel = []
            for c in range(self.npaths):
                j = np.where(np.abs(np.diff(self.flow_paths[c]))>1)[0][0]
                self.crossover_after_panel.append(self.flow_paths[c][j])

        return True
    
    #=============================================================================
    #--- modify the number of panels until each tube velocity equals the reference velocity
    def mod_Npanels(self,base_tube):
        """
        Qdes - (MWth) design power of the receiver
        HTFcpavg - (J/kg-K) average specific heat capacity of the HTF
        Tfout_design - (K) design outlet temperature
        Tfin_design - (K) design inlet temperature
        W - (m) receiver width
        vel_ref - (m/s) objective velocity to match
        """
        tol = 0.05
        upBound=self.vel_ref+tol*self.vel_ref
        lowBound=self.vel_ref-tol*self.vel_ref

        Tf_avg = (self.Tfout_design + self.Tfin_design)/2
        
        rho = base_tube.fluid.density(Tf_avg)
        cp_avg = base_tube.fluid.cp(Tf_avg)

        # Wpan = W / float(Npanels)                # JWENNER mod
        # ntubes_per_panel = int(floor(Wpan / tub.ro))
        Npanels = self.Npanels
        done = False
        firstLoop = True
        vel_old = 0.0001 # give some starting value
        while done == False:
            Wpan = self.D / float(Npanels)                # JWENNER mod
            ntubes_per_panel = int(floor(Wpan / self.tube_OD))
            mflow_design = self.Qdes*1.e6 / cp_avg / (self.Tfout_design - self.Tfin_design)
            mDot_path = mflow_design/self.npaths
            mDot_tube = mDot_path/ntubes_per_panel
            vel = mDot_tube/rho/(np.pi * (base_tube.ri**2) )
            
            if lowBound <= vel <= upBound:
                done = True
            elif (vel_old < lowBound) and (vel > upBound) and (firstLoop == False):
                done = True
            elif (vel_old > upBound) and (vel < lowBound) and (firstLoop == False):
                done = True
            elif vel > upBound:
                Npanels = Npanels - 1*self.npaths
            elif vel < lowBound:
                Npanels = Npanels + 1*self.npaths
            vel_old = vel # store previous loop's vel for comparison
            firstLoop = False
        print('Npanels updated to:',Npanels,)
        self.Npanels = int(Npanels) # re-assign to receiver attribute
        return True

    def make_informed_flux_scheme(self,tube, aiming_file ): ## NOTE: untested as of 3/6/2025
        '''
        make a matrix of aiming intensity levels for solarpilot to follow
        mDot - (kg/s) total mass flow
        npaths - receiver number of paths
        Apanel - (m2) area of each panel in receiver
        ctr_file - (.csv) contour file for making lifetime-dT contour. for now this is hardcoded
        k - (W/m-K) conductivity for Haynes 230
        th - (m) tube thickness
        hbar - (W/m2-K) heat transfer coefficient 
        Tstart_C, Tend_C - (C) fluid inlet and outlet temperatures
        '''
        def solveFlux(Tf2, Tf1, mdot, c, deltaTfun , A, Rtotal):
            '''
            deltaTfun - returns the ctr-allowable temperature difference for any Tf2
            Tf1, Tf2 - starting and ending temperatures of the fluid -- Tf1 is always known
            mdot - mass flow (kg/s)
            c - specific heat capacity (J/kg-K)
            A - heat flux area (m^2)
            '''
            qflux = deltaTfun(Tf2)/Rtotal # converts the allowable deltaT into an equivalent absorbed flux based on thermal resistance assumptions
            return qflux - mdot*c*(Tf2 - Tf1)/A

        Atotal = self.D*self.H
        Apanel = Atotal/self.Npanels
        npaths = self.npaths
        
        ## get the design mass flow, which may slightly differ from actual
        Tf_avg = (self.Tfout_design + self.Tfin_design)/2
        rho = tube.fluid.density(Tf_avg)
        cp_avg = tube.fluid.cp(Tf_avg)
        mflow_design = self.Qdes*1.e6 / cp_avg / (self.Tfout_design - self.Tfin_design)

        Wpan = self.D / float(self.Npanels)                # JWENNER mod
        ntubes_per_panel = int(floor(Wpan / self.tube_OD))

        mDot_path = mflow_design/self.npaths
        mDot_tube = mDot_path/ntubes_per_panel

        ## assign the mass flow to our base tube, then calculate the design flow heat transfer coefficient
        tube.mflow=mDot_tube
        hbar, Re, vel = tube.internal_h(Tf=Tf_avg)

        ## make the contour
        verts_df = pd.read_csv('dmg_tool_data/A230_30yr_ctrs_tempBase.csv')
        dT_ctr = verts_df['dT'].values
        Tf_ctr = verts_df["Tf"].values
        deltaT_fit =spInt.interp1d(Tf_ctr, dT_ctr,fill_value='extrapolate')

        ## thermal inputs for converting dT to flux values
        k= tube.tube_wall.k(Tf_avg+140) # added addtiional wall temperature because flux will be heating it. No good way to estimate this pre-simulation? (W/m-K)
        th=tube.twall #1.25e-3 # (m) tube thickness
        # hbar=8784   # (W/m2-K) from janna's thermal model. Weighted average that considers tube's location
        Rtotal=(th/k)+(1/hbar) #(m2-K/W)

        ## get the cp for the tube fluid

        ## convert Tstart and Tend to celsius because contour axis is in Celsius units
        Tstart_C = self.Tfin_design - 273.15
        Tend_C   = self.Tfout_design- 273.15

        ## get starting point of flowpath
        start_pt = self.start_pt


        ## get y flux profile distribution. Not really that important
        ny = self.n_flux_y


        # specify start and end
        Tstart_C=np.array([Tstart_C]) # (C)

        # step through the contour line, finding flux points along the way
        Tfs=[]
        deltaTs  =[] #allowable deltaTs at each each Tf
        fluxes_W =[]
        Tfs.append(Tstart_C)
        tol=1 #(C or K) outlet temperature tolerance
        while Tfs[-1]< Tend_C-tol:
            TfNext_limit=scOpt.fsolve(solveFlux,(Tfs[-1]+5),(Tfs[-1], mDot_path, cp_avg, deltaT_fit , Apanel, Rtotal))
            if TfNext_limit > Tend_C:
                TfNext_limit = np.array([Tend_C])   #return the ending temperature
            # qflux_limit = deltaT_fit(TfNext_limit)/Rtotal
            qflux_used=mDot_path*cp_avg*(TfNext_limit-Tfs[-1])/Apanel #calculate the actual flux used. In the case of the last panel, this will NOT equal the flux limit
            fluxes_W.append(qflux_used)
            deltaTs.append(deltaT_fit(TfNext_limit))
            Tfs.append(TfNext_limit) # add the next temperature in either case
        
        ## convert fluxes_kW list into a single npArray
        fluxes_W = np.array([flux[0] for flux in fluxes_W])
        fluxes_kW=fluxes_W/1000     # add resolution by repeating array elements. Should help solarpilot make more discrete flux patterns

        ## make an aiming scheme csv
        fluxes_normd=fluxes_W/fluxes_W.max()
        fluxes_normd =np.repeat(fluxes_normd, 3 ,axis=0)

        if start_pt=='ctr':
            RHS_pts=fluxes_normd
            LHS_pts=RHS_pts[::-1]
            fluxes_normd=np.concatenate((LHS_pts,RHS_pts))
            aiming_matrix=fluxes_normd*np.ones([ny,1])
            aiming_df=pd.DataFrame(data=aiming_matrix, index=None, columns=None, dtype=None, copy=None) # create empty dataframe 
            aiming_df.to_csv(aiming_file,header=None, index=None)
        else: 
            print('!!! no aiming scheme currently in place for this flowpath configuration !!!')
            return


        return
    # #===========================================================================
    # # Set up flow paths from input parameters. NOTE: for a cylindrical receiver only
    # def calculate_flow_paths(self):

    #     ncircuit = self.npaths
    #     self.npanel_per_path = int(self.Npanels / self.npaths)
        
    #     if self.Npanels%self.npaths > 0:
    #         print('Error: Number of panels is not an integer multiple of the number of flow circuits')
    #         return False
        
    #     if self.npaths%2>0:
    #         print ('Error: Total number of flow circuits must be even')
    #         return False
        
    #     if self.npaths > 2 and self.npanel_per_path%4 > 0:
    #         print ('Error: Specified number of panels must be an integer multiple of 4 if more than two paths are used')
    #         return False

        
    #     nc = int(ncircuit / 2)  # Number of flow circuits starting on one side of the receiver
    #     inorth = int(self.Npanels/2)-1     # North-most panel on the east-side of the receiver
    #     ieast = int(self.Npanels/4)-1        
       
    #     self.flow_paths = []
    #     if self.ncross == 0 or self.npanel_per_path == 1:  # Flow paths are not allowed to cross the receiver (or single-panel flow paths)
    #         i =  inorth
    #         nskip = 0 if not self.is_skip_panels else nc-1
    #         for c in range(nc):
    #             panels = [i - j - j*nskip for j in range(self.npanel_per_path)]  # Flow-circuit starting on east side
    #             panels_mirror = [self.Npanels-1-p for p in panels]      # Flow-circuit starting on west side
    #             self.flow_paths.append(panels)
    #             self.flow_paths.append(panels_mirror)
    #             i = i-self.npanel_per_path if not self.is_skip_panels else i-1
    
     
    #     elif self.npanel_per_path == 2 and self.ncross == 1:  # Two panels per path, crossing receiver
    #         i = inorth        
    #         j = self.Npanels-1-ieast if self.is_cross_to_high else self.Npanels-1 
    #         for c in range(nc):
    #             panels = [i,j]                              # Flow-circuit starting on east side
    #             panels_mirror = [self.Npanels-1-p for p in panels]  # Flow-circuit starting on west side
    #             self.flow_paths.append(panels)
    #             self.flow_paths.append(panels_mirror)
    #             i -=1
    #             j = j+1 if self.is_cross_to_high else j-1
                
    #     elif self.npanel_per_path>2 and self.ncross == 1:  
            
    #         if not self.is_skip_panels:
    #             # Find number of paths in north-east quadrant before crossing (for each flow pass)
    #             nq = self.Npanels/4
    #             nf = nq / float(nc)    # Average number of flow passes in the north-east quarter per flow pass
    #             n = int(nf)
    #             if n == nf:
    #                 nnorth = n * np.ones(nc, int)   # Number of panels in north-east quadrant before crossing
    #             else:
    #                 nnorth = np.tile([n, n+1], nc) if self.is_min_before_cross else np.tile([n+1, n], nc) 
    #             nsouth = self.npanel_per_path - nnorth
        
    #             # Set up flow circuits
    #             i = inorth
    #             j = self.Npanels-1-ieast if self.is_cross_to_high and nc>1 else self.Npanels-nsouth[0] 
    #             for c in range(nc):
    #                 panels = [i-k for k in range(nnorth[c])] + [j+k for k in range(nsouth[c])]  # Flow-circuit starting on east side
    #                 panels_mirror = [self.Npanels-1-p for p in panels]  # Flow-circuit starting on west side
    #                 self.flow_paths.append(panels)
    #                 self.flow_paths.append(panels_mirror)  
    #                 if c < nc-1:
    #                     i -= nnorth[c]
    #                     j = j+nsouth[c] if self.is_cross_to_high else j-nsouth[c+1]
        
    #         else:
    #             i = inorth
    #             nskip = 0 if not self.is_skip_panels else nc-1
    #             for c in range(nc):
    #                 panels = [i - j - j*nskip for j in range(self.npanel_per_path)] 
    #                 for p in range(len(panels)):
    #                     if panels[p] <= ieast:
    #                         panels[p] = self.Npanels-1-panels[p]
    #                 panels_mirror = [self.Npanels-1-p for p in panels]  # Flow-circuit starting on west side
    #                 self.flow_paths.append(panels)
    #                 self.flow_paths.append(panels_mirror)  
    #                 if c < nc-1:
    #                     i -= 1        
    
    #     return True
    
    #===========================================================================
    # Set up flow paths from input parameters
    def calculate_flow_paths_billboard(self):
        """
        creates a flowpath list based on starting point info, number of paths, and number of panels
        """
        self.npanel_per_path = int(self.Npanels / self.npaths)
        
        if self.Npanels%self.npaths > 0:
            print('Error: Number of panels is not an integer multiple of the number of flow circuits')
            return False
        
        # if self.npaths%2>0:
        #     print ('Error: Total number of flow circuits must be even')
        #     return False
        
        # if self.npaths > 2 and self.npanel_per_path%4 > 0:
        #     print ('Error: Specified number of panels must be an integer multiple of 4 if more than two paths are used')
        #     return False
        
        self.flow_paths = []
        if self.start_pt == 'ctr' and self.npaths == 2: # this is the "start in center, both flow paths go outwards" condition
            p_list = list(range(self.Npanels))
            flow_path_left= p_list[:int(self.Npanels/self.npaths)]
            flow_path_left.reverse()
            flow_path_right= p_list[int(self.Npanels/self.npaths):]
            self.flow_paths.append(flow_path_left)
            self.flow_paths.append(flow_path_right)
        elif self.start_pt == 'left' and self.npaths == 1:
            p_list = list(range(self.Npanels))
            self.flow_paths.append(p_list)
        else:
            print('Error: combination of starting point and number of paths inputs not recognized')
            return False
        return True


    #=========================================================================
    def estimate_loss_coeff(self, hext = 10, dTwallavg = 50):
        hext = 10
        n = 20
        dT = (self.Tfout_design - self.Tfin_design)/float(n)
        Twavg = 0.0
        Tw4avg = 0.0
        for j in range(n+1):
            Tw = (self.Tfin_design + j*dT) + dTwallavg
            Twavg += Tw/float(n+1)
            Tw4avg += (Tw**4)/float(n+1)
        loss_coeff = (hext*(Twavg-298.) + self.emis*5.6704e-8*(Tw4avg - (298.**4)))  # W/m2
        return loss_coeff
        
        

    #=========================================================================
    def get_flow_path_element(self, name):
        if name[0] == 'p':  
            i = int(name.split('p')[1])
            return self.tubes[i]
        elif name[0:2] == 'hc':
            i = int(name.split('hc')[1])
            return self.crossheaders[i]
        elif name[0] == 'h':
            i = int(name.split('h')[1])
            return self.crossheaders[i]
     
    def get_outlet_temperature(self, path, idx, time_index = None):
        elem = self.flow_path_order[path][idx]
        name = self.flow_path_names[path][idx]
        Tout = np.zeros(len(elem))
        for k in range(len(elem)):  # Parallel elements in this flow pass
            Tout[k] = elem[k].Tf[-1] if not self.options.is_transient else elem[k].Tf[-1, time_index]
        Tout = (self.tube_weights * Tout).sum() if name[0] == 'p' else Tout[0]
        return Tout

        

    
    #=========================================================================
    # Retrieve solution values from tube instances and store in a 2D array of panels, tubes
    # valtype = 'min', 'max', 'avg', 'inlet', 'outlet' (only required if the specified output variable name is an array)
    # wall_spec = 'front_outer', 'all_outer', 'front_inner', 'all_inner' for front/total wall circumferential nodes and outer/inner wall temperatures
    def get_tube_solns(self, name, combine_tubes = 'avg', panel = None, solntype = 'max', wall_spec = 'front_outer', time_index = None):      
        panels = np.arange(self.Npanels) if panel is None else [panel]
        npan = len(panels)
        nt = int((self.disc.ntheta+1)/2) if wall_spec[0:5] == 'front' else self.disc.ntheta
        rnode = 0 if wall_spec[-5:] == 'inner' else -1
        array = np.zeros((npan, self.ntubesim))
        for j in range(len(panels)):
            for k in range(self.ntubesim):
                p = panels[j]
                data = self.tubes[p][k].get_value(name)
                if self.options.is_transient:  # Take data only at specific timestep
                    data = data[...,time_index] if hasattr(data[0], '__len__') else data[time_index]
                    
                is_single_value = not hasattr(data, '__len__')
                if name == 'Tw':
                    data = data[:, 0:nt, rnode]
                    
                if is_single_value:
                    array[j,k] = data
                elif solntype == 'inlet':
                    array[j,k] = data[0]
                elif solntype == 'outlet':
                    array[j,k] = data[-1]
                elif solntype == 'max':
                    array[j,k] = data.max()
                elif solntype == 'min':
                    array[j,k] = data.min()
                elif solntype == 'avg':
                    array[j,k] = data.mean()
        
        if combine_tubes == 'avg':
            array = (self.tube_weights * array).sum(1)  # Average over simulated tubes
        elif combine_tubes == 'max':
            array = array.max(1)  # Maximum over simulated tubes
        elif combine_tubes == 'min':
            array = array.min(1)  # Maximum over simulated tubes            

        val = array
        if panel is not None:
            val = array[0] if combine_tubes is not None else array[0,:]

        return val
    
    def get_array_of_axial_profiles(self, name):
        array = np.zeros((self.disc.nz, self.Npanels, self.ntubesim))
        for p in range(self.Npanels):
            for k in range(self.ntubesim):
                data = self.tubes[p][k].get_value(name)
                ndim = len(data.shape)
                for j in range(ndim-1):
                    data = data.max(1)   # Max over all dimensions other than the axial dimension (first in all array results)
                array[:,p,k] = data[:]   # Axial positions (rows) are from inlet to outlet of each tube
        return array

    def get_array_of_Tinner_low_axial_profiles(self, name): # this only works with attribute 'Tw_inner_low'
        array = np.zeros((self.disc.nz, self.Npanels, self.ntubesim))
        for p in range(self.Npanels):
            for k in range(self.ntubesim):
                data = self.tubes[p][k].get_value(name)
                ndim = len(data.shape)
                for j in range(ndim-1):
                    data = data.min(1)   # min over all dimensions other than the axial dimension (first in all array results)
                array[:,p,k] = data[:]   # Axial positions (rows) are from inlet to outlet of each tube
        return array
    
    def get_array_of_Tinner_high_axial_profiles(self, name): # this only works with attribute 'Tw_inner_high'
        array = np.zeros((self.disc.nz, self.Npanels, self.ntubesim))
        for p in range(self.Npanels):
            for k in range(self.ntubesim):
                data = self.tubes[p][k].get_value(name)
                data = data[:,:,0].max(1)   # max over theta dimension, leave z dimension untouched, select the inner radial nodes corresponding to r=0
                array[:,p,k] = data[:]   # Axial positions (rows) are from inlet to outlet of each tube
        return array
    
    #=========================================================================
    # External convection coefficients coefficients (from Siebers and Kraabel, SAND84-8717) 
    def external_h_forced(self, Twall, is_worst_case = False):
        if self.operating_conditions.vwind10 == 0.0:
            return 0.0
        
        vwind = util.calculate_wind_velocity(self.operating_conditions.vwind10, self.Htower, self.helio_height)  # Wind velocity at receiver centerline
        Tfilm = 0.5*(Twall + self.operating_conditions.Tamb)
        air = materials.create_material('Air')
        rho = air.density(Tfilm)
        visc = air.viscosity(Tfilm)
        k = air.k(Tfilm)
        Re = rho*vwind*self.D/ visc
        rough = self.tube_OD/ 2.0 / self.D  # relative roughness
    
        if rough < 75e-5:
            pts = [0.0 , 75e-5]
        elif rough < 300e-5:
            pts = [75e-5 , 300e-5]
        elif rough < 900e-5:
            pts = [300e-5 , 900e-5]
        else:
            pts = [900e-5]
    
        Nupts = []
        for p in pts:
            if p == 0.0:  # Correlation for roughness = 0.0
                Nu = 0.3 + 0.488*pow(Re, 0.5 )*pow( (1.0+pow( (Re/282000), 0.625 )), 0.8) 
            elif p == 75e-5: # Correlation for roughness = 75e-5
                if Re<7e5:
                    Nu = 0.3 + 0.488*pow( Re, 0.5 )*pow( (1.0+pow( (Re/282000), 0.625 )), 0.8)
                elif Re<2.2e7:
                    Nu = 2.57E-3*pow( Re, 0.98 )
                else:
                    Nu = 0.0455*pow( Re, 0.81 )
            elif p == 300e-5: # Correlation for roughness = 300e-5
                if Re < 1.8e5:
                    Nu = 0.3 + 0.488*pow( Re, 0.5 )*pow( (1.0+pow( (Re/282000), 0.625 )), 0.8)
                elif Re < 4e6:
                    Nu = 0.0135*pow( Re, 0.89 )
                else:
                    Nu = 0.0455*pow( Re, 0.81 )            
            elif p == 900e-5:  # Correlation for roughness = 900e-5
                if Re<1e5:
                    Nu = 0.3 + 0.488*pow( Re, 0.5 )*pow( (1.0+pow( (Re/282000), 0.625 )), 0.8)
                else:
                    Nu = 0.0455*pow( Re, 0.81 )
            Nupts.append(Nu)

        if is_worst_case:  # Used highest Nu at bounding roughness points
            Nu = max(Nupts)
        else:  # Linearly interpolate between roughness points
            Nu = Nupts[0] if len(pts) == 1 else Nupts[0] + ( (Nupts[1]-Nupts[0]) / (pts[1]-pts[0]) ) * (rough - pts[0])
        h = Nu*k/self.D  # Forced convection coefficient
        return h
         
    def external_h_natural(self, Twall):   
        air = materials.create_material('Air')
        rho = air.density(self.operating_conditions.Tamb)
        visc = air.viscosity(self.operating_conditions.Tamb)
        k = air.k(self.operating_conditions.Tamb)
        kinvisc = visc / rho   # Kinematic viscosity (m2/s)
        Gr_nat = 9.8*(1./self.operating_conditions.Tamb)*(Twall-self.operating_conditions.Tamb)*self.H**3/kinvisc**2
        hext_nat = (k/self.H) * 0.098*Gr_nat**(1./3)*(Twall/self.operating_conditions.Tamb)**(-0.14)   
        hext_nat *= pi/2   # Increase in h to account for ribbed surface (per Siebers and Kraabel )
        return hext_nat
    
    def external_h(self, Twall_nat, Twall_forced):   
        hf = self.external_h_forced(Twall_forced)
        hn = self.external_h_natural(Twall_nat)
        hmix = (hf**self.m_comb + hn**self.m_comb)**(1./self.m_comb)
        return hmix * (2/pi)   # Return value relative to tube surface area

        
    #=========================================================================
    # Update input parameters in this class and all tube and header instances
    def update(self, D, is_update_tubes = True, is_update_headers = True):
        for name, val in D.items():
            self.set_value(name,val)
        if is_update_tubes:
            self.update_tube_inputs(D)
        if is_update_headers:
            self.update_header_inputs(D)
        return

    
    # Update input parameters in tube instances
    def update_tube_inputs(self, D, is_per_path = False, is_per_panel = False):
        for name, val in D.items():
            for path in range(self.npaths):
                for p in self.flow_paths[path]:
                    tubeval = val[p] if is_per_panel else (val[path] if is_per_path else val)
                    for t in range(self.ntubesim):
                        self.tubes[p][t].set_value(name, tubeval)
        return
    
    def update_header_inputs(self, D, is_per_path = False):
        for name, val in D.items():
            if not is_per_path:
                for h in self.headers+self.crossheaders:
                    h.set_value(name, val)
            else:
                for path in range(self.npaths):
                    for n in self.flow_path_names[path]:
                        if n[0:2] == 'hc':
                            self.crossheaders[int(n.split('hc')[1])].set_value(name, val[path])
                        elif n[0] == 'h':
                            self.headers[int(n.split('h')[1])].set_value(name, val[path])
        return

    def update_tube_mass_flow(self):
        tube_flow = [m / self.ntubes_per_panel for m in self.operating_conditions.mass_flow]  # Mass flow per tube per path
        header_flow = self.operating_conditions.mass_flow  # Mass flow per path
        self.update_tube_inputs({'mflow':tube_flow}, True)
        self.update_header_inputs({'mflow': header_flow}, True)
        return
        

    
    # Update operating conditions in tube instances based on current operating_conditions
    def update_tube_operating_conditions(self, is_update_flux = True):  
        skip = ['opteff', 'vwind10', 'rh', 'dni', 'dni_clearsky', 'day', 'tod', 'hour_offset', 'mass_flow']  # Conditions not used in tube class
        for k, val in vars(self.operating_conditions).items():
            if k in skip:
                continue
            #elif k == 'Tfin': # Only update in inlet panels
            #    inlet_panels = [path[0] for path in self.flow_paths]
            #    for p in inlet_panels:
            #        for t in range(self.ntubesim):
            #            self.tubes[p][t].set_value(k, self.operating_conditions.Tfin)
            elif k == 'inc_flux':
                if is_update_flux:
                    self.set_tube_flux()
            else:
                 self.update_tube_inputs({k:val})
                 self.update_header_inputs({k:val})
  
        return
    
 
    
    #=========================================================================
    # # Interpolate from supplied flux distribution to discretization points on all tubes
    # def calculate_tube_flux(self, rec_flux_dist):
        
    #     #--- Interpolate to axial resolution in self.disc.nz
    #     nz, nx = rec_flux_dist.shape
    #     zpts = np.arange(0.5/nz, 1.0, 1./nz)               # Axial points at flux profile resolution (SolarPILOT points are at interval centroids)
    #     zpts_tube = np.linspace(0, 1.0, self.disc.nz) + 0.5/nz  # Set flux at center of axial discretization elements -> Flux at last point is not meaningful but won't be used in solution because of forward differences 
    #     inc_flux_interp = np.zeros((self.disc.nz, nx)) 
    #     for z in range(self.disc.nz):
    #         inc_flux_interp[z,:] = util.interpolate1D(zpts_tube[z], zpts, rec_flux_dist) 
             
    #     #--- Calculate x-positions at tube locations
    #     xpts = np.zeros((self.Npanels, self.ntubesim))
    #     dxpan = 1./float(self.Npanels)
    #     xperpan = np.linspace(0, 1, self.ntubesim) * dxpan
    #     for j in range(self.Npanels):
    #         xpts[j,:] = (j+0.5)*dxpan if self.ntubesim == 1 else j*dxpan + xperpan    # Tube x-positions
        
    #     #--- Interpolate flux profiles to tube positions at receiver circumference
    #     qinc = np.zeros((self.disc.nz, self.Npanels, self.ntubesim))
    #     dx = 1./float(nx)
    #     for j in range(self.Npanels):
    #         for k in range(self.ntubesim):
    #             if xpts[j,k] < 0.5*dx:  # x-coordinate below first x-point in array (south side of receiver) -> interpolate using last point in array
    #                 qinc[:,j,k] = inc_flux_interp[:,0] + (inc_flux_interp[:,-1]-inc_flux_interp[:,0])/dx* (0.5*dx - xpts[j,k])
    #             elif xpts[j,k] > 1.0 - 0.5*dx:   # x-coordinate below first x-point in array -> interpolate using last point in array
    #                 qinc[:,j,k] = inc_flux_interp[:,-1] + (inc_flux_interp[:,0]-inc_flux_interp[:,-1])/dx * (xpts[j,k] - (1.0 - 0.5*dx))
    #             else:
    #                 i1 = int(floor((xpts[j,k]-0.5*dx)/dx))
    #                 qinc[:,j,k] = inc_flux_interp[:,i1] + (inc_flux_interp[:,i1+1]-inc_flux_interp[:,i1])/dx * (xpts[j,k] - (i1+0.5)*dx)
    #     return qinc
    
        # Interpolate from supplied flux distribution to discretization points on all tubes

# ------------------------- interpolate tube flux. Different from o.g. in that it does not place two adjacent xtubes at the same x position. They are now offset by half a tube width in either direction
    def calculate_tube_flux_jwenner(self, rec_flux_dist):
        
        #--- Interpolate to axial resolution in self.disc.nz
        nz, nx = rec_flux_dist.shape
        zpts = np.arange(0.5/nz, 1.0, 1./nz)               # Axial points at flux profile resolution (SolarPILOT points are at interval centroids)
        zpts_tube = np.linspace(0, 1.0, self.disc.nz) + 0.5/nz  # Set flux at center of axial discretization elements -> Flux at last point is not meaningful but won't be used in solution because of forward differences 
        inc_flux_interp = np.zeros((self.disc.nz, nx)) 
        for z in range(self.disc.nz):
            inc_flux_interp[z,:] = util.interpolate1D(zpts_tube[z], zpts, rec_flux_dist) 
             
        #--- Calculate x-positions at tube locations, but add a gap between edge tubes of two adjacent panels
        ntubes_total = self.ntubes_per_panel*self.Npanels
        gap = 1/ntubes_total # create a gap that is 1/ntubes_total fraction wide
        xpts = np.zeros((self.Npanels, self.ntubesim))
        dxpan = 1./float(self.Npanels)
        # xperpan = np.linspace(0, 1, self.ntubesim) * dxpan
        xperpan = np.linspace(gap, dxpan-gap, self.ntubesim)
        for j in range(self.Npanels):
            xpts[j,:] = (j+0.5)*dxpan if self.ntubesim == 1 else j*dxpan + xperpan    # Tube x-positions
        
        #--- Interpolate flux profiles to tube positions at receiver circumference
        qinc = np.zeros((self.disc.nz, self.Npanels, self.ntubesim))
        dx = 1./float(nx)
        for j in range(self.Npanels):
            for k in range(self.ntubesim):
                if xpts[j,k] < 0.5*dx:  # x-coordinate below first x-point in array (south side of receiver) -> interpolate using last point in array
                    qinc[:,j,k] = inc_flux_interp[:,0]
                    ## my receiver is flat so i just use the first value
                elif xpts[j,k] > 1.0 - 0.5*dx:   # x-coordinate below first x-point in array -> interpolate using last point in array
                    qinc[:,j,k] = inc_flux_interp[:,-1]
                    ## my receiver is flat so I just use the last value
                else:
                    i1 = int(floor((xpts[j,k]-0.5*dx)/dx))
                    qinc[:,j,k] = inc_flux_interp[:,i1] + (inc_flux_interp[:,i1+1]-inc_flux_interp[:,i1])/dx * (xpts[j,k] - (i1+0.5)*dx)
        return qinc
    
    # Set operating_conditions.inc_flux in each tube instance based on the receiver flux distribution currently in self.operating_conditions
    def set_tube_flux(self, verbose = False):
        rec_flux_dist = self.operating_conditions.inc_flux  # Receiver flux distribution
        tube_flux_dist = self.calculate_tube_flux_jwenner(rec_flux_dist) # Flux distributions in each tube (z, panel, tube)
        for j in range(self.Npanels):
            for k in range(self.ntubesim):
                # Apply flux to tube instance, reversing directionality if tube flow is from bottom to top  
                # Assumes flux distribution has top of receiver in first elements of array 
                self.tubes[j][k].operating_conditions.inc_flux = tube_flux_dist[:,j,k] if not self.tubes[j][k].flow_against_gravity else tube_flux_dist[::-1,j,k]

        #--- Compare total incident flux from native distribution and from interpolated distributions
        if verbose:
            qnative = rec_flux_dist.mean()
            qtubes = 0.0
            for j in range(self.Npanels):
                for k in range(self.ntubesim):
                    qtubes += self.tube_weights[k] * tube_flux_dist[:-1,j,k].mean() / self.Npanels
            print ('Avg. incident flux from native distribution = %.2f kW/m2'%qnative)
            print ('Avg. incident flux from tube distributions = %.2f kW/m2'%qtubes)
            
        return


    
    #=========================================================================
    # Combine results per tube into receiver total, total per panel, or total per path
    def calculate_aggregate_results(self, name, is_per_path = False, is_per_panel = False, time_index = None):
        
        # Solutions for tubes
        tube_solns = self.get_tube_solns(name, 'avg', time_index = time_index)
        if is_per_panel:
            total = tube_solns*self.ntubes_per_panel
        elif is_per_path:
            total = np.array([tube_solns[self.flow_paths[p]].sum() for p in range(self.npaths)]) * self.ntubes_per_panel
        else:
            total = tube_solns.sum() * self.ntubes_per_panel
            
        # Solutions for headers
        if not is_per_panel:
            for p in range(self.npaths):
                for j in range(len(self.flow_path_names[p])):
                    if self.flow_path_names[p][j][0] == 'h':
                        val = getattr(self.flow_path_order[p][j][0], name)
                        val = val[time_index] if self.options.is_transient else val
                        if is_per_path:
                            total[p] += val
                        else:
                            total += val
        return total
            

    #=========================================================================
    # Calculate average tube front-wall temperature for full receiver or per panel       
    def calculate_front_wall_T_avg(self, is_per_panel = False):
        nthetaq = int((self.Ntheta+1)/2)
        if not is_per_panel:
            avg = sum([self.tube_weights[k] * self.tubes[j][k].Tw[:,0:nthetaq,-1].mean() for j in range(self.Npanels) for k in range(self.ntubesim)]) / self.Npanels
        else:
            avg = np.zeros(self.Npanels)
            for p in range(self.Npanels):
                avg[p] = sum([self.tube_weights[k] * self.tubes[p][k].Tw[:,0:nthetaq,-1].mean() for k in range(self.ntubesim)])            
        return avg
    
    
    
    #=========================================================================
    # Set initial guess for mass flow
    def set_initial_mass_flow_guess(self, solve_simple_model = True, verbosity = 1):
        
        #--- Set initial guess for mass flow rates
        est_heat_loss = self.estimate_loss_coeff(hext = 10, dTwallavg = 100)
        Qest = np.zeros(self.npaths)
        self.mflow_per_path = np.zeros(self.npaths)
        for p in range(self.npaths):
            npan = len(self.flow_paths[p])
            avg_flux_inc = sum([self.tube_weights[k] * self.tubes[j][k].operating_conditions.inc_flux[:-1].mean() for j in self.flow_paths[p] for k in range(self.ntubesim)]) / npan
            Qest[p] = (avg_flux_inc * self.solar_abs * 1000 - est_heat_loss) * (npan * self.Wpan * self.H)  # Estimated heat into fluid (W)
            self.mflow_per_path[p] = Qest[p] / self.HTFcpavg / (self.Tfout_design - self.operating_conditions.Tfin)
            if not solve_simple_model and verbosity >=1:
                print ('Mass flow from initialization = %.2fkg/s'%(self.mflow_per_path.sum()))
        self.operating_conditions.mass_flow = self.mflow_per_path
        self.update_tube_mass_flow()
        
        #--- Solve for mass flow using simplified settings
        if solve_simple_model:
            start = timeit.default_timer()
            original_settings = {k:getattr(self.options, k) for k in ['crosswall_avg_k', 'use_full_rad_exchange', 'wall_detail']}
            new_settings = {'crosswall_avg_k':True, 'use_full_rad_exchange': self.options.use_full_rad_exchange, 'wall_detail': '1D' if original_settings['wall_detail'] in ['1D', '2D'] else '0D'}
            self.update_tube_inputs(new_settings)
            self.set_mass_flow_and_solve_steady_state(is_for_init = True, verbosity = 0)
            self.update_tube_inputs(original_settings)  # Revert back to original settings
            self.operating_conditions.mass_flow = self.mflow_per_path
            self.update_tube_mass_flow()
            if verbosity >=1:
                print ('Mass flow from initialization = %.2fkg/s, %d iterations'%(self.mflow_per_path.sum(), self.n_flow_iter))
                print('Time for initial guess = %.3fs'%(timeit.default_timer() - start))
            
        return
    
    #=========================================================================
    # Set initial guess for tube temperatures
    def set_initial_T_guess(self, verbosity = 1):
        start = timeit.default_timer()
        original_settings = {k:getattr(self.options, k) for k in ['crosswall_avg_k', 'use_full_rad_exchange', 'wall_detail']}
        new_settings = {'crosswall_avg_k':True, 'use_full_rad_exchange': self.options.use_full_rad_exchange, 'wall_detail': '1D' if original_settings['wall_detail'] in ['1D', '2D'] else '0D'}
        self.update_tube_inputs(new_settings)
        self.solve_steady_state_profiles(False, Ttol = 1.e-3, allow_initial_guess = False)
        self.update_tube_inputs(original_settings)
        if verbosity >=1:
            print('Time for initial guess = %.3fs'%(timeit.default_timer() - start))
        return    
    
    #=========================================================================
    # Get incident solar flux (used to populate incident flux when solution is unsuccessful)    
    def calculate_total_solar_incidence(self):
        for p in range(self.npaths):  
            for j in range(len(self.flow_path_names[p])):  # Consecutive elements in flow path
                elem = self.flow_path_order[p][j]
                for k in range(len(elem)):
                    qinc, qabs, elem[k].Qsinc, Qsabs = elem[k].calculate_tube_solar_flux_distributions()
        self.Qsinc = self.calculate_aggregate_results('Qsinc')
        return
    
    #==========================================================================
    # Solve for receiver SS temperature profiles, using currently stored flux profiles, ambient temperature, and wind speed
    def solve_steady_state_profiles(self, is_iterate_for_hext = False, Ttol = None, allow_initial_guess = True, verbosity = 0):

        self.update_tube_mass_flow()    

        #--- Initialize results
        self.Tfout_per_panel = np.zeros(self.Npanels)
        self.Tfout_per_path = np.zeros(self.npaths)
        self.hext_per_panel = np.zeros(self.Npanels)
        self.thermal_stress_fraction_of_allowable_max_per_panel = np.zeros(self.Npanels)

        is_soln_defined = min([len(self.tubes[j][0].Tw) for j in range(self.Npanels)]) > 0 and min([self.tubes[j][0].Tw.max() for j in range(self.Npanels)]) > 0
        if not is_soln_defined and allow_initial_guess:
            print ('Calculating initial guess for temperature solution')
            self.set_initial_T_guess(verbosity)
            is_soln_defined = True
        
        #--- Set temperature solution tolerance for tube iterations
        Ttol = self.Ttol_for_tube if Ttol is None else Ttol
        self.update_tube_inputs({'Ttol': Ttol})

        #--- Calculate initial guess for receiver-average wall temperature to used in calculating external convection coefficients
        Teval_forconv = 0.5*(self.Tfout_design + self.operating_conditions.Tfin) + 75
        Teval_natconv = np.zeros(self.Npanels)
        if is_soln_defined:
            Twavg = self.get_tube_solns('Tw', solntype ='avg', wall_spec ='front_outer')  # Average front wall temperature per panel
            Teval_forconv = Twavg.mean()
            Teval_natconv = Twavg

        niter = self.n_hext_iter_max if is_iterate_for_hext else 1
        soln_code = 0   # 0 = successful solution, 1 = Re under cutoff, 2 = Tube temperature solution failed to converge
        self.n_hext_iter = 0
        for m in range(niter):  # Iterations to converge external convection coefficients
        
            if soln_code>0:  # Stop iterations for hext evaluation temperature is Re is under cutoff or tubes haven't converged
                break
            
            self.n_hext_iter += 1
            #--- Loop over flow paths and solve tube temperatures
            for p in range(self.npaths):  
                
                if soln_code>0:  # Stop panel loop if Re under cutoff
                    break
                
                for j in range(len(self.flow_path_names[p])):  # Consecutive elements in flow path
                    name = self.flow_path_names[p][j]
                    elem = self.flow_path_order[p][j]
                    panel = int(name.split('p')[1]) if name[0] == 'p' else None
                    Tfin = self.operating_conditions.Tfin if j == 0 else self.get_outlet_temperature(p, j-1)
                    
                    # Calculate external convection coefficient
                    if panel is not None:
                        if m == 0 and not is_soln_defined:
                            Teval_natconv[panel] = Tfin
                        self.hext_per_panel[panel] = self.external_h(Teval_natconv[panel], Teval_forconv)  # External convection coefficient (W/m2/K)
    
                    # Solve for temperature profile of each simulated tube
                    for k in range(len(elem)):
                        if panel is not None:
                            elem[k].hext = self.hext_per_panel[panel]  
                        elem[k].operating_conditions.Tfin = Tfin    
                        elem[k].solve_steady_state(verbosity = verbosity-1)
                        if k < len(elem)-1:
                            elem[k+1].Tw = np.copy(elem[k].Tw)   # Use temperature of current tube as initial solution for next tube
                        
                        if panel is not None and elem[k].Re.min() < self.Re_cutoff:
                            soln_code = 1
                            print('Receiver solution stopped because Re < %d' % self.Re_cutoff)
                            break
                        
                        if panel is not None and not elem[k].converged:
                            soln_code = 2
                            print('Receiver solution stopped because tube temperature solution failed to converge')
                            break
                   
                    if soln_code>0:
                        break
   
                    if panel is not None:
                        self.Tfout_per_panel[panel] = self.get_outlet_temperature(p,j)  # Set outlet temperature from this panel  
                        
                    if j == len(self.flow_path_names[p])-1:  # Last element in flow path
                        self.Tfout_per_path[p] = self.get_outlet_temperature(p,j) 

            #--- Adjust evaluation temperatures for convection coefficients
            if niter > 1: 
                Twavg = self.get_tube_solns('Tw', solntype ='avg', wall_spec ='front_outer')  # Average front wall temperature per panel
                Teval_forconv_new = Twavg.mean()
                Teval_natconv_new = Twavg
                if verbosity == 1:
                    print ('Iteration %d: Change in forced conv. evaluation T = %.1fC, Max change in natural conv. evaluation T = %.1fC'% (m, (Teval_forconv_new - Teval_forconv), (Teval_natconv_new - Teval_natconv).max()))
                if np.abs(Teval_forconv_new - Teval_forconv)/Teval_forconv < self.Ttol_for_hext and  \
                    (np.abs(Teval_natconv_new - Teval_natconv)/Teval_natconv).max()<self.Ttol_for_hext:  # Iterations converged
                    break
                Teval_forconv = Teval_forconv_new
                Teval_natconv = Teval_natconv_new

        #--- Aggregate tube results over path and full receiver  
        if soln_code==0:
            self.Qfluid_per_path = self.calculate_aggregate_results('Qfluid', is_per_path = True)
            for k in ['Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qfluid']:
                setattr(self, k, self.calculate_aggregate_results(k))
            self.eta_therm = self.Qfluid / self.Qsabs 
            self.eta = self.Qfluid / self.Qsinc 
            
        else:  # Populate solution with incident solar flux
            self.calculate_total_solar_incidence()

        # for i,p in enumerate(list(range(self.npaths))):
        #     print('    Flow path %d: mass flow = %.3f kg/s, Flow path outlet T = %.2fC'%(p, self.operating_conditions.mass_flow[i], self.Tfout_per_path[i]-273.15))     


        return soln_code
    
    
    #==========================================================================
    # Solve for receiver SS temperature profiles and mass flow to achieve target outlet T using currently stored flux profiles, ambient temperature, and wind speed
    def set_mass_flow_and_solve_steady_state(self, is_for_init = False, use_current_flow_as_guess = False, verbosity = 0):   

        skip_initialization = is_for_init
        if use_current_flow_as_guess and len(self.mflow_per_path) == self.npaths and min(self.mflow_per_path>0.0):
            skip_initialization = True
            self.operating_conditions.mass_flow = np.copy(self.mflow_per_path)
        
        if not skip_initialization:
            self.set_initial_mass_flow_guess(solve_simple_model = True, verbosity = verbosity)
        

        mflow_per_path_prev, Tfout_per_path_prev, mtol_per_path = [np.zeros(self.npaths) for v in range(3)]
        converged = False
        stopped = False
        Ttol_for_tube = self.Ttol_for_tube
        
        start_time = timeit.default_timer()
        self.n_flow_iter = 0
        for m in range(self.n_flow_iter_max):
            self.n_flow_iter += 1
            #--- Set solution tolerance (let first iteration solve with target tolerance to permit convergence after 1 iteration)
            if m > 0:
                tube_Re = self.get_tube_solns('Re', 'min', solntype = 'min')  # Minimum Re for each panel (min over simulated tubes per panel)
                Ttol_for_tube = max(self.Ttol_for_tube, min(1e-3, 0.01*mtol_per_path.min()))    # Allow lower tube temperature tolerance during mass flow iterations, inner temperature iterations will converge with mass flow
                if tube_Re.min() < 3000:
                    Ttol_for_tube *=0.1  # Near-laminar flow conditions need better temperature tolerance for convergence
 
            #--- Solve for temperature profiles
            soln_code = self.solve_steady_state_profiles(is_iterate_for_hext = False, Ttol = Ttol_for_tube, allow_initial_guess = False, verbosity = verbosity) 
            tubes_converged = self.get_tube_solns('converged', 'min').min() == 1   # All tube solutions converged?  
            tubes_tolerance = self.get_tube_solns('tol', 'max').max()            
            
            if soln_code>0 or not tubes_converged:  # Stop mass flow iterations if Re is below cutoff or if tubes haven't converged
                print ('Stopping mass flow iterations')
                stopped = True
                break
            
            total_flow = self.mflow_per_path.sum()                          # Total mass flow rate (kg/s)
            mfract  = self.mflow_per_path / total_flow                      # Fraction of flow in each mass flow path
            overall_outlet_T = (self.Tfout_per_path*mfract).sum()            # Mass-weighted average receiver outlet temperature
            Tconv = np.abs(self.Tfout_per_path - self.Tfout_design)/self.Tfout_design      # Outlet temperature convergence tolerance

            #--- Print iteration results
            if verbosity > 0:
                max_tube_iter = self.get_tube_solns('iter', 'max').max()
                print ('Iteration %d: Mass flow = %.2f kg/s, Qfluid = %.3f MW, Max tube iter = %d,  Outlet T = %.2f C' %(m, total_flow, self.Qfluid*1.e-6, max_tube_iter, overall_outlet_T-273.15))
                if verbosity > 1 :
                    for p in range(self.npaths): 
                        print ('    Flow path %d: mass flow = %.3f kg/s, Flow path outlet T = %.2fC'%(p, self.mflow_per_path[p], self.Tfout_per_path[p]-273.15))     

            #--- Check for mass flow convergence
            if tubes_tolerance <= self.Ttol_for_tube and Tconv.max() <= self.Ttol_for_mflow:   # Tube temperature iterations converged to desired tolerance and outlet T converged
                converged = True
                if verbosity > 0:
                    print ('Mass flow solution converged in %d iterations. Elapsed time = %.4fs'%(m+1, timeit.default_timer() - start_time ))
                break
            
            #--- Calculate next guess for mass flow
            f = (self.Tfout_per_path - self.Tfout_design)
            if m == 0:  # Fractional mass flow update, limit to 5% 
                mflow_per_path_new = self.Qfluid_per_path / self.HTFcpavg / (self.Tfout_design-self.operating_conditions.Tfin)
                mfupdate = (mflow_per_path_new - self.mflow_per_path) / self.mflow_per_path
                mfupdate = np.minimum(0.05, np.abs(mfupdate)) * np.sign(f)
            else:
                dTdm = (self.Tfout_per_path - Tfout_per_path_prev) / (self.mflow_per_path - mflow_per_path_prev)
                mfupdate = -f / dTdm / self.mflow_per_path
                mfupdate[np.logical_and(f>0, mfupdate<0)] = 0.005   # Correct illogical mass flow updates
                mfupdate[np.logical_and(f<0, mfupdate>0)] = -0.005
                
                urfs = 0.9*np.ones(self.npaths)
                urfs[np.abs(mfupdate)>=0.1] = 0.1
                urfs[np.logical_and(np.abs(mfupdate)>=0.03, np.abs(mfupdate)<0.1)] = 0.5
                urfs[np.logical_and(np.abs(mfupdate)>=0.01, np.abs(mfupdate)<0.03)] = 0.8
                mfupdate *= urfs  
                if tube_Re.min() < 2300.:  # Limit size of mass flow updates to 10% for laminar flow conditions
                    mfupdate = min(0.1, abs(mfupdate)) * np.sign(mfupdate) 
                    
            mflow_per_path_new = (1.+mfupdate) * self.mflow_per_path
            mtol_per_path = np.abs(mfupdate)    # Fractional change in mass flow per path
            
            #--- Update values for next iteration
            mflow_per_path_prev = deepcopy(self.mflow_per_path)
            Tfout_per_path_prev = deepcopy(self.Tfout_per_path)     
            self.mflow_per_path = mflow_per_path_new 
            self.operating_conditions.mass_flow = self.mflow_per_path
            self.Tfout_per_path.fill(0.0)

            if m == self.n_flow_iter_max-1 and not is_for_init and not converged:
                 print ('Mass flow solution reached the maximum number of iterations. Largest deviation from target outlet temperautre is %.2fC' % (np.abs(self.Tfout_per_path - self.Tfout_design).max()))
        
        self.mass_flow_converged = converged
        self.stopped = stopped
        self.soln_code = soln_code  # 0 = successful, 1 = Re under cutoff, 2 = Tube solution failed to converge
        self.mflow = self.mflow_per_path.sum()
        
        #--- If necessary, solve for temperature solutions using designated mass flow control mode 
        if self.flow_control_mode != 0 and not is_for_init:
            if self.flow_control_mode == 1:
                self.mflow_per_path = self.mflow/self.npaths * np.ones(self.npaths)
            elif self.flow_control_mode == 2:
                self.mflow_per_path = max(self.mflow_per_path) * np.ones(self.npaths)
            self.mflow = self.mflow_per_path.sum()   
            self.operating_conditions.mass_flow = self.mflow_per_path
            self.solve_steady_state_profiles(is_iterate_for_hext = True, Ttol = self.Ttol_for_tube, allow_initial_guess = False, verbosity = verbosity) 
        return
    
    
    
    def solve_steady_state(self, calculate_mflow = True, use_current_flow_as_guess = False, verbosity = 0):
        self.update_tube_operating_conditions(True)
        
        if calculate_mflow:  # Calculate mass flow to achieve target outlet T
            self.set_mass_flow_and_solve_steady_state(False, use_current_flow_as_guess, verbosity = verbosity)
        else:  # Solve using current mass flow in self.mflow_per_path
            self.solve_steady_state_profiles(is_iterate_for_hext = True, verbosity = verbosity)

        if not self.stopped:  # Solution was successfully completed
            self.calculate_pressure_profiles()
            self.calculate_pumping_power()
            self.eta_pump = (self.Qfluid - self.Qpump) / self.Qsinc
            if self.options.calculate_stress:
                self.calculate_stresses()
                self.thermal_stress_fraction_of_allowable_max_per_panel = self.get_tube_solns('thermal_stress_fraction_of_allowable', combine_tubes = 'max', solntype = 'max')
                self.thermal_stress_fraction_of_allowable_max = self.thermal_stress_fraction_of_allowable_max_per_panel.max()
                self.pressure_stress_fraction_of_allowable_max = self.get_tube_solns('pressure_stress_fraction_of_allowable', combine_tubes = 'max', solntype = 'max').max()

        return 
    
    
    # Calculate min tube size for currently-defined operating conditions that can meet a velocity and pressure constraint
    def calculate_min_tube_size(self, velocity_constraint, pressure_constraint):
        
        #--- Initial guess for tube diameter
        self.initialize()
        est_heat_loss = self.estimate_loss_coeff(hext = 10, dTwallavg = 100) #guess that dT is 100 C... which is kinda low actually
        Qest = np.zeros(self.npaths)
        mflow_per_path = np.zeros(self.npaths)
        for p in range(self.npaths):
            npan = len(self.flow_paths[p])
            avg_flux_inc = sum([self.tube_weights[k] * self.tubes[j][k].operating_conditions.inc_flux[:-1].mean() for j in self.flow_paths[p] for k in range(self.ntubesim)]) / npan
            Qest[p] = (avg_flux_inc * self.solar_abs * 1000 - est_heat_loss) * (npan * self.Wpan * self.H)  # Estimated heat into fluid (W)
            mflow_per_path[p] = Qest[p] / self.HTFcpavg / (self.Tfout_design - self.operating_conditions.Tfin)

        mflow = mflow_per_path.max()
        rho = self.tubes[0][0].fluid.density(np.linspace(self.Tfin_design, self.Tfout_design, 25)).min() 
        OD = self.tube_OD
        for j in range(5):  # Iteratively update OD solution
            nperpanel = int(floor(self.Wpan / OD))
            OD = (4*(mflow / nperpanel) / np.pi / rho / velocity_constraint)**0.5 + 2*self.tube_twall  # Tube OD that would meet the velocity constraint at all locations

        #--- Find solutions that meet constraints
        def iterate(name, constraint, ODguess, tol):
            ODprev = None
            fprev = None
            OD = ODguess
            niter = 50
            for j in range(niter):
                self.tube_OD = OD
                self.initialize()
                self.solve_steady_state(verbosity = 0)
                vmax = self.get_tube_solns(name, combine_tubes = 'max', solntype = 'max').max()
                f = (vmax - constraint)
                if abs(f) < tol:
                    break
                if j == 0:
                    ODnew = 1.02*OD if f>0 else 0.98*OD
                else:
                    urf = 0.85
                    dfdOD = (f-fprev) / (OD-ODprev)
                    ODnew = OD - urf*(f/dfdOD)
                ODprev = OD
                fprev = f
                OD = ODnew
                
            if j == niter:
                print('Tube diameter solution failed to converged')
                
            return OD
        
        calc_stress = self.options.calculate_stress
        self.options.calculate_stress = False  # Turn off stress calculations for faster solutions
        OD = iterate('velocity', velocity_constraint, OD, 0.05)
        pmax = self.get_tube_solns('P', combine_tubes = 'max', solntype = 'max').max()
        if pmax > pressure_constraint+0.05:  # Tube diameter needs to be larger to meet pressure constraint
            ODp = iterate('P', pressure_constraint, 1.1*OD, 0.05)
            OD = max(OD, ODp)
        self.options.calculate_stress = calc_stress
            
        return OD
        
        
        
        
        
        
        
        
        
        

    
    #==========================================================================
    def initialize_transient_results(self):
        nstep = int(self.soln_time / self.time_step)  
        for k in ['Tfout', 'hext']:
            setattr(self, k+'_per_panel', np.zeros((self.Npanels, nstep)))
        for k in ['mflow', 'Qfluid', 'Tfout', 'pressure_drop']:
            setattr(self, k+'_per_path',  np.zeros((self.npaths, nstep)))
        for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qpump', 'mflow', 'eta_therm', 'eta', 'eta_pump', 'pressure_drop_with_tower']:
            setattr(self, k, np.zeros(nstep))
            
        if self.options.calculate_stress:
            self.max_stress_per_tube = np.zeros((self.Npanels, self.ntubesim, nstep))
        return
    
    
    #--- Set initial conditions in each tube instance
    def set_tube_initial_conditions(self, existing_solution = None, Tf = None, Tw = None):
        for p in range(self.npaths):
            for j in range(len(self.flow_path_names[p])):
                elems = self.flow_path_order[p][j]
                for k in range(len(elems)):
                    if existing_solution:
                        elems[k].Tf_initial_condition = np.copy(existing_solution.flow_path_order[p][j][k].Tf) 
                        elems[k].Tw_initial_condition = np.copy(existing_solution.flow_path_order[p][j][k].Tw) 
                    else:
                        elems[k].Tf_initial_condition = Tf
                        elems[k].Tw_initial_condition = Tw
                    
                    elems[k].initialize_transient_results()  # Call this after defining initial conditions
        return             
        
    #-------------------------------------------------------------------------
    #--- Set up transient operating conditions in each tube instance
    def set_up_transient_tube_operating_conditions(self):
        
        #--- Interpolate receiver flux distributions at each tabulated time point to tube flux distributions
        flux_times = []
        rec_flux_dist = []
        if isinstance(self.trans_operating_conditions.inc_flux, list) and len(self.trans_operating_conditions.inc_flux) == 2:
            flux_times = self.trans_operating_conditions.inc_flux[0]
            rec_flux_dist = self.trans_operating_conditions.inc_flux[1]
        else:
            flux_times = [0.0]
            rec_flux_dist = [self.trans_operating_conditions.inc_flux]
        tubes_flux_dist = []
        for t in range(len(flux_times)):  # Time points with defined flux distributions
            tubes_flux_dist.append(self.calculate_tube_flux(rec_flux_dist[t]))       # Tube flux distributions interpolated from receiver flux distribution
        
        #--- Set 'trans_operating_conditions' for each tube and header
        flow_times = self.trans_operating_conditions.mass_flow[0]
        mass_flow = self.trans_operating_conditions.mass_flow[1]
        for path in range(self.npaths):
            for name in self.flow_path_names[path]:
                is_panel = (name[0] == 'p')
                panel = int(name[1:]) if is_panel else None
                tubes = self.get_flow_path_element(name) if is_panel else [self.get_flow_path_element(name)]  # Parallel flow elements
                for j in range(len(tubes)):
                    
                    # Ambient conditions
                    for key in ['Tamb', 'Tambrad', 'is_interpolate_flux']:
                        setattr(tubes[j].trans_operating_conditions, key, getattr(self.trans_operating_conditions,key))  # Set tube class transient operating conditions to receiver class conditions
                                       
                    # Mass flow rates
                    tubes[j].trans_operating_conditions.mass_flow = [flow_times, []]
                    for t in range(len(flow_times)):
                        flow = mass_flow[t][path]/self.ntubes_per_panel if is_panel else mass_flow[t][path]
                        tubes[j].trans_operating_conditions.mass_flow[1].append(flow)
                        
                    # Incident flux
                    if is_panel:
                        tubes[j].trans_operating_conditions.inc_flux = [flux_times,[]]
                        for t in range(len(flux_times)):
                            tube_flux = tubes_flux_dist[t][:,panel,j] if not tubes[j].flow_against_gravity else tubes_flux_dist[t][::-1,panel,j]  # Reverse axial directionality if tube flow is from bottom to top  
                            tubes[j].trans_operating_conditions.inc_flux[1].append(tube_flux)
                    else:
                        tubes[j].trans_operating_conditions.inc_flux = 0.0
        return
                  

        
 
    #-------------------------------------------------------------------------
    def solve_transient(self,  verbosity = 0):   
        nstep = int(self.soln_time / self.time_step)  
        nper10 = int(0.1*self.soln_time / self.time_step)
        
        if self.parallel_tube_calculation:
            pool = multiprocessing.Pool(processes = self.nprocess)
        
        #--- Set initial conditions 
        if self.initial_condition in ['steady_state']:
            receiverSS = deepcopy(self)
            receiverSS.operating_conditions = self.trans_operating_conditions.get_operating_conditions_at_time(0)
            receiverSS.mflow_per_path = receiverSS.operating_conditions.mass_flow
            receiverSS.update({'is_transient':False, 'calculate_stress':False}, True, True)
            receiverSS.Ttol_for_tube = 1e-5
            receiverSS.reinitialize_tubes()  # Re-initialize to set up coefficient matricies
            receiverSS.solve_steady_state(False, verbosity = verbosity)  # Solve for steady state temperature profiles using mass flow for t = 0 operating condition
            self.set_tube_initial_conditions(receiverSS)
            receiverSS = None
        elif self.initial_condition == 'constant':  # Note mass flow must already be specified in self.mflow_per_path
            self.set_tube_initial_conditions(Tf = self.initial_constant_Tf, Tw = self.initial_constant_Tw)
        else:
            print('Initial condition specification was not recognized')
            return
            
        #--- Initialize receiver results
        self.initialize_transient_results()  
        self.Tfout_per_panel[:, 0] = self.get_tube_solns('Tf', combine_tubes ='avg', solntype ='outlet', time_index = 0) 
        

        start = timeit.default_timer()
        
        #-----------------------------------------------------------------------
        #--- Solve all time points per tube at once (better for parallelization, but requires fixed evaluation temperatures for external heat transfer coefficients)
        if self.options.is_transient_sequential_panel_soln:
            
            # Set up transient operating conditions in each tube instance 
            self.set_up_transient_tube_operating_conditions()
           
            # Evalute external heat transfer coefficients and mass flows for all time points
            Twavg = self.get_tube_solns('Tw', solntype ='avg', wall_spec ='front_outer', time_index = 0)  # Average front wall temperature per panel at first time step
            for t in range(nstep):
                time = t*self.time_step 
                self.operating_conditions = self.trans_operating_conditions.get_operating_conditions_at_time(time)   
                self.mflow_per_path[:,t] = self.operating_conditions.mass_flow  # Update mass flow results
                for panel in range(self.Npanels):
                    self.hext_per_panel[panel, t] = self.external_h(Twavg[panel], Twavg.mean())  # Panel external convection coefficient (W/m2/K)
                    

            # Solve for tube temperatures
            timepts = np.arange(nstep)*self.time_step
            Tfin = [timepts, np.zeros(nstep)]
            for p in range(self.npaths): 
                n = len(self.flow_path_names[p])
                for j in range(n):  # Consecutive elements in flow path
                    print('Solving path %d, element %d of %d. Elapsed time = %.2fs'%(p, j, n, timeit.default_timer()-start))
                    name = self.flow_path_names[p][j]
                    elem = self.flow_path_order[p][j]
                    panel = int(name.split('p')[1]) if name[0] == 'p' else None   
                    if j > 0:   # Populate inlet temperature vs. time based on exit from previous panel
                        for t in range(nstep):
                            Tfin[1][t] = self.get_outlet_temperature(p, j-1, t)
                    
                    # Solve for temperature profile of each simulated tube at time (t+1)
                    for k in range(len(elem)):
                        elem[k].trans_operating_conditions.Tfin = Tfin if j > 0 else self.trans_operating_conditions.Tfin
                        if panel is not None:
                            elem[k].trans_operating_conditions.hext = [timepts, self.hext_per_panel[panel, :]]
                        if not self.parallel_tube_calculation:
                            elem[k].solve_transient(initialize_soln = False, calculate_stress = False, verbosity = verbosity)

                    if self.parallel_tube_calculation:
                        inputs = [[tuberef, False, False, verbosity] for tuberef in elem]
                        ret = list(pool.map(tube.solve_tube_transient, tuple(inputs) ))
                        for k in range(len(elem)):
                            elem[k] = ret[k]

                            
                            
                            
                            
        #-----------------------------------------------------------------------
        #--- Solve time points sequentially, populating full receiver solution at each time point.  Allow time-dependent external h, but not good for paralellization
        else:
            inc_flux_prev = None
            for t in range(nstep-1):
                time = t*self.time_step     # Current time relative to first point in self.trans_operating_conditions
                if t%nper10 == 0:
                    print('%d%% complete. Elapsed time = %.2fs'%((time/self.soln_time)*100, timeit.default_timer()-start))
    
                #--- Update operating conditions and tube flux distributions
                self.operating_conditions = self.trans_operating_conditions.get_operating_conditions_at_time(time) # Update self.operating_conditions at this point in time   
                self.mflow_per_path[:,t] = self.operating_conditions.mass_flow  # Update mass flow results
                self.update_tube_mass_flow()
                update_tube_flux = (t == 0) or np.abs(self.operating_conditions.inc_flux - inc_flux_prev).max() > 0.1  # Should flux distributions be updated?
                self.update_tube_operating_conditions(update_tube_flux)      
                
                #--- Get average external tube temperatures at current time point
                Twavg = self.get_tube_solns('Tw', solntype ='avg', wall_spec ='front_outer', time_index = t)  # Average front wall temperature per panel at time t
    
                #--- Solve time point for each tube in the receiver
                for p in range(self.npaths): 
                    for j in range(len(self.flow_path_names[p])):  # Consecutive elements in flow path
                        name = self.flow_path_names[p][j]
                        elem = self.flow_path_order[p][j]
                        panel = int(name.split('p')[1]) if name[0] == 'p' else None
                        Tfin = self.operating_conditions.Tfin if j == 0 else self.get_outlet_temperature(p, j-1, t+1)
                        if panel is not None:
                            self.hext_per_panel[panel, t] = self.external_h(Twavg[panel], Twavg.mean())  # Panel external convection coefficient (W/m2/K)
                    
                        # Solve for temperature profile of each simulated tube at time (t+1)
                        for k in range(len(elem)):
                            if panel is not None:
                                elem[k].hext = self.hext_per_panel[panel,t] 
                            elem[k].operating_conditions.Tfin = Tfin   
                            #elem[k].trans_operating_conditions.Tfin = Tfin   
                            if not self.parallel_tube_calculation:
                                elem[k].solve_time_point(time, update_operating_conditions = False, calculate_stress = False, verbosity = verbosity)
    
                        if self.parallel_tube_calculation:
                            inputs = [[tuberef, time, False, False, verbosity] for tuberef in elem]
                            ret = list(pool.map(tube.solve_tube_time_point, tuple(inputs) ))
                            for k in range(len(elem)):
                                elem[k] = ret[k]
    
                #--- Store current flux profile (to compare with flux profile at next time point)
                inc_flux_prev = np.copy(self.operating_conditions.inc_flux)
                
     
     
        #---------------------------------------------------------------------
        # Populate derived solution quantities at each time step
        #start = timeit.default_timer()
        for t in range(nstep-1):
            for p in range(self.Npanels):
                self.Tfout_per_panel[p, t] = self.get_tube_solns('Tf', combine_tubes ='avg', panel = p, solntype ='outlet', time_index = t) 
            
            #--- Calculate and store aggregated results
            self.Qfluid_per_path[:,t] = self.calculate_aggregate_results('Qfluid', is_per_path = True, time_index = t)
            for k in ['Qfluid','Qsinc', 'Qsabs', 'Qconv', 'Qrad']:
                attr = getattr(self, k)
                attr[t] = self.calculate_aggregate_results(k, time_index = t)
            
        
        self.eta_therm = self.Qfluid / np.maximum(self.Qsabs, 1e-6)  # Note: last time point is not filled in
        self.eta = self.Qfluid / np.maximum(self.Qsinc, 1e-6)
        self.mflow = self.mflow_per_path.sum(0)
        outlet_panels = [path[-1] for path in self.flow_paths]
        self.Tfout_per_path[:,:] = self.Tfout_per_panel[outlet_panels,:] 

        #--- Pressure profiles and pumping power: Note, pressure profile per tube is calculated within the loop over time points, assuming an outlet pressure of zero.  Now just need to adjust
        self.shift_pressures()  
        self.set_pressure_drop_per_path()
        self.calculate_pumping_power()
        self.eta_pump = (self.Qfluid - self.Qpump) / np.maximum(self.Qsinc, 1e-6)
        #print('Time to populate derived solution quantities = %.2fs'%(timeit.default_timer()-start))
        
        #--- Elastic stress
        start = timeit.default_timer()
        if self.options.calculate_stress:
            self.calculate_stresses()  
        print('Time to calculate stress = %.2fs'%(timeit.default_timer()-start))
        

        return 
     
    

    #=========================================================================
    # Calculate pressure profiles
    
    # Shift pressure calculated in each tube (assuming an outlet pressure of zero) to full sequence of tubes
    def shift_pressures(self):
        for path in range(self.npaths):
            n = len(self.flow_path_names[path])
            outlet_pressure = 0.0 
            for j in range(n):
                i = n-1-j
                elem = self.flow_path_order[path][i]
                for k in range(len(elem)):
                    elem[k].P += outlet_pressure
                pin_avg = self.get_inlet_pressure(path, i)
                outlet_pressure = pin_avg
        return 

    def get_inlet_pressure(self, path, j): # path=flowpath, j=panel?
        name = self.flow_path_names[path][j]
        elem = self.flow_path_order[path][j]
        panel = int(name.split('p')[1]) if name[0] == 'p' else None
        shape = elem[0].P.shape
        pin_avg = 0.0 if len(shape) == 1 else np.zeros_like(elem[0].P[0,:])
        for k in range(len(elem)):
            pin = elem[k].P[0,...]
            if panel is not None:
                pin_avg += self.tube_weights[k] * pin # Average inlet pressure at this panel
            else:
                pin_avg += pin     
        return pin_avg

    def set_pressure_drop_per_path(self):
        self.pressure_drop_per_path = np.array([self.get_inlet_pressure(path,0) for path in range(self.npaths)])
        inlet_tube = self.tubes[self.flow_paths[0][0]][0]  
        Tfin = inlet_tube.Tf[0,...] 
        # self.pressure_drop_with_tower = self.pressure_drop_per_path.max(0) +  inlet_tube.fluid.density(Tfin)*9.8*self.Htower*1e-6
        self.pressure_drop_with_tower = [self.pressure_drop_per_path.max(0) +  inlet_tube.fluid.density(Tfin)*9.8*self.Htower*1e-6 for i in range(self.npaths)] # jwenner mod
        return
    
    # Calculate pressure profiles (only used for steady state calculations)
    def calculate_pressure_profiles(self):
        for path in range(self.npaths):
            n = len(self.flow_path_names[path])
            for j in range(n):
                elem = self.flow_path_order[path][j]      
                for k in range(len(elem)):
                     elem[k].calculate_pressure_profile(0.0)  # Sets pressure profile in tube class assuming an outlet pressure of zero         
        self.shift_pressures()  
        self.set_pressure_drop_per_path()
        return
    
    #=========================================================================
    # Pump power requirements
    def calculate_pumping_power(self):
        inlet_tube = self.tubes[self.flow_paths[0][0]][0]  
        Tfin = inlet_tube.Tf[0,...] 
        Tfout = (self.mflow_per_path * self.Tfout_per_path).sum(0) / np.maximum(self.mflow_per_path.sum(0), 1e-6)  # Mass-weighted avg exit T
        rho_avg = self.tubes[0][0].fluid.density((Tfin+Tfout)/2)
        
        pump_power = []
        for path in range(self.npaths):
            # dp = self.pressure_drop_per_path[path,...]*1e6 if not self.include_tower_in_Qpump else self.pressure_drop_with_tower[path,...]*1e6  # Pa
            dp = self.pressure_drop_per_path[path,...]*1e6 if not self.include_tower_in_Qpump else self.pressure_drop_with_tower[path]*1e6  # Pa (jwenner mod)
            pump_power.append((self.mflow_per_path[path,...] / rho_avg) * dp / self.pump_efficiency)
        total_pump_power = np.array(pump_power).sum(0)  # Total pump power (W)
        self.Qpump = total_pump_power / self.expected_cycle_eff  
        return 
    
    #=========================================================================
    # Calculate stress distributions
    def calculate_stresses(self):
        if not self.options.is_transient:
            self.max_stress_per_tube = np.zeros((self.Npanels, self.ntubesim))
        
        for p in range(self.Npanels):
            for k in range(self.ntubesim):
                self.tubes[p][k].calculate_stress()
                self.tubes[p][k].calculate_thermal_stress_relative_to_allowable()
                self.tubes[p][k].calculate_pressure_stress_relative_to_allowable()
                self.max_stress_per_tube[p,k,...] = self.tubes[p][k].stress_equiv.max(0).max(0).max(0)  # Max equivalent stress per tube     
                
        return 
               
    

