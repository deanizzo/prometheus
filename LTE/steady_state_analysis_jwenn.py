import numpy as np
import util
import operating_conditions
from analysis_jwenn import Analysis #need jwenn version of analysis to call solarpilot with new method
import steady_state_plots
"""
saved as steady_state_analysis_jwenn to attempt implementation of new solarpilot api
"""

class SteadyStateAnalysis(Analysis):
    def __init__(self, rec = None, field = None):   # Note: rec needs to be passed after rec.initialize() has been called 

        super().__init__(rec, field)

        self.analysis_mode = 'design_day'     # 'design_point', 'design_day', 'three_day', 'selected_days', 'user_defined'
        self.delta_hour = 1                   # Hourly spacing between evaluation points (not used if user-specified time points are provided)
        self.is_half_day = False              # Only simulate half-days (assuming symmetry in morning/afternoon hours)
        self.analysis_days = [172]            # Days to use in performance analysis (172 = summer solstice, 264 = equinox, 355 = winter solstice).  Will be set automatically for all but analysis_mode ='selected_days' or 'user_defined'
        self.analysis_time_points =  []       # Time points (relative to solar noon) used in the analysis (only for analysis_mode = 'user_defined', all others are set by 'delta_hour')

        self.dni_setting = 'user-defined-noon'  # Method to set DNI set per time point (clearsky, weatherfile, user-defined-noon)
        self.ambient_setting = 'user-defined'   # Method to set ambient conditions set per time point (weatheravg, weatherfile, or user-defined).  If weatheravg, hourly-average conditions will be calculated over the surrounding month of weather file data
        self.weather_avg_weeks = 4              # Number of weeks to use in average weather (half before, half after any given day)
        
        self.user_defined_dni = {172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
        self.user_defined_Tamb = 25+273.15    # Constant user-defined ambient T (K) (ignored unless ambient_setting = user-defined)
        self.user_defined_RH = 25             # Constant user-defined relative humidity (K) (ignored unless ambient_setting = user-defined)
        self.user_defined_vwind10 = 0         # Constant user-defined wind speed (m/s at 10m height) (ignored unless ambient_setting = user-defined)

        self.dni_cutoff = 250
        
        # Inputs related to re-sizing tube diameter to meet velocity or pressure constraint
        self.resize_tube_diameter = False  # Resize tube diameter to meet velocity or pressure constraint
        self.tube_OD_ratio = 1.0            # Ratio of tube OD to minimum computed tube OD that meets constraints
        self.tube_diameter_velocity_constraint = 4.0  # Max allowable velocity (m/s)
        self.tube_diameter_pressure_constraint = 2.5  # Max allowable pressure (m/s)
        
        #--- Time-series solution values
        self.results = self.SteadyStateResults()

        return


    class SteadyStateResults:
        def __init__(self):
            self.clear_solution()
            
        def clear_solution(self):
            
            self.soln_code = [] 
            
            #--- Results per time point
            self.day = []         # Day of year
            self.time = []        # Time (hour of day)
            self.time_offset = [] # Time relative to solar noon (hr)   
            self.dni = []         # DNI (W/m2)
            self.Tamb = []        # Ambient temperature (K)
            self.rh = []          # Relative humidity (%)
            self.vwind10 = []     # Wind speed at 10m (m/s)
            self.Tambrad = []     # Ambient temperature for radiation (K)

            self.Qfluid = []      # Total heat transfer into fluid (W)
            self.Qsinc = []       # Total incident solar energy (W)
            self.Qsabs = []       # Total absorbed solar energy (W)
            self.Qconv = []       # Total convection loss (W)
            self.Qrad = []        # Total IR radiation loss (W)
            self.Qpump = []       # Pump power (W)
            self.tol = []         # Solution temperature tolerance
            self.eta_field = []   # Field efficiency (including spillage loss, but not including reflection loss
            self.eta_therm = []   # Thermal efficiency (Qfluid / Qsabs)
            self.eta = []         # Receiver efficiency (Qfluid / Qsinc)
            self.eta_pump = []    # Receiver efficiency including pump loss ((Qfluid - Qpump) / Qsinc)
            self.eta_overall = []  # Receiver efficiency including pump loss and spillage loss ((Qfluid - Qpump) / Qfield)
            self.hext = []        # External heat loss coefficient (W/m2/K)
            self.pressure_drop = []  # Pressure drop (MPa)
            self.mflow = []       # Reciever mass flow (kg/s)
            
            self.peak_abs_flux = []         # Highest absorbed solar flux (kW/m2)
            self.stress_max = []            # Highest tube equivalent total stress (MPa)
            self.thermal_stress_max = []    # Highest tube equivalent thermal stress (MPa)
            self.pressure_stress_max = []   # Highest tube equivalent pressure stress (MPa)
            self.Tw_max = []                # Highest tube temperature (K)
            self.Re_max = []                # Highest Re 
            self.Re_min = []                # Min Re
            self.htube_max = []             # Highest internal h (W/m2/K)
            self.htube_min = []             # Lowest internal h (W/m2/K)
            self.htube_avg = []             # average internal h (W/m^2-K) shape:(time)
            self.velocity_max = []          # Highest HTF velocity (m/s)
            self.allow_thermal_stress_fraction_max = []  # Maximum (thermal stress / allowable thermal stress) anywhere on the receiver
            self.allow_pressure_stress_fraction_max = []  # Maximum (pressure stress / allowable pressure stress) anywhere on the receiver
            

            self.mflow_per_path = []  # Mass flow per path (kg/s)
            self.Tf = []            # Fluid T (K) (axial position, panel, tube, time)
            self.Tw = []            # Max wall T (K) (axial position, panel, tube, time)
            self.Tw_inner_low = []      # minimum wall T (K) (axial position, panel, tube, time)
            self.Tw_inner_high = []     # maximum INNER wall temp (K) (axial position, panel, tube, time)
            self.htubes = []        # heat transfer coefficients (W/m^2-K) (z,theta)
            self.abs_flux = []      # Peak absorbed flux (kW/m2) (axial position, panel, tube, time)        
            self.stress = []        # Max stress (MPa) per tube (axial position, panel, tube, time)
            self.allow_thermal_stress_fraction_prof = [] # (stress / allowable stress) for [axial position, panel, tube, time]
            
            
            #--- Time-averaged results, over all time points including those when the solution was not returned because of conditions below the miniminum Re or the minimum turndown ratio
            self.Qfluid_avg = 0.0       # Total heat transfer into fluid (W)
            self.Qsinc_avg = 0.0        # Total incident solar energy (W)
            self.Qsabs_avg = 0.0        # Total absorbed solar energy (W)
            self.Qconv_avg = 0.0        # Total convection loss (W)
            self.Qrad_avg = 0.0         # Total IR radiation loss (W)
            self.Qpump_avg = 0.0        # Pump power (W)
            self.eta_field_avg = 0.0    # Field efficiency (including spillage loss, but not including reflection loss
            self.eta_therm_avg = 0.0    # Thermal efficiency (Qfluid / Qsabs)
            self.eta_avg = 0.0          # Receiver efficiency (Qfluid / Qsinc)
            self.eta_pump_avg = 0.0     # Receiver efficiency including pump loss ((Qfluid - Qpump) / Qsinc)
            self.eta_overall_avg = 0.0  # Receiver efficiency including pump loss and spillage loss ((Qfluid - Qpump) / Qfield_inc)
            
            #--- Time-averaged results during only those points when the receiver was operating
            self.Qfluid_avg_op = 0.0       # Total heat transfer into fluid (W)
            self.Qsinc_avg_op = 0.0        # Total incident solar energy (W)
            self.Qsabs_avg_op = 0.0        # Total absorbed solar energy (W)
            self.Qconv_avg_op = 0.0        # Total convection loss (W)
            self.Qrad_avg_op = 0.0         # Total IR radiation loss (W)
            self.Qpump_avg_op = 0.0        # Pump power (W)
            self.eta_field_avg_op = 0.0    # Field efficiency (including spillage loss, but not including reflection loss
            self.eta_therm_avg_op = 0.0    # Thermal efficiency (Qfluid / Qsabs)
            self.eta_avg_op = 0.0          # Receiver efficiency (Qfluid / Qsinc)
            self.eta_pump_avg_op = 0.0     # Receiver efficiency including pump loss ((Qfluid - Qpump) / Qsinc)
            self.eta_overall_avg_op = 0.0  # Receiver efficiency including pump loss and spillage loss ((Qfluid - Qpump) / Qfield_inc)
            
            
            #--- Time maximum results
            self.pressure_drop_tmax = 0.0           # Pressure drop (MPa)
            self.peak_abs_flux_tmax = 0.0           # Highest absorbed solar flux (kW/m2)
            self.stress_max_tmax = 0.0              # Highest tube equivalent total stress (MPa)
            self.thermal_stress_max_tmax = 0.0      # Highest tube equivalent thermal stress (MPa)
            self.pressure_stress_max_tmax = 0.0     # Highest tube equivalent pressure stress (MPa)
            self.Tw_max_tmax = 0.0                  # Highest tube temperature (K)
            self.velocity_max_tmax = 0.0            # Highest HTF velocity (m/s)
            self.htube_max_tmax = 0.0               # Highest tube heat transfer coefficient (W/m2/K)
            self.allow_thermal_stress_fraction_max_tmax = 0.0  # Maximum (stress / allowable stress) anywhere on the receiver
            self.allow_pressure_stress_fraction_max_tmax = 0.0  # Maximum (stress / allowable stress) anywhere on the receiver
            
            
            #--- Design point result
            self.Qfluid_des = 0.0       # Total heat transfer into fluid (W)
            self.Qsinc_des = 0.0        # Total incident solar energy (W)
            self.Qsabs_des = 0.0        # Total absorbed solar energy (W)
            self.Qconv_des = 0.0        # Total convection loss (W)
            self.Qrad_des= 0.0          # Total IR radiation loss (W)
            self.Qpump_des = 0.0        # Pump power (W)
            self.eta_field_des = 0.0    # Field efficiency (including spillage loss, but not including reflection loss
            self.eta_therm_des = 0.0    # Thermal efficiency (Qfluid / Qsabs)
            self.eta_des = 0.0          # Receiver efficiency (Qfluid / Qsinc)
            self.eta_pump_des = 0.0     # Receiver efficiency including pump loss ((Qfluid - Qpump) / Qsinc)
            self.eta_overall_des = 0.0  # Receiver efficiency including pump loss and spillage loss ((Qfluid - Qpump) / Qfield_inc)
            self.velocity_max_des = 0.0   # Maximum design point velocity (m/s)
            self.pressure_drop_des = 0.0  # Design point pressure drop (MPa)
            self.allow_thermal_stress_fraction_max_des = 0.0  # Maximum (stress / allowable stress) anywhere on the receiver
            self.allow_pressure_stress_fraction_max_des = 0.0  # Maximum (stress / allowable stress) anywhere on the receiver
            
            
            
            #--- Time-integrated results 
            self.Qfluid_tot = 0.0                # Total heat transfer into fluid (W)
            self.Qfluid_pump_tot  = 0.0          # Total heat transfer into fluid less pumping loss (W)
            
            #--- Design parameters that are (optionally) updated during solution
            self.tube_OD = 0.0
            self.receiver_diameter = 0.0

            #--- 
            self.Ndisable = 0                   # number of disabled heliostats
            return
        

    #=========================================================================
    # Update inputs in this class or nested classes from a dictionary of parameter names
    def update_inputs(self, params):
        for key, val in params.items():
            if key in vars(self).keys():
                setattr(self, key, val)   
            elif key in vars(self.receiver).keys():
                setattr(self.receiver, key, val)   
            elif key in vars(self.SF).keys():
                setattr(self.SF, key, val)   
        self.receiver.initialize()
        return  
       
    # Get current input value, either from this class or nested classes
    def get_input(self, name):
        val = None
        if name in vars(self).keys():
            val = getattr(self, name)
        elif name in vars(self.receiver).keys():
            val = getattr(self.receiver, name)
        elif name in vars(self.SF).keys():
            val = getattr(self.SF, name)
        return val
            
        
    
    #=========================================================================
    def initialize(self):
        self.receiver.initialize()
        self.receiver.helio_height = self.SF.helio_height
        self.SF.solar_resource_file = self.weather_file
        self.operating_conditions.Tfin = self.receiver.Tfin_design
        self.receiver.site = util.read_weatherfile_header(self.weather_file)
        
        # Update resolution of flux profile computation to the minimum required
        self.delta_flux_hrs = max(self.delta_flux_hrs, self.delta_hour)
        if self.analysis_mode == 'design_point':
            self.n_flux_days = 2
            self.delta_flux_hrs = 10
            self.analysis_days = [172]
        elif self.analysis_mode == 'design_day':
            self.n_flux_days = 2
            self.analysis_days = [172]
        elif self.analysis_mode == 'equinox_day':
            self.n_flux_days = 3
            self.analysis_days = [264]
        elif self.analysis_mode == 'winter_day':
            self.n_flux_days = 2
            self.analysis_days = [355]
        elif self.analysis_mode == 'three_day':
            self.n_flux_days = 3
            self.analysis_days = [172, 264, 355]
        elif self.analysis_mode in ['selected_days', 'user_defined']:
            self.n_flux_days = 8
        else:
            print ('Analysis mode is not recognized')
            
        return
    

    #=========================================================================
    # Set DNI, ambient T, relative humidity based on model settings.  Must call this after setting day/time in operating_conditions
    def set_weather(self):
        doy = self.operating_conditions.day
        hour_of_day = self.operating_conditions.tod
        
        #--- Read weather data (if needed)
        if self.dni_setting == 'weatherfile' or self.ambient_setting == 'weatherfile':
            oc_point = operating_conditions.OperatingConditions()
            oc_point.set_from_weather_file(self.weather_file, doy, hour_of_day, wf_steps_per_hr = self.weather_steps_per_hour, useavg = False)
        if self.ambient_setting == 'weatheravg':
            oc_avg = operating_conditions.OperatingConditions()
            oc_avg.set_from_weather_file(self.weather_file, doy, hour_of_day, wf_steps_per_hr = self.weather_steps_per_hour, useavg = True, avghrs = self.weather_avg_weeks*7*24)
            
        #--- Set DNI    
        if self.dni_setting == 'clearsky':
            self.operating_conditions.dni = util.calculate_clearsky_DNI(self.receiver.site, doy, hour_of_day)
        elif self.dni_setting == 'weatherfile':
            self.operating_conditions.dni =  oc_point.dni
        elif self.dni_setting == 'user-defined-noon':
            clearsky = util.calculate_clearsky_DNI(self.receiver.site, doy, hour_of_day)
            if doy in self.user_defined_dni.keys():
                clearsky_noon = util.calculate_clearsky_DNI(self.receiver.site, doy, hour_offset = 0)
                self.operating_conditions.dni = clearsky * (self.user_defined_dni[doy] / clearsky_noon)      
            else:
                print('User-defined DNI at solar noon was not specified. Using calculated clear-sky DNI directly')
                self.operating_conditions.dni = clearsky
        else:
            print("Specified 'dni_setting' is not recognized")

        #--- Set Tamb, wind speed
        if self.ambient_setting == 'weatherfile':
            for k in ['Tamb', 'RH', 'wspd']:
                setattr(self.operating_conditions, k, getattr(oc_point, k))
        elif self.ambient_setting == 'weatheravg':
            for k in ['Tamb', 'RH', 'wspd']:
                setattr(self.operating_conditions, k, getattr(oc_avg, k))
        elif self.ambient_setting == 'user-defined':
            self.operating_conditions.Tamb = self.user_defined_Tamb
            self.operating_conditions.rh = self.user_defined_RH   
            self.operating_conditions.vwind10 = self.user_defined_vwind10 
        else:
            print("Specified 'ambient_setting' is not recognized")
        
        self.operating_conditions.Tambrad = util.calculate_sky_temperature(self.operating_conditions.Tamb, self.operating_conditions.rh, self.operating_conditions.hour_offset)  # Set sky temperature for radiative loss

        return
    
    #=========================================================================
    # Get time points with sufficient DNI to attempt
    def find_allowable_time_points(self):
        days = []
        times = []
        times_offset = []
        for doy in self.analysis_days:
            if self.analysis_mode != 'user_defined':
                # Set possible time points (conservative at beginning/end of day)
                time_pt_bounds = [-8,10] if not self.is_half_day else [0,10]
                time_pts_rel = np.array([0], dtype = int) if self.analysis_mode == 'design_point' else np.arange(time_pt_bounds[0], time_pt_bounds[1], self.delta_hour)
                self.analysis_time_points = time_pts_rel
                time_pts = util.get_hour_of_day(self.receiver.site, doy, time_pts_rel)
            else:
                time_pts = [util.get_hour_of_day(self.receiver.site, doy, t) for t in self.analysis_time_points]
                
            for p in time_pts:
                self.operating_conditions.day = doy
                self.operating_conditions.tod = p
                hour_offset = util.get_offset(self.receiver.site, doy, p)
                self.operating_conditions.hour_offset = hour_offset
        
                self.set_weather()
                if self.operating_conditions.dni >= self.dni_cutoff:  
                    days.append(doy)
                    times.append(p)
                    times_offset.append(hour_offset)
                
        return days, times, times_offset
                    

    
    def populate_operating_conditions(self, doy, hour_of_day):
        self.operating_conditions.day = doy
        self.operating_conditions.tod = hour_of_day
        hour_offset = util.get_offset(self.receiver.site, doy, hour_of_day)
        self.operating_conditions.hour_offset = hour_offset
        self.set_weather()
        inc_flux, opteff = self.SF.get_flux_profile_wCP(doy, self.operating_conditions.dni, hour_of_day = hour_of_day)
        # print('---using get_flux_profile_wlayoutSubset method rn---')
        # inc_flux, opteff, field_IDs = self.SF.get_flux_profile_wlayoutSubset(doy, hour_of_day = hour_of_day)

        ## commented out because my copylot output is not in terms of nominal dni.There is no reason to convert to actual operating conditions. Check this later?
        # self.operating_conditions.inc_flux = np.array(inc_flux)*self.operating_conditions.dni / self.SF.dni_des

        self.operating_conditions.inc_flux = np.array(inc_flux)
        self.operating_conditions.opteff = opteff
        self.receiver.operating_conditions = self.operating_conditions
        

        return
    

    #=========================================================================
    # Solve receiver model at a given day-of-year and hour of day
    def solve_time_point(self, doy, hour_of_day, verbosity = 0):

        self.receiver.clear_solution()
        code = 0  # 0 = Solution was successful, 1 = Solution mass flow below min turndown, 2 = Solution stopped for low Re, 3 = receiver mass flow did not converge, 4 = Solution not attempted because of low DNI, 
        self.populate_operating_conditions(doy, hour_of_day)
        if verbosity > 0:
            print('\nSolving day = %d, time = %.2f h, DNI = %.0f'%(doy, hour_of_day, self.operating_conditions.dni))
        
        if self.operating_conditions.dni < self.dni_cutoff:  
            if verbosity > 0:
                print('Stopping solution for low DNI')
            code = 4
            return code

        self.receiver.solve_steady_state(verbosity = verbosity)
        
        if self.receiver.Qfluid*1e-6 < self.receiver.min_turndown * self.receiver.Qdes:
            code = 1
        elif self.receiver.soln_code == 1:  # RE under cutoff
            code = 2
        elif self.receiver.soln_code == 2: # Tube temperature solutions failed to converge
            code = 3
        elif not self.receiver.mass_flow_converged:
            code = 4
            
        return code
    
    
    #=========================================================================
    # Solve receiver model at all time points in the analysis
    def get_min(self, name_in_tube):
        return self.receiver.get_tube_solns(name_in_tube,'min',solntype = 'min').min()
    
    def get_max(self, name_in_tube):
        return self.receiver.get_tube_solns(name_in_tube,'max',solntype = 'max').max()

    def get_mean(self, name_in_tube):
        return self.receiver.get_tube_solns(name_in_tube,'avg', solntype = 'avg',wall_spec='all_inner').mean()
    
    def solve(self, display_soln = True, verbosity = 0):
        noon = self.find_design_point()
        self.results.clear_solution()
        self.initialize()
        self.calculate_flux_distributions(verbose = verbosity>0)
        
        #--- Find days/times to include in the analysis (DNI > self.dni_cutoff) and initialize results
        days, times, times_offset = self.find_allowable_time_points()
        npts = len(days)
        for k in vars(self.results).keys():
            if type(getattr(self.results,k)) == type([]):
                setattr(self.results, k, np.zeros(npts))
        self.results.mflow_per_path = np.zeros((self.receiver.npaths,npts))      
        shape = [self.receiver.disc.nz, self.receiver.Npanels, self.receiver.ntubesim]
        self.results.Tf, self.results.Tw, self.results.Tw_inner_low , self.results.Tw_inner_high , self.results.htubes, self.results.abs_flux, self.results.stress, self.results.allow_thermal_stress_fraction_prof = [np.zeros(tuple(shape+[npts])) for v in range(8)]
        

        #--- Adjust tube size at design point
        if self.resize_tube_diameter:
            dp = np.where(np.logical_and(np.abs(np.array(days)-172)<0.5, np.abs(times_offset)<0.1))[0][0]    # Design point index
            self.populate_operating_conditions(days[dp], times[dp])
            ODmin = self.receiver.calculate_min_tube_size(self.tube_diameter_velocity_constraint, self.tube_diameter_pressure_constraint)
            print('Tube OD reset to %.4f m'%(ODmin*self.tube_OD_ratio))
            self.receiver.tube_OD = ODmin*self.tube_OD_ratio
            self.initialize()


               
        #--- Solve each time point 
        for p in range(npts):
            doy = days[p]
            time = times[p]
            time_offset = times_offset[p]
            ret = self.solve_time_point(doy, time, verbosity)  
            
            # Get results that only require DNI above cutoff and don't require a successful solution
            if ret<4:
                self.results.soln_code[p] = ret
                self.results.day[p] = doy
                self.results.time[p] = time
                self.results.time_offset[p] = time_offset
                for k in ['dni', 'Tamb', 'rh', 'vwind10']:
                    getattr(self.results,k)[p] = getattr(self.operating_conditions, k)
                self.results.Qsinc[p] = self.receiver.Qsinc
                self.results.eta_field[p] = self.operating_conditions.opteff 
                
            # Get results that require receiver solution to have completed
            if ret == 0:  
                # Single-valued solution results
                for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qpump', 'tol', 'eta_therm', 'eta', 'eta_pump', 'mflow']:
                    getattr(self.results,k)[p] = getattr(self.receiver, k)
                
                self.results.eta_overall[p] = self.receiver.Qfluid / (self.receiver.Qsinc/self.operating_conditions.opteff)                        
                self.results.Tambrad[p] = util.calculate_sky_temperature(self.results.Tamb[p], self.results.rh[p], self.results.time_offset[p])
                self.results.hext[p] = self.receiver.hext_per_panel.mean()
                self.results.pressure_drop[p] = self.receiver.pressure_drop_per_path.max()
                self.results.peak_abs_flux[p] = self.get_max('inc_flux') * self.receiver.solar_abs

                for k in ['Tw', 'Re', 'htube', 'velocity']:
                    getattr(self.results, k+'_max')[p] = self.get_max(k)
                for k in ['Re', 'htube']:
                    getattr(self.results, k+'_min')[p] = self.get_min(k)    
                
                getattr(self.results, 'htube_avg' )[p] = self.get_mean('htube')
                
                if self.receiver.options.calculate_stress:
                    self.results.stress_max[p] = self.get_max('stress_equiv')
                    self.results.thermal_stress_max[p] = self.get_max('max_thermal_stress_equiv')
                    self.results.pressure_stress_max[p] = self.get_max('max_pressure_stress_equiv')
                    self.results.allow_thermal_stress_fraction_max[p] = self.receiver.thermal_stress_fraction_of_allowable_max
                    self.results.allow_pressure_stress_fraction_max[p] = self.receiver.pressure_stress_fraction_of_allowable_max
                    
                # Solution profiles
                self.results.mflow_per_path[:,p] = self.receiver.mflow_per_path
                self.results.Tf[...,p] = self.receiver.get_array_of_axial_profiles('Tf')
                self.results.Tw[...,p] = self.receiver.get_array_of_axial_profiles('Tw')
                self.results.Tw_inner_low[...,p] = self.receiver.get_array_of_Tinner_low_axial_profiles('Tw') # this only works with attribute 'Tw'
                self.results.Tw_inner_high[...,p] = self.receiver.get_array_of_Tinner_high_axial_profiles('Tw') # this only works with attribute 'Tw'
                self.results.htubes[...,p] = self.receiver.get_array_of_axial_profiles('htube')
                self.results.abs_flux[...,p] = self.receiver.get_array_of_axial_profiles('inc_flux')*self.receiver.solar_abs
                if self.receiver.options.calculate_stress:
                    self.results.stress[...,p] = self.receiver.get_array_of_axial_profiles('stress_equiv')
                    self.results.allow_thermal_stress_fraction_prof[...,p] = self.receiver.get_array_of_axial_profiles('thermal_stress_fraction_of_allowable')
                 
        self.populate_time_averaged_solution()
        self.populate_design_point_solution()
        self.populate_time_integrated_solution()
        
        self.results.tube_OD = self.receiver.tube_OD
        self.results.receiver_diameter = self.receiver.D
        self.results.Ndisable = self.SF.Ndisable

        if display_soln:
            self.display_soln()

        return   
    
    def find_design_point(self):
        if 172 in self.results.day and 0 in self.results.time_offset:  # Design point included in the simulations
            d = np.intersect1d(np.where( self.results.day==172)[0], np.where( self.results.time_offset==0)[0])[0]
        elif 0 in  self.results.time_offset:
            d = np.where( self.results.time_offset==0)[0][0]           
        else:
            d = 0
        return d
    
    def populate_time_averaged_solution(self):
        
        #--- Time average over all points, including those when the receiver is not operating
        npt = len(self.results.Qsinc)
        for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qpump']:  # Time-average
            setattr(self.results, k+'_avg', getattr(self.results, k).sum()/npt)
        for k in ['pressure_drop', 'peak_abs_flux', 'stress_max', 'thermal_stress_max', 'pressure_stress_max', 'Tw_max', 'velocity_max', 'htube_max', 'allow_thermal_stress_fraction_max', 'allow_pressure_stress_fraction_max']:  # Max over time
            setattr(self.results, k+'_tmax', getattr(self.results, k).max())            
            
        self.results.eta_therm_avg = self.results.Qfluid.sum()/ self.results.Qsabs.sum()
        self.results.eta_avg = self.results.Qfluid.sum() / self.results.Qsinc.sum()
        self.results.eta_pump_avg = (self.results.Qfluid.sum() - self.results.Qpump.sum()) / self.results.Qsinc.sum()
        Qfield = self.results.Qsinc / self.results.eta_field  # Field efficiency = Power delivered to receiver / Power incident on field 
        self.results.eta_field_avg = self.results.Qsinc.sum() / Qfield.sum()
        self.results.eta_overall_avg = self.results.Qfluid.sum() / Qfield.sum()
        
        #--- Time average over only those points when the receiver is operating
        pts = np.where(self.results.Qfluid>0.0)[0]
        npt = len(pts)
        for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qpump']:  # Time-average
            setattr(self.results, k+'_avg_op', getattr(self.results, k)[pts].sum()/npt)
        self.results.eta_therm_avg_op = self.results.Qfluid[pts].sum()/ self.results.Qsabs[pts].sum()
        self.results.eta_avg_op = self.results.Qfluid[pts].sum() / self.results.Qsinc[pts].sum()
        self.results.eta_pump_avg_op = (self.results.Qfluid[pts].sum() - self.results.Qpump[pts].sum()) / self.results.Qsinc[pts].sum()
        self.results.eta_field_avg_op = self.results.Qsinc[pts].sum() / Qfield[pts].sum()
        self.results.eta_overall_avg_op = self.results.Qfluid[pts].sum() / Qfield[pts].sum()
        return
    
    def populate_design_point_solution(self):
        d = self.find_design_point()
        for k in ['Qfluid', 'Qsinc', 'Qsabs', 'Qconv', 'Qrad', 'Qpump', 'eta_field', 'eta_therm', 'eta', 'eta_pump', 'eta_overall', 
                  'velocity_max', 'pressure_drop', 'allow_thermal_stress_fraction_max', 'allow_pressure_stress_fraction_max']:
            setattr(self.results, k+'_des', getattr(self.results, k)[d])         
        return
    

    
    def populate_time_integrated_solution(self):
        self.results.Qfluid_tot = self.results.Qfluid.sum()
        self.results.Qfluid_pump_tot = self.results.Qfluid.sum() - self.results.Qpump.sum()
        return 
    
    
    def display_soln(self):
        d = self.find_design_point()
        pts = np.where(self.results.Qfluid>0.0)[0] # Points with receiver operation
        print(' ')
        print('Peak absorbed flux (kW/m2): Design = %.0f, Overall = %.0f'% (self.results.abs_flux[...,d].max(), self.results.abs_flux.max()))
        print('Peak wall T (C): Design = %.0f, Overall = %.0f'% (self.results.Tw_max[d]-273.15, self.results.Tw_max_tmax-273.15))
        print('Peak thermal stress (MPa): Design = %.0f, Overall = %.0f'% (self.results.thermal_stress_max[d], self.results.thermal_stress_max_tmax))
        print('Peak pressure stress (MPa): Design = %.0f, Overall = %.0f'% (self.results.pressure_stress_max[d], self.results.pressure_stress_max_tmax))
        print('Pressure drop (MPa): Design = %.2f, Overall = %.2f'% (self.results.pressure_drop[d], self.results.pressure_drop_tmax))

        print('Receiver and field efficiency: Design %.4f, Avg = %.4f, Avg during operation = %.4f'%(self.results.eta_overall[d], self.results.eta_overall_avg, self.results.eta_overall_avg_op))
        print('Receiver efficiency: Design = %.4f, Avg = %.4f, Avg during operation = %.4f'%(self.results.eta[d], self.results.eta_avg, self.results.eta_avg_op))
        print('   Reflection loss: Design = %.4f'%(1-self.results.Qsabs[d]/self.results.Qsinc[d]))
        print('   IR radiation loss: Design = %.4f'%(self.results.Qrad[d]/self.results.Qsinc[d]))        
        print('   Convection loss: Design = %.4f'%(self.results.Qconv[d]/self.results.Qsinc[d])) 
        print('   Pump loss: Design = %.4f'%(self.results.Qpump[d]/self.results.Qsinc[d])) 
        # print('design point htc (max) [W/m^2-K] = %.4f, (min) = %.4f' % (self.results.htube_max[d],self.results.htube_min[d]) )
        print('design point average thermal efficiency = %.4f' % self.results.eta_therm[d])
        print('design point avg htc [W/m^2-K] = %.4f' % self.results.htube_avg[d] )

        return
    
    

                   

                        
