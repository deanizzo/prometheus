
import numpy as np
import util


class OperatingConditions:
    def __init__(self):
        
        # Single-valued parameters
        self.day = float('nan')             # Day of year at start of simulation
        self.tod = float('nan')             # Hour of day at start of simulation
        self.hour_offset = float('nan')     # Hour of day (relative to solar noon) at start of simulation
        
        # Parameters that can be supplied as either a single value (or single array for inc_flux), or as a set of conditions at various time points
        # Conditions at time points must be specified as: [[array of time points relative to start of simulation (s)], [array of operating condition values]]
        self.dni = float('nan')             # DNI (W/m2) 
        self.dni_clearsky  = float('nan')   # Clearsky DNI (W/m2)
        self.Tamb = float('nan')            # Ambient temperature for convective loss (K)
        self.Tambrad = float('nan')         # Ambient temperature for radiative loss (K)
        self.rh = float('nan')              # Relative humidity (%)
        self.vwind10 = float('nan')         # Wind speed at 10m (m/s)     
        self.Tfin = float('nan')            # Receiver (or tube) inlet temperature
        self.opteff = float('nan')          # Field optical efficiency (not including receiver reflection loss)
        self.inc_flux = []                  # Receiver or tube incident flux profile (kW/m2): Receiver is (axial position, circumferential position), Tube is (axial position)

        self.is_interpolate_weather = False   # Linearly interpolate weather between conditions at provided time point (only used for transient analysis).  If False, the specified time point prior to the given solution time will be used
        self.is_interpolate_flux = False       # Linearly interpolate provided incident flux profiles between conditions at provided time points.  If False, the specified time point prior to the given solution time will be used
    
        self.mass_flow = float('nan')

    
    def set_from_weather_file(self, weatherfile, doy, tod, duration = 0, wf_steps_per_hr = 1, useavg = False, avghrs = 7*24):
        '''
        Read data from weather file for given day of year (doy) and time of day (tod)
        duration = time duration (hours) of points to include (for transient operating conditions)
        if useavg = False: Data will be read for closest time points and interpolated linearly if necessary
                    True: Data will be averaged over 'avghrs' centered around the designated time point
        '''

        wfstep = 1./wf_steps_per_hr   # Weather file time step (hr)
        toy = (doy*24 + tod)  # Time of year (hr)
        r = int(toy * wf_steps_per_hr)  # Closest row in weather file       
        if not useavg:  
            if duration == 0:
                weather = util.read_weather(weatherfile, rows = [r, r+1])
                weather = {k: weather[k][0] + (weather[k][1]-weather[k][0])/wfstep * (toy - r*wfstep) for k in weather.keys()}  # Interpolate between closest points
            else:
                nrows = int(duration * wf_steps_per_hr) + 1
                weather = util.read_weather(weatherfile, rows = np.arange(r, r+nrows+1))
 
        else:
            offset = int(0.5*avghrs * wf_steps_per_hr)  # offset on either side of designated time point
            weather = util.read_weather(self.weather_file, [max(0,r-offset), min(8760*wf_steps_per_hr,r+offset)])
            inds = np.where(weather['DNI']>0.0)[0]  # only average over daylight periods
            weather = {k:weather[k][inds].mean() for k in weather.keys()}     
        
        self.day = doy
        self.tod = tod
        
        if duration == 0:
            self.dni = weather['DNI']
            self.Tamb = weather['Tdry']+273.15
            self.rh = weather['RH'] if 'RH' in weather.keys() else util.calculate_rh(self.weather['Tdry'], self.weather['Tdew'])    
            self.vwind10 = weather['wspd']
        else:
            timepts = np.arange(len(weather['DNI'])) * 3600/wf_steps_per_hr 
            self.dni = [timepts, weather['DNI']]
            self.Tamb = [timepts, weather['Tdry']+273.15]
            rh = weather['RH'] if 'RH' in weather.keys() else util.calculate_rh(self.weather['Tdry'], self.weather['Tdew'])    
            self.rh = [timepts, rh]
            self.vwind10 = [timepts, weather['wspd']]

        return 
    
    #=========================================================================
    # Extract new OperatingConditions instance with operating conditions at time "time_sec" relative to the simulation start
    def get_operating_conditions_at_time(self, time_sec, ignore = None):
        base_ignore = ['day', 'tod', 'hour_offset', 'is_interpolate']  # Receiver class doesn't need these since flux distributions and sky temperature are inputs
        ignore = base_ignore if not ignore else base_ignore + ignore
        oc = OperatingConditions()
        for name in vars(self).keys():
            if name in ignore:
                continue
            interpolate = self.is_interpolate_weather if name != 'inc_flux' else self.is_interpolate_flux
            setattr(oc, name, self.get_value_at_time(name, time_sec, interpolate))
        return oc

    def get_value_at_time(self, name, time_sec, isinterp = False):
        data = getattr(self, name)
        if not hasattr(data, '__len__') or not hasattr(data[0], '__len__'): # Fixed single-value or fixed 1D array
            ret = data   
        elif len(data)>2:  # Flux input arrays  # TODO: Better way to identify these...
            ret = data
        else:
            times = np.array(data[0])
            vals = np.array(data[1])  
            ndim = len(vals.shape) - 1 
            i = 0 if time_sec == 0  else np.where(times <= time_sec)[0][-1]  # Last tabulated time point less than or equal to 'time_sec'. 
            vals0 = vals[i] if ndim == 0 else vals[i,...]
            if not isinterp or time_sec == 0 or i == len(times)-1:
                ret = vals0
            else:
                vals1 = vals[i+1] if ndim == 0 else vals[i+1,...]
                ret = vals0 + (vals1-vals0) * (time_sec-times[i])/(times[i+1]-times[i])   
        return ret

    
    #=========================================================================
    # Calculate sky temperature from specifications of ambient temperature and RH
    # The time resolution will be matched to that of Tamb (if a time-dependent Tamb is specified).  If not, the time resolution will be calculated based on 'duration' and 'step_sec'
    def set_sky_temperature_vs_time(self, site, duration_sec = None, step_sec = None):        
        timepts = np.arange(0, duration_sec+1, step_sec)
        if hasattr(self.Tamb, '__len__'):  # Match time resolution of ambient temperature variation (assumed rh has same time variation or no time variation)
            timepts = self.Tamb[0]
        self.Tambrad = [timepts, []]
        for j in range(len(timepts)):
            t = timepts[j]
            Tamb = self.Tamb if not hasattr(self.Tamb, '__len__') else self.Tamb[1][j]
            rh = self.rh if not hasattr(self.rh, '__len__') else self.rh[1][j]
            doy = self.day + int((self.tod+t/3600)/24)
            tod = (self.tod+t/3600) % 24
            hour_offset = util.get_offset(site, doy, tod) 
            self.Tambrad[1].append(util.calculate_sky_temperature(Tamb, rh, hour_offset))
        if len(timepts) == 1:
            self.Tambrad = self.Tambrad[1][0]
        return
    
    #=========================================================================
    # Calculate clear-sky DNI
    # The time resolution will be matched to that for DNI variability (if a time-dependent DNI is specified).  If not, the time resolution will be calculated based on 'duration' and 'step_sec'
    # Note that the time resolution used here will have implications on mass flow control. If using 
    def set_clearsky_dni_vs_time(self, site, duration_sec = None, step_sec = None):        
        timepts = np.arange(0, duration_sec+1, step_sec)
        self.dni_clearsky = [timepts, []]
        for j in range(len(timepts)):
            t = timepts[j]
            doy = self.day + int((self.tod+t/3600)/24)
            tod = (self.tod+t/3600) % 24            
            self.dni_clearsky[1].append(util.calculate_clearsky_DNI(site, doy, tod))
        if len(timepts) == 1:
            self.dni_clearsky = self.dni_clearsky[1][0]
        return    
    
    
    
    
    #=========================================================================
    # Get list of times points at which conditions change -> Used to calculate mass flow rates
    def get_time_points_at_change_in_conditions(self, is_clearsky = False):
        times = [0]
        names = ['Tamb', 'vwind10', 'Tfin', 'inc_flux']  # Operating conditions that would require a change in mass flow
        if not is_clearsky:
            names += ['dni']
        else:
            names += ['dni_clearsky']
        
        for v in names: 
            data = getattr(self, v)
            if hasattr(data, '__len__') and len(data) == 2:
                for j in range(len(data[0])):
                    t = data[0][j]
                    if t == 0 and j>0:
                        t = 0.001   # Distinguish this time point from the data at t = 0 used for the inital condition
                    times.append(t)       
        times = np.unique(times)
        return times


        
                

            
