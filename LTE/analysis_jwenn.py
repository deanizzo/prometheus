import timeit
import numpy as np
import operating_conditions
from billboard_receiver import BillboardReceiver
from field_and_flux_jwenn import SolarField
'''
saved analysis as analysis_jwenner to start messing with solar flux
'''


class Analysis:
    def __init__(self, rec = None, field = None):   # Note: rec needs to be passed after rec.initialize() has been called 
    
        self.receiver = rec if rec is not None else BillboardReceiver()
        self.SF = field if field is not None else SolarField(self.receiver)
        self.weather_file = 'climate_files/USA CA Daggett (TMY2).csv' 
        self.weather_steps_per_hour = 1
        
        self.operating_conditions = operating_conditions.OperatingConditions()
        
        #--- Resolution for flux profile calculations. Flux profiles and field efficiency will be interpolated between these time points
        self.n_flux_days = 8        # Will be re-set in initialize() based on designated analysis_mode
        self.delta_flux_hrs = 1
        
        #--- Options for re-sizing receiver diameter to meet a peak flux constraint
        self.resize_receiver_diameter = False  # Iteratively resize receiver diameter to meet target maximum flux.  If false the current diameter in self.receiver.D will be used
        self.target_flux_max = 1000.  # Target peak absorbed flux for receiver sizing optimization (kW/m2)
        
        #--- Options for reading/saving flux distributions
        #    Filename is pre-defined based on location, thermal capacity, receiver height, tower height, and either receiver diameter or peak flux limit (if diameter is being set iteratively).  
        #    See SolarField.get_file_name())
        self.check_for_existing_flux_distributions = True  # Use pre-existing flux distributions if they exist for the specified inputs
        self.save_flux_distributions = True                # Save outputs from field_and_flux_simulation
        
        
        
        return

    
    #=========================================================================
    # Call SolarPILOT to calculate flux distributions
    def calculate_flux_distributions(self, verbose = True):
        start = timeit.default_timer()
        self.SF.n_flux_days = self.n_flux_days
        self.SF.delta_flux_hrs = self.delta_flux_hrs       
        self.SF.resize_receiver_diameter = self.resize_receiver_diameter
        self.SF.target_flux_max = self.target_flux_max

        # self.SF.simulate_field_and_flux_maps_wCP(check_saved = self.check_for_existing_flux_distributions, save_flux = self.save_flux_distributions, verbose = verbose)
        # self.SF.calculate_interpolation_parameters()

        #print ('Time for flux profiles = %.3fs'%(timeit.default_timer() - start))
        return
    
    #=========================================================================
    # Set the flux distribution for the current operating conditions or the set of operating conditions (for transient models)
    # Note this should be called after populating time and dni in self.operating_conditions
    def set_flux_distribution(self, flux_update_times = None, is_clearsky = False):     
        if flux_update_times is None:
            dni = self.operating_conditions.dni if not is_clearsky else self.operating_conditions.dni_clearsky
            inc_flux, opteff = self.get_field_efficiency_and_flux_distribution(0.0, dni)
            self.operating_conditions.inc_flux = inc_flux
            self.operating_conditions.opteff = opteff
        else:
            self.operating_conditions.inc_flux = [flux_update_times, []]
            self.operating_conditions.opteff = [flux_update_times, []]
            for j in range(len(flux_update_times)):  
                t = flux_update_times[j]
                dni = self.operating_conditions.get_value_at_time('dni', t, False)
                inc_flux, opteff = self.get_field_efficiency_and_flux_distribution(t, dni)
                self.operating_conditions.inc_flux[1].append(inc_flux)
                self.operating_conditions.opteff[1].append(opteff)
        return 
    
    def get_field_efficiency_and_flux_distribution(self, time_sec, dni):
        day = self.operating_conditions.day
        tod = self.operating_conditions.tod + time_sec/3600          
        if tod >=24:
            day += 1
            tod -= 24   
        inc_flux, opteff = self.SF.get_flux_profile(day, tod)  # Note this is incident (not absorbed) flux, and optical efficiency does not include reflection loss
        inc_flux = np.array(inc_flux)*dni / self.SF.dni_des
        return inc_flux, opteff
            
                