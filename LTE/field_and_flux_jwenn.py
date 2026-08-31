# Functions to run solarPILOT and read/write flux profiles
# Current ssc.dll compiled from NREL/ssc/develop on 9/24/2021
# TODO: SolarPILOT is run through the ssc API using PySSC ->update to SolarPILOT API


import numpy as np
from math import pi, acos, floor, ceil, isnan
import os
from copy import deepcopy
import json
import util
from copylot import CoPylot
import csv
import matplotlib.pyplot as plt
import random
import pandas as pd
import helpers_thermal_model
import timeit


# Note that this class uses SolarPILOT variable names for easier data transfer
class SolarField:
    def __init__(self, rec):
        self.solar_resource_file = 'USA CA Daggett (TMY2).csv'  # Weather file
        self.dni_des = 950              # Design point DNI for field layout (W/m2)      
        self.Ndisable = None

        self.helio_height = 12.2        # Heliostat height (m)
        self.helio_width = 12.2         # Heliostat width (m)
        self.helio_image_err = 3.0      # Total heliostat reflected image error (mrad)
        
        self.receiver = rec
        self.resize_receiver_diameter = False
        self.target_flux_max = 1000

        self.n_flux_days = 8            # Number of days in the flux map calculations
        self.delta_flux_hrs = 1         # Time resolution (hr) between  flux calculation points

        # self.n_flux_x = 30
        # self.n_flux_y = 30
        
        # Optional files containing heliostat field layout
        self.field_layout_file = ''     # Optional file containing stored heliostat field (required if create_field_layout = False and flux_profile_file is empty) 
        self.flux_profile_file = ''     # Optional file containing stored flux distributions
        self.opt_eff_file = ''          # Optional file containing field optical performance
        
        self.interpolation_scaling = {'zen':90, 'az':360, 'opteff':0.7}
        self.interpolation_method = 'Gauss-Markov'   # 'RBF' or Gauss-Markov

        self.gauss_markov_params = {'beta': 1.99, 'nu': 0.0}
        
        # Location to save and/or read previously-simulated flux distributions
        self.saved_flux_direc = './flux_sims/'
        self.flux_tol = 10      # Tolerance in peak flux for re-calculation of flux profiles
        self.height_tol = 0.1   # Tolerance in receiver height for re-calculation of flux profiles
        
        
        #--- Results
        self.helio_positions = []   # Heliostat positions
        self.days = []      # Array of simulated days
        self.zen = []       # Array of simulated zenith angles
        self.az = []        # Array of simulated azimuth angles
        self.opteff = []    # Field Optical efficiency
        self.incflux = []   # Distribution of incident solar flux 
        self.relflux = []   # Relative flux distributions (fraction of total flux incident on each node)
        self.design_idx = 0 # Index of time point corresponding to design point
        self.Nhel = 0       # Number of heliostats
        self.Asf = 0        # Solar field area
        
        self.opteff_interp_params = []

        return
    
    #==========================================================================
    # Create dictionary of inputs to send to SolarPILOT
    #### OG THAT JANNA WROTE
    def create_input_dict(self):
        inputs = {}
        for k in ['solar_resource_file', 'dni_des', 'helio_height', 'helio_width', 'n_flux_days', 'delta_flux_hrs', 'n_flux_x', 'n_flux_y']:
            inputs[k] = getattr(self, k)
        inputs['q_design'] = self.receiver.Qdes
        inputs['rec_height'] = self.receiver.H
        inputs['rec_aspect'] = self.receiver.H/self.receiver.D
        inputs['h_tower'] = self.receiver.Htower
        inputs['rec_absorptance'] = self.receiver.solar_abs
        inputs['rec_hl_perm2'] = self.receiver.est_heat_loss
        inputs['flux_max'] = self.target_flux_max
        inputs['helio_optical_error'] = (( self.helio_image_err**2)/4)**0.5 / 1000  # Slope error in rad
        return inputs
    ###

    def create_input_dict_cp(self):
        # make an input dictionary that can be useed by the api
        inputs = {}
        inputs["ambient.0.weather_file"] = self.receiver.solar_resource_file
        inputs["receiver.0.rec_type"] = self.receiver.rec_type
        inputs["receiver.0.q_des"] = self.receiver.Qdes
        inputs["receiver.0.rec_height"] = self.receiver.H
        # inputs["receiver.0.rec_diameter"] = self.receiver.D                       # use this line instead of rec_width line if testing a cylindrical receiver
        inputs["receiver.0.rec_width"] = self.receiver.D
        inputs["solarfield.0.tht"] = self.receiver.Htower
        inputs["receiver.0.peak_flux"] = self.target_flux_max
        inputs["fluxsim.0.sigma_limit_x"] = self.receiver.sigma_limit_x
        inputs["fluxsim.0.sigma_limit_y"] = self.receiver.sigma_limit_y
        inputs["fluxsim.0.x_res"] = self.receiver.n_flux_x
        inputs["fluxsim.0.y_res"] = self.receiver.n_flux_y
        if self.receiver.aiming_file != "None":
            inputs['aiming_file'] = self.receiver.aiming_file
            inputs['receiver.0.flux_profile_type']="User"     
            print('setting the flux_profile_type to --User--')
        else:
            inputs['receiver.0.flux_profile_type']="Uniform" 

        inputs['layout_file'] = self.receiver.layout_file # always need a layout file
        print('using layout file:',self.receiver.layout_file)
        # if self.receiver.hel_disable_file != "None":
        #     inputs['hel_disable_file'] = self.receiver.hel_disable_file
        inputs["align_Qdes"] = self.receiver.align_Qdes
        inputs['err_reflect_x'] = self.receiver.err_reflect_x
        inputs['err_reflect_y'] = self.receiver.err_reflect_y
        inputs['err_surface_x'] = self.receiver.err_surface_x
        inputs['err_surface_y'] = self.receiver.err_surface_y

        # for k in ['solar_resource_file', 'dni_des', 'helio_height', 'helio_width', 'n_flux_days', 'delta_flux_hrs', 'n_flux_x', 'n_flux_y']:
        #     inputs[k] = getattr(self, k)

        # inputs['rec_absorptance'] = self.receiver.solar_abs
        # inputs['rec_hl_perm2'] = self.receiver.est_heat_loss
        # inputs['flux_max'] = self.target_flux_max
        # inputs['helio_optical_error'] = (( self.helio_image_err**2)/4)**0.5 / 1000  # Slope error in rad
        # inputs['rec_type'] = self.receiver.rec_type # added 12/5/24
        # inputs['flux_profile_type']="User"

        return inputs
    ###

    #==========================================================================
    '''
    Use Solar PILOT to create field layout and optimize receiver diameter to meet peak flux limit
    custom_inputs = dictionary with any other custom inputs (must use same names as SolarPILOT inputs)
    '''
    def simulate_field_and_flux_maps(self, allow_resize = True, check_saved = True, save_flux = False, verbose = False): 
        
        # Try to read in profiles from external file
        if check_saved:
            ok = self.read_flux_profiles(verbose)
            if ok:
                return

        # Calculate new flux profiles
        inputs = self.create_input_dict()
        inputs['check_max_flux'] = 1   
        inputs['calc_fluxmaps'] = 1
    
        if not self.resize_receiver_diameter or not allow_resize:
            results = run_solarpilot(inputs, True, verbose) 
        else:
            original_res = {'n_flux_days':self.n_flux_days, 'delta_flux_hrs':self.delta_flux_hrs}
            new_res = {'n_flux_days':2, 'delta_flux_hrs':8}  # Lowest possible flux profile resolution for iterations (just need design point)
            inputs.update(new_res)
            vars(self).update(new_res)
            self.adjust_receiver_diameter(nmax = 30, fluxtol = 5, verbose = verbose)  # Calculate receiver diameter to achieve desired peak flux.  Overwrites diameter in rec.D
            inputs['rec_aspect'] = self.receiver.H/self.receiver.D
            inputs.update(original_res)
            vars(self).update(original_res) # revert back to original parametres for final flux profile calculations       
            results = run_solarpilot(inputs, True, verbose) 
            self.receiver.initialize(True)  # Re-initialize receiver 
        
        self.postprocess_flux_sim(results)   # flux table size: 44 x 30 x30 Converts to optical efficiency and incident flux that DO NOT include receiver absorption efficiency
        self.helio_positions = results['heliostat_positions']
        self.Nhel = results['number_heliostats']
        self.Asf = results['area_sf']
  
        observed_design_pt_flux_max = self.incflux[:,:,self.design_idx].max() * self.receiver.solar_abs
        observed_overall_flux_max = self.incflux.max() * self.receiver.solar_abs
        area = pi * self.receiver.D*self.receiver.H
        Qabs_des = area * self.incflux[:,:,self.design_idx].mean()/1000 * self.receiver.solar_abs
        QHTF_des_est = Qabs_des - self.receiver.est_heat_loss*area/1000

        if verbose:
            print('Number of heliostats = %d' % (self.Nhel))
            #print('Specified max flux = %.0f kW/m2' % (self.target_flux_max))        
            print('Peak absorbed flux from profiles: Design point = %.0f kW/m2, Overall = %.0f kW/m2' % (observed_design_pt_flux_max, observed_overall_flux_max))
            print('Target thermal power to HTF = %.0f MW' % (self.receiver.Qdes))        
            print('Actual design point absorbed thermal power = %.0f MW' % (Qabs_des)) 
            print('Expected design point thermal power to HTF = %.0f MW' % (QHTF_des_est)) 

        #--- Re-initialize receiver is diameter was re-sized
        if self.resize_receiver_diameter:
            self.receiver.initialize(True) 
            
        #--- Save profiles
        if save_flux:
            self.save_flux_profiles()
            
        return
    
    def simulate_field_and_flux_maps_wCP(self, allow_resize = True, check_saved = True, save_flux = False, verbose = False): 
        
        # Try to read in profiles from external file
        if check_saved:
            ok = self.read_flux_profiles(verbose)
            if ok:
                return

        # Calculate new flux profiles
        inputs = self.create_input_dict_cp()
        inputs['check_max_flux'] = 1   
        inputs['calc_fluxmaps'] = 1
    
        if not self.resize_receiver_diameter or not allow_resize:
            results = run_solarpilot_wCP(inputs, True, verbose) 
        else:
            original_res = {'n_flux_days':self.n_flux_days, 'delta_flux_hrs':self.delta_flux_hrs}
            new_res = {'n_flux_days':2, 'delta_flux_hrs':8}  # Lowest possible flux profile resolution for iterations (just need design point)
            inputs.update(new_res)
            vars(self).update(new_res)
            self.adjust_receiver_diameter(nmax = 30, fluxtol = 5, verbose = verbose)  # Calculate receiver diameter to achieve desired peak flux.  Overwrites diameter in rec.D
            inputs['rec_aspect'] = self.receiver.H/self.receiver.D
            inputs.update(original_res)
            vars(self).update(original_res) # revert back to original parametres for final flux profile calculations       
            results = run_solarpilot_wCP(inputs, True, verbose) 
            self.receiver.initialize(True)  # Re-initialize receiver 
        
        self.postprocess_flux_sim(results)   # Converts to optical efficiency and incident flux that DO NOT include receiver absorption efficiency
        self.helio_positions = results['heliostat_positions']
        self.Nhel = results['number_heliostats']
        self.Asf = results['area_sf']
  
        observed_design_pt_flux_max = self.incflux[:,:,self.design_idx].max() * self.receiver.solar_abs
        observed_overall_flux_max = self.incflux.max() * self.receiver.solar_abs
        area = pi * self.receiver.D*self.receiver.H
        Qabs_des = area * self.incflux[:,:,self.design_idx].mean()/1000 * self.receiver.solar_abs
        QHTF_des_est = Qabs_des - self.receiver.est_heat_loss*area/1000

        if verbose:
            print('Number of heliostats = %d' % (self.Nhel))
            #print('Specified max flux = %.0f kW/m2' % (self.target_flux_max))        
            print('Peak absorbed flux from profiles: Design point = %.0f kW/m2, Overall = %.0f kW/m2' % (observed_design_pt_flux_max, observed_overall_flux_max))
            print('Target thermal power to HTF = %.0f MW' % (self.receiver.Qdes))        
            print('Actual design point absorbed thermal power = %.0f MW' % (Qabs_des)) 
            print('Expected design point thermal power to HTF = %.0f MW' % (QHTF_des_est)) 

        #--- Re-initialize receiver is diameter was re-sized
        if self.resize_receiver_diameter:
            self.receiver.initialize(True) 
            
        #--- Save profiles
        if save_flux:
            self.save_flux_profiles()
            
        return 



    #========================================================================== 
    # Iteratively adjust reciever diameter to achieve target thermal power
    def adjust_receiver_diameter(self, nmax = 30, fluxtol = 5, verbose = True):
        Dguess = (self.receiver.Qdes*1000./0.9)/(pi*self.receiver.H*0.7*self.target_flux_max)   # Initial guess for D assuming (avg flux / max flux) = 0.7 and receiver is 90% efficient
        flux_max_target = self.target_flux_max
        self.receiver.D = Dguess
        iteration_history = []
        for i in range(nmax):
            self.simulate_field_and_flux_maps(False, False, False, False)  # Run SolarPILOT with lower flux map resolution for speed
            flux_max_observed = self.incflux[:,:,self.design_idx].max() * self.receiver.solar_abs  # Observed peak absorbed flux
            f = flux_max_observed-flux_max_target
            if verbose:
                print(i, self.receiver.D, flux_max_observed)
            iteration_history.append([self.receiver.D, f])
            if abs(f)<=fluxtol:
                print('Solution converged with receiver diameter = %.4f m ' % (self.receiver.D)) 
                break
            
            if i == 0:
                self.receiver.D += self.receiver.D * (flux_max_observed - flux_max_target)/flux_max_target
            else:
                deriv = (iteration_history[-1][1] - iteration_history[-2][1]) / (iteration_history[-1][0] - iteration_history[-2][0])
                Dnew = self.receiver.D - f/deriv
                self.receiver.D = Dnew
            if i == nmax-1:
                print ('Solution for receiver diameter failed to converge')
        return 
            
    
            
    #==========================================================================  
    # Postprocess flux simulation results -> Create absorbed flux profiles and extract information on field efficiency and solar position
    def postprocess_flux_sim(self, results):  
        nx = self.n_flux_x
        ny = self.n_flux_y
        nfluxdays = self.n_flux_days
        zenith = np.array([p[1] for p in results['opteff_table']])
        azimuth = np.array([p[0]+180 for p in results['opteff_table']])
        opteff = np.array([p[2] for p in results['opteff_table']])
        elem_area = pi*self.receiver.D*self.receiver.H/nx/ny
    
        npts = zenith.shape[0]
        abs_flux = np.zeros((ny, nx, npts))   # Each flux profile is array with [height, position around diameter (i.e. panel index)]
        rel_flux = np.zeros((ny,nx,npts))
        for i in range(npts):
            eff = opteff[i]  # Note field efficiency includes receiver reflection loss
            for j in range(ny):
                for k in range(nx):
                    rel_flux[j,k,i] = results['flux_table'][i*ny+j][k]
                    abs_flux[j,k,i] = results['flux_table'][i*ny+j][k]  * (eff * results['dni_des'] * results['area_sf'] /  1000.) / elem_area   # Absorbed flux at design point DNI

        #--- Find day of year for each flux profile and set design point index
        simdays = []
        for i in range(nfluxdays):
            d = 355 - floor((355-172) * acos(-1 + 2*i/(float(nfluxdays-1)))/pi)
            simdays.append(d)
        
        days, times = [np.zeros_like(zenith) for v in range(2)]
        if npts == len(simdays):  # Only one point (solar noon) per simulated day
            days = np.array(simdays)
        else:
            days[0] = simdays[0]
            d = 0
            for i in range(1, npts):
                if azimuth[i] < azimuth[i-1]:  
                    d+=1                  
                days[i] = simdays[d]
 
        noons = np.where(np.abs(azimuth - 180) < 1)[0]   # Time points at solar noon 
        for d in simdays:
            inds = np.where(d == days)[0]                   # Points on this day
            sn = np.intersect1d(inds, noons)[0]             # Point at noon on this day
            times[inds] = (inds - sn) * self.delta_flux_hrs # Time points on this day (hours in offset from solar noon)            
                             
        design_pt = np.where(np.logical_and(days == 172, times == 0))[0][0]  # Index of design point on design day (day 172)
        
        
        #--- Convert to incident flux and optical efficiency that DO NOT include receiver solar absorption
        incflux = abs_flux / self.receiver.solar_abs    
        opteff = opteff / self.receiver.solar_abs

        self.days = days
        self.times = times
        self.zen = zenith
        self.az = azimuth
        self.opteff = opteff
        self.incflux = incflux
        self.relflux = rel_flux
        self.design_idx = int(design_pt)

        return
    
    
    #==========================================================================
    def get_file_name(self, D):
        site = util.read_weatherfile_header(self.solar_resource_file) 
        
        height = self.height_tol * int((D['rec_height']+0.0001) / self.height_tol)
        diam = self.height_tol * int((D['rec_height']/D['rec_aspect']+0.0001) / self.height_tol)
        tower = D['h_tower']
        flux =  self.flux_tol * int((D['flux_max']+0.0001) /self.flux_tol)

        geom = 'lat%.2f_lon%.2f_Q%d_H%.1f_Htow%d_abs%d'%(site['lat'], site['lon'], D['q_design'], height, tower, D['rec_absorptance']*100)
        if self.resize_receiver_diameter:
            name = '%s_flux%d.json'%(geom, flux)
        else:
            name = '%s_D%.1f.json'%(geom, diam)
        return name        

    
    def save_flux_profiles(self):
        # Note field efficiency saved in file DOES NOT include receiver absorption efficiency
        ok = os.path.exists(self.saved_flux_direc)
        if not ok:
            os.mkdir(self.saved_flux_direc)
            
        D = self.create_input_dict() 
        if not self.resize_receiver_diameter:  # Value in 'flux_max' wasn't used, update to actual flux_max
            D['flux_max'] = self.incflux[:,:,self.design_idx].max()*self.receiver.solar_abs
            
        for k in ['Nhel', 'Asf', 'design_idx', 'helio_positions', 'days', 'zen', 'az', 'opteff', 'relflux']:
            val = getattr(self,k)
            if type(val) == type(np.zeros(1)):
                val = deepcopy(val).tolist()
            D[k] = val
            
        name = self.get_file_name(D)
        with open(self.saved_flux_direc+name, 'w') as f: 
            json.dump(D, f)
        return

    def read_flux_profiles(self, verbose = False):
        D = self.create_input_dict()
        name = self.get_file_name(D)
        if name not in os.listdir(self.saved_flux_direc):  # File doesn't exist
            return False
        
        if verbose:
            print ('Reading flux distributions from '+name)
        with open(self.saved_flux_direc+name, 'r') as f:
            data = json.load(f)
            
        # Ignore stored profiles if time resolution isn't high enough
        if data['n_flux_days'] < self.n_flux_days or data['delta_flux_hrs']> self.delta_flux_hrs:
            print('Stored profiles found with lower than specified flux time point resolution. Recalculating...') 
            return False
        
        # Change receiver diameter if required:
        if self.resize_receiver_diameter and abs(self.receiver.D - data['rec_height']/data['rec_aspect']) > 0.01:
            self.receiver.D = data['rec_height']/data['rec_aspect']
            self.receiver.initialize(True) 

        # Check inputs against current parameters
        for k, val in data.items():
            if k == 'flux_max' and not self.resize_receiver_diameter:
                continue
            elif k in ['n_flux_x', 'n_flux_y', 'n_flux_days', 'delta_flux_hrs']:
                continue
            elif type(val) != type([]) and type(val) != type(' '):
                diff = (data[k] - val) / np.maximum(1e-6,val)
                if abs(diff) > 0.001:
                    print ('Warning: parameter %s differs from stored value by %.2f%%'%(k, diff))

        # Store data
        for k in ['Nhel', 'Asf', 'dni_des', 'helio_positions', 'days', 'zen', 'az', 'opteff', 'relflux']:
            if type(data[k]) == type([]):
                data[k] = np.array(data[k])
            setattr(self, k, data[k])  
            
        elem_area = pi*self.receiver.D*self.receiver.H/self.n_flux_x/self.n_flux_y
        self.incflux = np.zeros_like(self.relflux)
        for j in range(self.incflux.shape[2]):
            self.incflux[:,:,j] = self.relflux[:,:,j] * self.opteff[j]*self.dni_des * self.Asf / 1000 / elem_area 

        return True

    #==========================================================================  
    # Calculate parameters for radial basis function interpolation
    def calculate_interpolation_parameters(self):
        x = np.stack([self.zen/self.interpolation_scaling['zen'], self.az/self.interpolation_scaling['az']], axis = 1)
        y = self.opteff/self.interpolation_scaling['opteff']
        if self.interpolation_method == 'RBF':
            
            self.opteff_interp_params = util.radial_basis_function_params(x, y, self.opteff_rbf)   
        elif self.interpolation_method == 'Gauss-Markov':
            self.opteff_interp_params = util.gauss_markov_interp_params(x, y, self.gauss_markov_params['beta'], self.gauss_markov_params['nu'])
        else:
            print('Interpolation method %s for field efficiency is not recognized'% self.interpolation_method)

        # Test interpolation
        npt = len(self.zen)
        err = np.zeros(npt)
        for i in range(npt):
            err[i] = self.opteff[i] - self.interpolate_optical_efficiency(self.zen[i], self.az[i])
        if np.abs(err).max() > 0.0005:
            print ('Warning: Max interpolation error at tabulated points = %.4f%%'%(np.abs(err).max()))

        return
    
    def interpolate_optical_efficiency(self, zen, az):
        xpt = np.array([zen/self.interpolation_scaling['zen'], az/self.interpolation_scaling['az']])
        x = np.stack([self.zen/self.interpolation_scaling['zen'], self.az/self.interpolation_scaling['az']], axis = 1)
        y = self.opteff/self.interpolation_scaling['opteff']
        if self.interpolation_method == 'RBF':
            opteff_rbf = lambda r: np.exp(-25*(r**2))
            opteff_scaled = util.radial_basis_function_interp(xpt, x, params = self.opteff_interp_params, rbf = opteff_rbf)
        elif self.interpolation_method == 'Gauss-Markov':
            opteff_scaled = util.gauss_markov_interp(xpt, x, y, self.opteff_interp_params, self.gauss_markov_params['beta'], self.gauss_markov_params['nu'])
        opteff = opteff_scaled*self.interpolation_scaling['opteff']

        return opteff
    
    
    
    #==========================================================================  
    # Get flux profile at given day of year and either hour of day, or hour offset from solar noon
    def get_flux_profile(self, doy, hour_of_day = None, hour_offset = None):
        ok = False
        
        site = util.read_weatherfile_header(self.solar_resource_file)
        zen, az = util.calculate_solar_position(site, doy, hour_of_day, hour_offset)   # Zenith/azimuth angle at desired time point
        
        # First look for exact match in tabulated flux points
        if hour_offset is not None and not isnan(hour_offset):  
            inds = np.where(np.logical_and(self.days == doy, np.abs(self.az - az)<1.0))[0]
            if len(inds) > 0:  # Exact match for flux profile
                #print ('Using flux profile without interpolation')
                opteff = self.opteff[inds[0]]
                inc_flux = self.incflux[:,:,inds[0]]
                #nz, nx, nprof = self.relflux.shape
                #relative_flux = self.relflux[:,:,inds[0]]
                #elem_area = np.pi*self.receiver.D*self.receiver.H/nz/nx
                #inc_flux = relative_flux * (opteff * self.dni_des * self.Asf / 1000.) / elem_area   # Incident flux at design point DNI
                ok = True

        
        # Interpolate if not exact match exists
        # TODO: The flux interpolation doesn't work very well for cases when the time point is very close to a single time point. Flux profiles end up overweighted by points beyond just the single closest point
        if not ok: 
            opteff = self.interpolate_optical_efficiency(zen,az)    # Note: optical efficiency here does not include receiver reflection loss
            
            # Interpolate flux distribution (based on flux interpolation used in SAM)
            zen_scale = self.interpolation_scaling['zen']
            az_scale = self.interpolation_scaling['az']           
            navg = min(6, len(self.zen))                            # Number of points used for interpolation
            dist = ((zen/zen_scale - self.zen/zen_scale)**2 + (az/az_scale - self.az/az_scale)**2)**0.5   # Distance from current zen/az point to tabulated points
            inds = dist.argsort()[0:navg]                           # Index locations of closest points
            wts = np.exp(-(dist[inds]/dist[inds].mean())**2)        # Non-normalized weights for selected interpolation points
            wts = wts / wts.sum()                                   # Normalized weights for selected interpolation points
            nz, nx, nprof = self.relflux.shape
            relative_flux = np.zeros((nz,nx))
            for j in range(navg):
                relative_flux[:,:] += wts[j] * self.relflux[:,:,inds[j]]
            relative_flux = relative_flux / relative_flux.sum()
            elem_area = np.pi*self.receiver.D*self.receiver.H/nz/nx # total area of the receiver divided by solarpilot grid resolution
            inc_flux = relative_flux  * (opteff * self.dni_des * self.Asf / 1000.) / elem_area   # Incident flux at design point DNI

        return inc_flux, opteff

#==========================================================================  
    # call solarpilot to grab flux profile given a day of the year and hour of day
    def get_flux_profile_wCP(self, doy, dni, hour_of_day = None):

        inputs = self.create_input_dict_cp()
        
        layout_file = inputs["layout_file"]
        if "aiming_file" in inputs:
            aiming_file = inputs["aiming_file"]

        ## print the flux resolution
        nx_cp =54
        ny_cp =54
        print(f'SPT resolution is {nx_cp} x {ny_cp}')

        ## use these for debugging   
        # cp.data_get_number(r,"fluxsim.0.flux_solar_el")
        # cp.data_get_number(r,"fluxsim.0.flux_solar_az")
        ##

        ## update receiver results with Ndisable for final report
        self.Ndisable = 0

        ## set the cp day/time after align_inc_powers checked the copylot instance at design conditions

        # ## test the input file directly by overriding cp and using it as the flux
        flux_df =pd.read_csv(aiming_file, header=None)
        inc_flux_W=flux_df.values
        inc_flux=inc_flux_W # inc_flux should be in kW/m2
        print('currently reading flux from CSV')
        # ##

        ## increase flux resolution to maintain interpolation integrity
        block_factor    =9
        inc_flux = self.increase_flux_resolution_blocked_custom(np.array(inc_flux), nx_cp*block_factor, ny_cp)
        print(f'NOTE: increasing flux resolution post SPT, prior to thermal model input via blocking factor of {block_factor} in x direction')
        ##
        opteff=1
        
        ## plot flux profile
        if abs(12 - hour_of_day) < 1:   # save the flux profile if we're within an hour of solar noon 
            fontsize = 14
            fluxFig, fluxAx = plt.subplots()
            im = fluxAx.imshow(inc_flux, extent=[-self.receiver.D/2, self.receiver.D/2, -self.receiver.H/2, self.receiver.H/2])
            cbr = fluxFig.colorbar(im)
            cbr.set_label(label='incident flux [kW/m$^2$]', size=fontsize)
            cbr.ax.tick_params(labelsize=fontsize)
            fluxAx.set_xlabel('x position (m)',size=fontsize)
            fluxAx.set_ylabel('y position (m)',size=fontsize)

            # # for xlabel_i in fluxAx.axes.get_xticklabels():
            # #     xlabel_i.set_visible(False)
            # #     xlabel_i.set_fontsize(0.0)
            # # for xlabel_i in fluxAx.axes.get_yticklabels():
            # #     xlabel_i.set_fontsize(0.0)
            # #     xlabel_i.set_visible(False)
            # # for tick in fluxAx.axes.get_xticklines():
            # #     tick.set_visible(False)
            # # for tick in fluxAx.axes.get_yticklines():
            # #     tick.set_visible(False)

            # plt.savefig(f'imgs/flux_profile_doy{doy}_hod{hour_of_day}_AR{self.receiver.H/self.receiver.D:.2f}.jpg', dpi=300)
            plt.savefig(f'imgs/default_save_spt_output.jpg', dpi=300)
            # plt.show()
            plt.close()
       
        # ## plot field layout
        # plt.scatter(field['x_location'], field['y_location'], s=1.5)
        # plt.tight_layout()
        # plt.show()
        
        return inc_flux, opteff
    
    def increase_flux_resolution_blocked_custom(self,flux_low_res, npts_horizontal, npts_vertical):
        """
        take in a low resolution flux, repeat the values to a higher resolution
        flux_low_res                    - (kW/m2) y X x array of fluxes, where x spans the horizontal direction
        npts_horizontal/vertical        - (int) new desired resolution
        ---
        returns:
        flux_high_res   - (kW/m2) block-increased array of flux values for thermal model
        """
        npts_vertical_og =flux_low_res.shape[0]
        npts_horizontal_og=flux_low_res.shape[1]

        if (npts_vertical % npts_vertical_og != 0) or (npts_horizontal % npts_horizontal_og != 0):
            print('resquested dimension must be a multiple of original dimension')
        else:
            # this increases it in the vertical direction, assuming y is first dimension
            flux_high_res = np.repeat(flux_low_res, int(npts_vertical/npts_vertical_og), axis = 0 )
            # this increases it in the horizontal direction, assuming x is second dimension
            flux_high_res = np.repeat(flux_high_res, int(npts_horizontal/npts_horizontal_og), axis = 1 )

        
        return flux_high_res
       
#==========================================================================   
   # call solarpilot to grab flux profile given a day of the year and hour of day
   ### sort of mothballed this method on 3/7/2025. No longer in use, may not be maintained
    def get_flux_profile_wlayoutSubset(self, doy, hour_of_day = None,nOffs=0):
        def disable_heliostats(cp_inst,r,field,N=0):
            """
            disables N heliostats that are selected by generating a random list of heliostat id's
            """    
            off_range_list = list(range(0,len(field['id'])))
            off_indices=random.sample(off_range_list,N)
            field_IDs  = [field['id'][index] for index in off_indices]    # grab IDs at each index
            bool_arr = np.array(np.zeros(len(field_IDs)), dtype='bool')
            off_dict = {'id':field_IDs,'enabled':bool_arr}
            ok = cp_inst.modify_heliostats(r,off_dict) 
            return field_IDs
        
        def remove_heliostats_from_list(cp_inst,r,disable_csv):
            """
            also disables N heliostats, but field IDs are already known
            """
            removal_df=pd.read_csv(disable_csv)
            removal_list=removal_df.values
            removal_list=np.squeeze(removal_list).tolist()
            bool_arr = np.array(np.zeros(len(removal_list)), dtype='bool')
            off_dict = {'id':removal_list,'enabled':bool_arr}
            assert cp_inst.modify_heliostats(r,off_dict) 

            return removal_list

        inputs = self.create_input_dict_cp()

        cp = CoPylot() # create copylot instance
        r  = cp.data_create() # specific instance case. R is just a memory ID
        
        _set_cp_data_from_dict(cp, r, inputs)  # Update from currently defined inputs
        
        layout_file = inputs["layout_file"]
        if "aiming_file" in inputs:
            aiming_file = inputs["aiming_file"]
        if "aiming_file2" in inputs:
            aiming_file2 = inputs["aiming_file2"]

        assert self.assign_gend_layout(cp,r,inputs) # pass in the pre-generated layout

        field_entire = cp.get_layout_info(r)
        # ##--- disable some heliostats
        # off_indices=np.random.randint(0,len(field_entire['id']), nOffs) # random number of indices to disable heliostats
        # field_IDs  = [field_entire['id'][index] for index in off_indices]    # grab IDs at each index
        # field_IDs.sort()
        # bool_arr = np.array(np.zeros(len(field_IDs)), dtype='bool')
        # off_dict = {'id':field_IDs,'enabled':bool_arr}
        # cp.modify_heliostats(r,off_dict) # shuts off some of the heliostats        
        # ##---
        if nOffs > 0:
            field_IDs = disable_heliostats(cp,r,field_entire,nOffs) # get the og field IDs of the disabled heliostats
        elif "hel_disable_file" in inputs:
            disable_file = inputs['hel_disable_file']
            field_IDs = remove_heliostats_from_list(cp,r,disable_file)
        else:
            field_IDs = 0



        if "aiming_file" in inputs:
            cp.data_set_matrix_from_csv(r, "receiver.0.user_flux_profile", aiming_file )
            print('---')
            print('using aiming file:',aiming_file)
            print('---')
        # elif "aiming_file2" in inputs:
        #     cp.data_set_matrix_from_csv(r, "receiver.0.user_flux_profile", aiming_file2 )

        M,D=util.get_month_and_day(doy)

        cp.data_set_string(r,"fluxsim.0.flux_time_type",'Hour/Day')

        cp.data_set_number(r,"fluxsim.0.flux_month",M)
        cp.data_set_number(r,"fluxsim.0.flux_day",  D)

        cp.data_set_number(r,"fluxsim.0.flux_hour",hour_of_day)      
        # cp.data_get_number(r,"fluxsim.0.flux_solar_el")
        # cp.data_get_number(r,"fluxsim.0.flux_solar_az")

        cp.simulate(r)
        
        # inc_flux = np.array(cp.get_fluxmap(r))
        inc_flux = cp.get_fluxmap(r) #should return the incident flux NOT including absorbtivity
        field = cp.get_layout_info(r)
        # alpha_SPT = cp.data_get_number(r,"receiver.0.absorptance")
        # inc_flux = inc_flux/alpha_SPT
        # inc_flux.tolist()
        

        summ_dict = cp.summary_results(r,save_dict=True)
        opteff=summ_dict['Solar field optical efficiency']/100 # does not include alpha
        
        # plot flux profile
        # flux
        fontsize = 16
        fluxFig, fluxAx = plt.subplots()
        im = fluxAx.imshow(inc_flux)
        cbr = fluxFig.colorbar(im)
        cbr.set_label(label='incident flux [kW/m^2]', size=fontsize)
        fluxAx.set_xlabel('x node',size=fontsize)
        fluxAx.set_ylabel('y node',size=fontsize)
        plt.savefig('imgs/latest_flux_profile.png', dpi=300)
        plt.show()
     
        plt.close()
        
        # # plot layout
        # plt.scatter(field['x_location'], field['y_location'], s=1.5)
        # plt.tight_layout()
        # plt.show()

        cp.data_free(r)
        

        return inc_flux, opteff, field_IDs
#========================================================================== 
# disable the requested number of heliostats from existing field
    def disable_heliostats(self,cp_inst,r,field,N=0):
        """
        disables N heliostats that are selected by generating a random list of heliostat id's
        """    
        off_range_list = list(range(0,len(field['id'])))
        # off_indices=random.sample(off_range_list,N)   # original heliostat removal method was randomly selecting mirrors to defocus
        off_indices=off_range_list[0:N] # selecting the "worst" heliostats to remove. The field df already has sorted the heliostat field in ascending order of field metric
        field_IDs  = [field['id'][index] for index in off_indices]    # grab IDs at each index
        bool_arr = np.array(np.zeros(len(field_IDs)), dtype='bool')
        off_dict = {'id':field_IDs,'enabled':bool_arr}
        ok = cp_inst.modify_heliostats(r,off_dict) 
        return field_IDs
    #========================================================================== 
    # assign  pregenerated layout to copylot instance
    def assign_gend_layout(self,cp,r,inputs):
        layout_file = inputs["layout_file"]
        with open(layout_file, 'r') as file:
            # print("HARDCODING HELIOSTAT LAYOUT")
            csv_reader = csv.reader(file,delimiter=',')
            helio_list_str= list(csv_reader)
            ncols = len(helio_list_str[0])
            helio_list =  [float(element) for row in helio_list_str for element in row] #convert all strings to floats
            helio_list = [helio_list[i:i + ncols ] for i in range(0, len(helio_list), ncols)] # reshape back to original
        cp.assign_layout(r, helio_list,1)
        return True


    #========================================================================== 
    # check if the incident power exceeds the design power at design conditions. If so, remove the least productive heliostats      
    def align_inc_powers(self,cp,r,inputs,doy):
        ## align the number of heliostats once based on design conditions. Set date and time in copylot
        M,D=util.get_month_and_day(172)
        cp.data_set_string(r,"fluxsim.0.flux_time_type",'Hour/Day')
        cp.data_set_number(r,"fluxsim.0.flux_month",M)
        cp.data_set_number(r,"fluxsim.0.flux_day",  D)
        cp.data_set_number(r,"fluxsim.0.flux_hour", 11.819)
        ##

        ## have we already found the number of heliostats to disable?
        is_aligned =True if self.Ndisable != None else False

        Qdes        = inputs["receiver.0.q_des"]
        cp.simulate(r)  # simulate the scenario as is
        inc_flux = cp.get_fluxmap(r) #should return the incident flux NOT including absorbtivity
        H=cp.data_get_number(r,"receiver.0.rec_height")
        W=cp.data_get_number(r,"receiver.0.rec_width")
        Arec=H*W
        Qsinc_kW=np.mean(inc_flux)*Arec
        Qsinc_MW=Qsinc_kW/1000
        eta_r = 0.91 # estimated overall receiver efficiency
        Qfluid_est=2000 # some impossibly high MWth starting value to get loop going
        
        Ndisable=self.Ndisable if is_aligned else 0
        if not is_aligned:
            print(f'Aligning field with Qdes based on solar conditions at {M} month, {D} day, {11.819} hour')
            iterate=True
            while iterate== True:
            ## assign the og layout
                self.assign_gend_layout(cp,r,inputs)
                ## get the entire field's info, including ID's
                field_entire = cp.get_layout_info(r)
                disabled_ids = self.disable_heliostats(cp,r,field_entire,N=Ndisable)
                cp.simulate(r)  # simulate the scenario as is
                inc_flux = cp.get_fluxmap(r) #should return the incident flux NOT including absorbtivity
                Qsinc_kW=np.mean(inc_flux)*Arec
                Qsinc_MW=Qsinc_kW/1000
                Qfluid_est=Qsinc_MW*eta_r

                if Qfluid_est<Qdes: # evaluate loop criteria
                    iterate = False
                else:
                    Ndisable=Ndisable+20 # remove more heliostats on next iteration
        elif is_aligned:
            self.assign_gend_layout(cp,r,inputs)
            ## get the entire field's info, including ID's
            field_entire = cp.get_layout_info(r)
            disabled_ids = self.disable_heliostats(cp,r,field_entire,N=Ndisable)

        print('removed: ',Ndisable,' heliostats')
        return Ndisable
#==========================================================================   

# Set ssc data values from dictionary object 
def _set_ssc_data_from_dict(ssc_api, ssc_data, Dict):
    for key in Dict.keys():
        try:
            if type(Dict[key]) in [type(1), type(1.), type(np.ones(1, dtype=int)[0]), type(np.ones(1)[0])]:
               ssc_api.data_set_number(ssc_data, key.encode("utf-8"), Dict[key])
           
            elif type(Dict[key]) == type(True):
               ssc_api.data_set_number(ssc_data, key.encode("utf-8"), 1 if Dict[key] else 0)
           
            elif type(Dict[key]) == type(""):
               ssc_api.data_set_string(ssc_data, key.encode("utf-8"), Dict[key].encode("utf-8"))
               
            elif type(Dict[key]) == type([]):
               if len(Dict[key]) > 0:
                   if type(Dict[key][0]) == type([]):
                       ssc_api.data_set_matrix(ssc_data, key.encode("utf-8"), Dict[key])
                   else:
                       ssc_api.data_set_array(ssc_data, key.encode("utf-8"), Dict[key])
               else:
#                   print ("Did not assign empty array " + key)
                   pass
            else:
               print ("Could not assign variable " + key )
               raise KeyError
        except:
            print ("Error assigning variable " + key + ": bad data type")

# def _set_ssc_layout_and_aiming(ssc_api, ssc_data):
#     # get the aiming scheme and heliostat field file names from json
# DOESN"T WORK BECAUSE SSC DOESN"T HAVE THE FUNCTIONS I NEED
#     print('--HARDCODED HELIOSTAT AND AIMING FILES--')
#     aiming_file='aiming_scheme.csv'
#     layout_file='200MWt_layout.csv'
#     # get an instance of copylot for the apply_layout method
#     cp = CoPylot()

#     # # set aiming scheme
#     ssc_api.data_set_matrix_from_csv(ssc_data, "n_user_flux_profile".encode("utf-8"), aiming_file) # doesn't work because "double" not defined in Janna's API
#     # with open(aiming_file, 'r') as file:
#     #     csv_reader = csv.reader(file,delimiter=',')
#     #     flux_pts_str= list(csv_reader)
#     #     ncols = len(flux_pts_str)
#     #     flux_pts =  [float(element) for row in flux_pts_str for element in row] #convert all strings to floats
#     #     flux_pts = [flux_pts[i:i + ncols ] for i in range(0, len(flux_pts), ncols)] # reshape back to original
#     # # cp.data_set_matrix(ssc_data, "receiver.0.n_user_flux_profile", flux_pts )
#     # ssc_api.data_set_matrix(ssc_data, "receiver.0.n_user_flux_profile", flux_pts)

#     # # set heliostat layout using copylot method
#     # with open(layout_file, 'r') as file:
#     #     csv_reader = csv.reader(file,delimiter=',')
#     #     helio_list_str= list(csv_reader)
#     #     ncols = len(helio_list_str[0])
#     #     helio_list =  [float(element) for row in helio_list_str for element in row] #convert all strings to floats
#     #     helio_list = [helio_list[i:i + ncols ] for i in range(0, len(helio_list), ncols)] # reshape back to original
#     # cp.assign_layout(ssc_data, helio_list)
#     # # ssc_api.assign_layout(ssc_data, helio_list, 1)





#==========================================================================   
# Assign default values to all SolarPILOT inputs
def _set_defaults(ssc_api, ssc_data):
    ssc_api.data_set_number(ssc_data, 'q_design'.encode("utf-8"), 670.)
    ssc_api.data_set_number(ssc_data, 'rec_height'.encode("utf-8"), 21.6)    
    ssc_api.data_set_number(ssc_data, 'rec_aspect'.encode("utf-8"), 21.6/17.65)  
    ssc_api.data_set_number(ssc_data, 'h_tower'.encode("utf-8"), 193.4575)  
    ssc_api.data_set_string(ssc_data, 'solar_resource_file'.encode("utf-8"), 'ssc/USA CA Daggett (TMY2).csv'.encode("utf-8"))     
    ssc_api.data_set_number(ssc_data, 'helio_width'.encode("utf-8"), 12.2)
    ssc_api.data_set_number(ssc_data, 'helio_height'.encode("utf-8"), 12.2)   
    ssc_api.data_set_number(ssc_data, 'helio_optical_error'.encode("utf-8"), 0.00153)
    ssc_api.data_set_number(ssc_data, 'helio_active_fraction'.encode("utf-8"), 0.99)
    ssc_api.data_set_number(ssc_data, 'dens_mirror'.encode("utf-8"), 0.97)
    ssc_api.data_set_number(ssc_data, 'helio_reflectance'.encode("utf-8"), 0.90)
    ssc_api.data_set_number(ssc_data, 'rec_absorptance'.encode("utf-8"), 0.94)
    ssc_api.data_set_number(ssc_data, 'rec_hl_perm2'.encode("utf-8"), 30)
    ssc_api.data_set_number(ssc_data, 'dni_des'.encode("utf-8"), 950)
    ssc_api.data_set_number(ssc_data, 'land_max'.encode("utf-8"), 9.5)
    ssc_api.data_set_number(ssc_data, 'land_min'.encode("utf-8"), 0.75)
    ssc_api.data_set_number(ssc_data, 'c_atm_0'.encode("utf-8"), 0.0067889997735619545)
    ssc_api.data_set_number(ssc_data, 'c_atm_1'.encode("utf-8"), 0.10459999740123749)
    ssc_api.data_set_number(ssc_data, 'c_atm_2'.encode("utf-8"), -0.017000000923871994)
    ssc_api.data_set_number(ssc_data, 'c_atm_3'.encode("utf-8"),  0.002845000009983778)
    ssc_api.data_set_number(ssc_data, 'n_facet_x'.encode("utf-8"), 2)
    ssc_api.data_set_number(ssc_data, 'n_facet_y'.encode("utf-8"), 8)
    ssc_api.data_set_number(ssc_data, 'focus_type'.encode("utf-8"), 1)
    ssc_api.data_set_number(ssc_data, 'cant_type'.encode("utf-8"), 1)
    ssc_api.data_set_number(ssc_data, 'n_flux_days'.encode("utf-8"), 8)
    ssc_api.data_set_number(ssc_data, 'delta_flux_hrs'.encode("utf-8"), 2)    
    ssc_api.data_set_number(ssc_data, 'calc_fluxmaps'.encode("utf-8"),0)
    ssc_api.data_set_number(ssc_data, 'n_flux_x'.encode("utf-8"), 12)
    ssc_api.data_set_number(ssc_data, 'n_flux_y'.encode("utf-8"), 1)
    ssc_api.data_set_number(ssc_data, 'check_max_flux'.encode("utf-8"), 0)
    ssc_api.data_set_number(ssc_data, 'tower_fixed_cost'.encode("utf-8"), 3000000)
    ssc_api.data_set_number(ssc_data, 'tower_exp'.encode("utf-8"), 0.0113)
    ssc_api.data_set_number(ssc_data, 'rec_ref_cost'.encode("utf-8"), 103000000)
    ssc_api.data_set_number(ssc_data, 'rec_ref_area'.encode("utf-8"), 1571)
    ssc_api.data_set_number(ssc_data, 'rec_cost_exp'.encode("utf-8"), 0.7)
    ssc_api.data_set_number(ssc_data, 'site_spec_cost'.encode("utf-8"), 16)
    ssc_api.data_set_number(ssc_data, 'heliostat_spec_cost'.encode("utf-8"), 145)
    ssc_api.data_set_number(ssc_data, 'land_spec_cost'.encode("utf-8"), 10000)
    ssc_api.data_set_number(ssc_data, 'contingency_rate'.encode("utf-8"), 7)
    ssc_api.data_set_number(ssc_data, 'sales_tax_rate'.encode("utf-8"), 5)
    ssc_api.data_set_number(ssc_data, 'sales_tax_frac'.encode("utf-8"), 80)
    ssc_api.data_set_number(ssc_data, 'cost_sf_fixed'.encode("utf-8"), 0)
    ssc_api.data_set_number(ssc_data, 'is_optimize'.encode("utf-8"), 0)
    ssc_api.data_set_number(ssc_data, 'flux_max'.encode("utf-8"), 1000)
    ssc_api.data_set_number(ssc_data, 'opt_init_step'.encode("utf-8"), 0.06)
    ssc_api.data_set_number(ssc_data, 'opt_max_iter'.encode("utf-8"), 200 )
    ssc_api.data_set_number(ssc_data, 'opt_conv_tol'.encode("utf-8"), 0.001)
    ssc_api.data_set_number(ssc_data, 'opt_flux_penalty'.encode("utf-8"), 0.25)
    ssc_api.data_set_number(ssc_data, 'opt_power_penalty'.encode("utf-8"), 2.0)
    return


#========================================================================== 
# Set default SolarPILOT inputs, replace those specified in "inputs" dictionary, and run solarpilot. 
def run_solarpilot(inputs, savepositions = True, verbose = False):   
    ssc_api = PySSC.PySSC()
    ssc_api.module_exec_set_print(0)  #0 = no, 1 = yes (print progress updates)
    dat = ssc_api.data_create()
    solarpilot = ssc_api.module_create("solarpilot".encode("utf-8"))
    
    ### the two ways inputs are entered to the ssc
    _set_defaults(ssc_api, dat)                    # Set default inputs
    _set_ssc_data_from_dict(ssc_api, dat, inputs)  # Update from currently defined inputs
    ###
    # adding my method to set heliostat field and aiming scheme DIDN"T WORK
    # _set_ssc_layout_and_aiming(ssc_api, dat)
    ###
    
    #ssc_api.module_exec_set_print( 1 )
    ssc_api.module_exec(solarpilot , dat)
    if (ssc_api.module_exec(solarpilot , dat) == 0):       
        print(ssc_api.module_log(solarpilot , 0))     
    
    results = {} 
    
    # Extract data from optimization messages and display in console
    if 'is_optimize' in inputs.keys() and inputs['is_optimize']:
        msgs = []
        for j in range(100):
            msg = ssc_api.module_log(solarpilot , j)
            msgs.append(msg)
            if msg is None:
                break
            if verbose:
                print (msg)
            if msg[0:2] == '--' and msgs[-2][0] == '[':  # Optimization is finished
                sep = msgs[-2].split('|')
                optflux = float(sep[-2].strip(' '))  # Flux from optimization
                obj = float(sep[-3].strip(' '))      # Objective function from optmization
                results['optimization_objective'] = obj
                results['optimization_flux'] = optflux
                print ('Peak flux from optimization = %.0f kW/m2'%optflux)
            if msg[0:9] == 'Algorithm':
                break

    results['solar_resource_file'] = ssc_api.data_get_string(dat, 'solar_resource_file'.encode("utf-8")) 
    for k in ['number_heliostats', 'flux_max_observed', 'area_sf', 'rec_height_opt', 'rec_aspect_opt', 'h_tower_opt', 'dni_des', 'flux_max', 'rec_absorptance', 'q_design']:
        results[k] = ssc_api.data_get_number(dat, k.encode("utf-8"))  
       
    for k in ['opteff_table', 'flux_table']:
        results[k] = ssc_api.data_get_matrix(dat, k.encode("utf-8"))  
    
    if savepositions:
        results['heliostat_positions'] = ssc_api.data_get_matrix(dat, 'heliostat_positions'.encode("utf-8"))  
        
    ssc_api.module_free(solarpilot)
    ssc_api.data_free(dat)

    return results

def run_solarpilot_wCP(inputs, savepositions = True, verbose = False):
    cp = CoPylot() # create copylot instance
    r  = cp.data_create() # specific instance case. R is just a memory ID
    
    _set_cp_data_from_dict(cp, r, inputs)  # Update from currently defined inputs

    layout_file = inputs["layout_file"]
    aiming_file = inputs["aiming_file"]

    with open(layout_file, 'r') as file:
        # print("HARDCODING HELIOSTAT LAYOUT")
        csv_reader = csv.reader(file,delimiter=',')
        helio_list_str= list(csv_reader)
        ncols = len(helio_list_str[0])
        helio_list =  [float(element) for row in helio_list_str for element in row] #convert all strings to floats
        helio_list = [helio_list[i:i + ncols ] for i in range(0, len(helio_list), ncols)] # reshape back to original
    cp.assign_layout(r, helio_list,1)

    # cp.data_set_matrix_from_csv(r, "receiver.0.user_flux_profile", aiming_file ) ### commented out on 1/27/25 for testing

    cp.simulate(r)
    field = cp.get_layout_info(r)
    flux = cp.get_fluxmap(r)

    
    # plt.scatter(field['x_location'], field['y_location'], s=1.5)
    # plt.tight_layout()
    # plt.show()

    # # flux
    # im = plt.imshow(flux)
    # plt.colorbar(im)
    # plt.tight_layout()
    # plt.show()

    ###
    # postprocess results 
    results={}

    results['solar_resource_file'] = cp.data_get_string(r, 'ambient.0.weather_file')

    # add various metrics to results. 1/17/2025: couldn't resolve: number_heliostats, flux_max_observed, flux_max 
    for name,id in zip([ 'area_sf',                'rec_height_opt',       'rec_aspect_opt',           'h_tower_opt',              'dni_des',           'rec_absorptance',          'q_design'],
                        [ 'solarfield.0.sf_area', 'receiver.0.rec_height', 'receiver.0.rec_aspect', 'receiver.0.optical_height', 'receiver.0.dni_des', 'receiver.0.absorptance', 'solarfield.0.q_des']
                    ):
        print(name)
        results[name] = cp.data_get_number(r, id)  
    # opt_eff_table in normal run yields a 22 x 3 list of percentages.
    # flux table yields a 660 x 30

    # currently I just get one 30 x 30 flux result. I need to run solarpilot for each timepoint
    cp.data_free(r)

    return results

def _set_cp_data_from_dict(cp, r, Dict):
    for key in Dict.keys():
        try:
            if type(Dict[key]) in [type(1), type(1.), type(np.ones(1, dtype=int)[0]), type(np.ones(1)[0])]:
               cp.data_set_number(r, key, Dict[key])
           
            elif type(Dict[key]) == type(True):
               cp.data_set_number(r, key, 1 if Dict[key] else 0)
           
            elif type(Dict[key]) == type(""):
               cp.data_set_string(r, key, Dict[key])
               
            elif type(Dict[key]) == type([]):
               if len(Dict[key]) > 0:
                   if type(Dict[key][0]) == type([]):
                       cp.data_set_matrix(r, key, Dict[key])
                   else:
                       cp.data_set_array(r, key, Dict[key])
               else:
#                   print ("Did not assign empty array " + key)
                   pass
            else:
               print ("Could not assign variable " + key )
               raise KeyError
        except:
            print ("Error assigning variable " + key + ": bad data type")




    