"""
created by jwenner on 7/11/2025
purpose is to provide auxiliary functions for lightweight and full thermal models, which only solves the receiver for one flux profile

"""

import settings
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import billboard_receiver
import util
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import materials
from scipy.optimize import curve_fit

def np_to_json(vars, vals, filename):
    """
    vars: a list of variable names that will be dictionary keys in the json
    vals: (list of np arrays) associated with the keys
    filename: (str)
    """
    dict = {}   #initialize an empty dictionary
    with open(filename, "w") as f:
        for var,val in zip(vars,vals):
            dict[var]= val.tolist()
        json.dump(dict, f) 

def json_to_np(filename,var):
    """
    filename: json file containing a dictionary of temperatures of interest
    variable of interest
    returns np array for variable of interest
    """
    with open(filename,) as f:
        dict=json.load(f)
    return np.array(dict[var])


def increase_flux_resolution_blocked(flux_low_res,ndim_new):
    """
    take in a low resolution flux, repeat the values to a higher resolution
    flux_low_res    - (kW/m2) square array of flux values, likely from SolarPILOT
    new_ndim        - 
    ---
    returns:
    flux_high_res   - (kW/m2) block-increased array of flux values for thermal model
    """
    ndim_og = flux_low_res.shape[0]
    ndim_ogB= flux_low_res.shape[1] 
    if ndim_og !=ndim_ogB:
        print('warning! blocking an unsquare array')
    if ndim_new % ndim_og != 0:
        print('resquested dimension must be a multiple of original dimension')
    else:
        flux_high_res = np.repeat(flux_low_res, int(ndim_new/ndim_ogB), axis = 0 )
        flux_high_res = np.repeat(flux_high_res, int(ndim_new/ndim_ogB), axis = 1 )

    
    return flux_high_res

def increase_flux_resolution_blocked_custom(flux_low_res, npts_horizontal, npts_vertical):
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

def plot_flux_w_panels(Ws,W,H,flux,save=None,cmap='viridis',vmin=None,vmax=None):
    """
    plot a flux distribution and overlay the panels on top of the image
    Ws - iterable of widths
    W  - total receiver width
    H  - total receiver height
    flux - array of fluxes
    """
    fontsize=16
    fig,ax = plt.subplots()
    im = ax.imshow(flux,extent=[0,W,0,H],origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
    cbr = fig.colorbar(im)
    cbr.set_label(label='flux [kW/m$^2$]', size=fontsize)
    cbr.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('x [m]',size=fontsize)
    ax.set_ylabel('y [m]',size=fontsize)
    
    for xlabel_i in ax.axes.get_xticklabels():
        xlabel_i.set_fontsize(fontsize)
    #     xlabel_i.set_visible(False)
    for xlabel_i in ax.axes.get_yticklabels():
        xlabel_i.set_fontsize(fontsize)
        # xlabel_i.set_visible(False)
    # for tick in ax.axes.get_xticklines():
    #     tick.set_visible(False)
    # for tick in ax.axes.get_yticklines():
    #     tick.set_visible(False)
    ### plot the panel width locations
    panel_rhs=0 # starting from left side
    for panel_w in Ws:
        panel_rhs=panel_rhs+panel_w
        ax.vlines(x=panel_rhs,ymin=0,ymax=H, colors='goldenrod')
    if save:
        plt.savefig(save)

    plt.show()
    # plt.close()
    return  

def make_results_file(inputName,outputName,SS_obj,d):
    """
    take the json inputs file, and add key results from a thermal model results object. Only works with full model

    filename - (.json) input file used to run thermal simulation
    SS_obj - steady state object from thermal model. Should have pre-generated results for access
    d - the timepoint of interest. Usually design point
    """
    ## make new dictionary with things to add
    results_dict = {}
    Qfield = SS_obj.results.Qsinc[d] / SS_obj.results.eta_field
    results_dict['Qsinc'] = SS_obj.results.Qsinc[d]
    results_dict['eta_overall'] = SS_obj.results.eta_overall[d]
    results_dict['inc_flux_max']=SS_obj.results.abs_flux[...,d].max() / SS_obj.receiver.solar_abs
    results_dict['htube_avg']= SS_obj.results.htube_avg[d]
    results_dict['mflow']=SS_obj.results.mflow[d]
    results_dict['Qfluid']=SS_obj.results.Qfluid[d]
    results_dict['Ndisabled_heliostats']=SS_obj.results.Ndisable
    ## get all inputs from json input file
    inputName = inputName + '.json'
    outputName= outputName+ '.json'
    with open(inputName,) as f:
        report_dict=json.load(f)
    ## combine dictionaries
    report_dict.update(results_dict)
    ## put both dictionaries in a new file
    with open(outputName, "w") as f:
        json.dump(report_dict, f)
    return True

def make_timeseries_results_file(inputName,outputName,SS_obj):
    """
    take the json inputs file, and add key results from a thermal model results object. Only works with full model

    Intended for timeseries studies that return values at each solved timepoint

    filename - (.json) input file used to run thermal simulation
    SS_obj - steady state object from thermal model. Should have pre-generated results for access
    d - the timepoint of interest. Usually design point
    """
    ## make new dictionary with things to add
    results_dict = {}
    results_dict['Qsinc'] = SS_obj.results.Qsinc.tolist()
    results_dict['htube_avg']= SS_obj.results.htube_avg.tolist()
    results_dict['mflow']=SS_obj.results.mflow.tolist()
    results_dict['Qfluid']=SS_obj.results.Qfluid.tolist()
    results_dict['Qpump']=SS_obj.results.Qpump.tolist()
    results_dict['eta_field']=SS_obj.results.eta_field.tolist()
    results_dict['eta_opt']=(SS_obj.results.eta_field*SS_obj.receiver.solar_abs).tolist()
    results_dict['times']=SS_obj.results.time.tolist()
    results_dict['Ndisabled_heliostats']=SS_obj.results.Ndisable
    ## get all inputs from json input file
    inputName = inputName + '.json'
    outputName= outputName+ '_timeseries.json'
    with open(inputName,) as f:
        report_dict=json.load(f)
    ## combine dictionaries
    report_dict.update(results_dict)
    ## put both dictionaries in a new file
    with open(outputName, "w") as f:
        json.dump(report_dict, f)
    return True

def get_thermal_results(receiver):
    """
    receiver - object from NREL thermal model. Should contain some results
    ----
    returns z x npanels x ntubes shape array of total temperature differences (C), fluid temperatures (C), and absorbed flux (kW/m2)
        also returns Rs, which is the temperature difference ratio and helps determine what LTE lookup table to use. 
    """
    K_offset = 273.15 

    Tw_inner_low    = receiver.get_array_of_Tinner_low_axial_profiles('Tw') - K_offset # this only works with attribute 'Tw'
    Tw_inner_high   = receiver.get_array_of_Tinner_high_axial_profiles('Tw') - K_offset
    Tw              = receiver.get_array_of_axial_profiles('Tw') - K_offset # this is the maximum wall temperature
    
    dTs             = Tw - Tw_inner_low
    dTs_wall        = Tw - Tw_inner_high
    dTs_conv        = Tw_inner_high - Tw_inner_low
    Rs              = dTs_wall/dTs_conv

    Tfs             = receiver.get_array_of_axial_profiles('Tf')-K_offset
    qabs            = receiver.get_array_of_axial_profiles('inc_flux')*receiver.solar_abs
    
    ### orient results
    npanels     = dTs.shape[1]
    ntubes_sim  = dTs.shape[2]
    for p in range(npanels):
        if receiver.tubes[p][0].flow_against_gravity:
            dTs[:,p,:]  = dTs[::-1,p,:]
            Tfs[:,p,:]  = Tfs[::-1,p,:]
            qabs[:,p,:] = qabs[::-1,p,:]
            Rs[:,p,:]   = Rs[::-1,p,:]

    return dTs, Tfs, qabs, Rs

def plot_results(receiver, results, label_name, savename='default_receiver_heatmap', vmin=0, vmax=1):
    """
    makes a heatmap using imshow, based on receiver width and height. Resolution is dependent on number of tubes/panel
    --
    results - a nz x npanels x ntubes/panel size array of whatever result you want to plot
    """
    nz  =results.shape[0]
    W   =receiver.D
    H   =receiver.H
    cbr_fontsize =12
    fig,ax=plt.subplots()
    if label_name == 'lifetimes (yrs)':
        cmap_choice ='plasma_r'
    else: 
        cmap_choice ='inferno'
    im = ax.imshow(results.reshape(nz,receiver.Npanels*receiver.ntubesim), extent=[-W/2, W/2, 0, H], vmin=vmin, vmax=vmax, cmap=cmap_choice )
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    divider = make_axes_locatable(ax)  
    cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
    cbr = fig.colorbar(im, cax=cax)
    cbr.set_label(label=label_name, size=cbr_fontsize)
    cbr.ax.tick_params(labelsize=cbr_fontsize)
    fig.savefig(f'imgs/{savename}',dpi=300)
    plt.show()
    plt.close(fig)
    return 


def setup_LWT_thermal_model(filestring,ntubes_sim):
    """
    sets up a lightweight NREL model based on receiver .json and some other assumptions
    ---
    filestring - a full relative string used to find the receiver file
    """
    ### set up numerical resolution for flux and model
    nr = 2      # was 5
    ntheta = 79 # was 79
    nz = 50     # was 50 

    ### instantiate a receiver object
    receiver=billboard_receiver.BillboardReceiver()     # make a receiver object
    receiver.load_from_json(filestring)    # set the receiver inputs using json

    # manually set operating conditions for timepoint
    receiver.operating_conditions.hour_offset = 0
    mDot_total=(receiver.Qdes*1e6)/receiver.HTFcpavg/(receiver.Tfout_design-receiver.Tfin_design)
    mass_flow=[mDot_total/receiver.npaths for path in range(receiver.npaths)]
    receiver.operating_conditions.mass_flow = mass_flow 
    receiver.operating_conditions.vwind10 = 0           # (m/s) set windspeed to be zero
    receiver.operating_conditions.rh = 25                  # relative humidity
    receiver.operating_conditions.Tamb = 298.15         # (K)  ambient temperature

    receiver.operating_conditions.Tambrad = util.calculate_sky_temperature(receiver.operating_conditions.Tamb, receiver.operating_conditions.rh, receiver.operating_conditions.hour_offset)
    receiver.operating_conditions.Tfin = receiver.Tfin_design

    # settings
    #--- Update numerical solution options as desired (all inputs have default values defined in the CylindricalReceiver class)
    receiver.disc = settings.Discretization(nr, ntheta, nz)   # number of r, theta, z nodes 
    dimensions = '1D'
    receiver.options.wall_detail = dimensions
    receiver.options.calculate_stress = False         # Calculate elastic thermal stress distributions?
    receiver.flow_control_mode = 0                    # 0 = Control each path independently
    receiver.ntubesim = ntubes_sim
    receiver.options.use_full_rad_exchange = True


    receiver.initialize()
    ##--- update tube inputs to reduce solution time. Introduces average of 2.5% error in lifetime according to run_ideal_and_uniform_cases
    new_settings = {'crosswall_avg_k':True, 'use_full_rad_exchange': receiver.options.use_full_rad_exchange, 'wall_detail': dimensions}
    receiver.update_tube_inputs(new_settings) # 

    return receiver

def solve_LWT_thermal_model(receiver,flux):
    """
    solves LWT thermal model based on a given flux
    --
    receiver: object instantiated from NREL parent code
    flux: (kW/m2) array of receiver flux values, must match the flux resolution specified in the receiver model
    --
    returns ntubes x nz arrays of the big four results in C, C, kW/m2, and (-) units respectively
    """
    receiver.operating_conditions.inc_flux = flux # (kW/m2) flux profile dimensions are z x circ/width
    receiver.update_tube_operating_conditions(True)
    receiver.solve_steady_state_profiles(allow_initial_guess=False) 

    dTs, Tfs, qabs, Rs = get_thermal_results(receiver)

    return dTs, Tfs, qabs, Rs

def parabolic_fun(x, p1, p2, p3):
    return p1*(x-p2)**2 + p3

def eta_fun(x, a, b):
    return a - np.exp(-b*x)


def interp_Qs(model, sample_times, sample_DNIs):
        '''
        Takes solved model and creates curve fits for receiver power and pump power vs dni.
        Makes season-specific curve fits for variables
        --- inputs:
        model       - steady state analysis model that has been solved for the desired day.
        DNI_samples - array of DNIs (W/m2) at which we want to predict Qfluid and Qpump. Need not be the same as the simulated DNI points
        --- returns:
        Qfluids_sample - an array of fluid powers (W) for each DNI sample point
        Qpumps_sample  - an array of total pump power required (W) for each DNI sample point
        '''
        ## put the values we want in new variables
        Qfluids_model   =model.results.Qfluid
        Qpumps_model    =model.results.Qpump
        DNIs_model      =model.results.dni
        Qinc_model      =model.results.Qsinc   # incident power for each timestep
        eta_model       =model.results.eta  # receiver's thermal efficiency (Qfluid/Qsinc)


        ### Qfluid: interpolate field optical efficiency and receiver efficiency to equate
        # print('note: assuming DNI and Qfluid always increase/decrease simulataneously')
        # ps, pcovs                   =curve_fit(parabolic_fun, DNIs_model, Qfluids_model, maxfev=100000) # high maxfev but necessary for the winter fit
        # p1, p2, p3                  =ps
        # Qfluids_samples                                 =parabolic_fun(DNI_samples, p1, p2, p3)
        # Qfluids_samples[DNI_samples < model.dni_cutoff] =0
        eta_field_samples     =np.interp(sample_times, model.results.time, model.results.eta_field)

        # ## plot the field efficiencies to check if desired
        # fig,ax  =plt.subplots()
        # fontsize=14
        # ax.scatter(model.results.time, model.results.eta_field, label='model',color='k',marker='s',s=15)
        # ax.scatter(sample_times, eta_field_samples, label='from fit',color='orangered',s=15,marker='^')
        # ax.plot(model.results.time, model.results.eta_field,color='gray', linestyle='--', linewidth=1.5)

        # ax.set_xlabel('simulation or sample time (hr)', fontsize=fontsize)
        # ax.set_ylabel('field efficiency', fontsize=fontsize)
        # ax.tick_params('both', labelsize=fontsize)
        # ax.legend(fontsize=fontsize)
        # # ax.set_ylim(0,3)
        # # ax.set_xlim(time)
        # ax.grid(True,alpha=0.5)
        # fig.savefig('imgs/default_save_field_eff_curve.png', dpi=300)
        # plt.show()
        # plt.close()

        ## calculate the sampled incident power
        n_hstats_used   =pd.read_csv(model.receiver.layout_file).index.max()+1-model.SF.Ndisable    # get total number of mirrors, subtract the number of disabled heliostats
        print('assuming heliostats each have area 144 m2')
        A_sf            =144.4*n_hstats_used
        Qsinc_samples   =A_sf*sample_DNIs.squeeze()*eta_field_samples
        Qsinc_samples_MW=Qsinc_samples/1e6

        ## interpolate the receiver efficiency as a function of incident power
        Qinc_model_MW               =Qinc_model/1e6     # input to this curve fit needs to be lower magnitude. Large disparity in x,y ranges leads to poor fit performance
        ps, pcovs                   =curve_fit(parabolic_fun, Qinc_model_MW, eta_model, maxfev=100000) # high maxfev but necessary for the winter fit
        p1_eff, p2_eff, p3_eff      =ps
        eta_samples                 =parabolic_fun(Qsinc_samples_MW, p1_eff, p2_eff, p3_eff)

        ## calculate the fluid power at sampling points with interpolated inc. flux, interpolated rec. eff, and assuming absorptivity
        Qfluids_samples             =Qsinc_samples*eta_samples # eta already includes solar absorptivity
        Qfluids_samples[sample_DNIs.squeeze() < model.dni_cutoff] =0

        # ## plot the receiver efficiency versus incident power as a logic check
        # fig,ax  =plt.subplots()
        # fontsize=14
        # ax.scatter(Qinc_model/1e6, eta_model, label='model',color='k',marker='s',s=15)
        # ax.scatter(Qsinc_samples/1e6, eta_samples, label='from fit',color='orangered',s=15,marker='^')
        # ax.plot(np.linspace(0,Qinc_model_MW.max()), 
        #         parabolic_fun(np.linspace(0,Qinc_model_MW.max()),p1_eff, p2_eff, p3_eff), label=' model fit', linestyle='--',color='gray',linewidth=1.5)

        # ax.set_xlabel('incident receiver power (MWth)', fontsize=fontsize)
        # ax.set_ylabel('receiver efficiency', fontsize=fontsize)
        # ax.tick_params('both', labelsize=fontsize)
        # ax.legend(fontsize=fontsize)
        # ax.grid(True,alpha=0.5)
        # fig.savefig('imgs/default_save_receiver_eff_curve.png', dpi=300)
        # plt.show()
        # plt.close()

        # ## plot the power to fluid as a logic check
        # fig,ax  =plt.subplots()
        # fontsize=14
        # ax.scatter(model.results.time, Qfluids_model/1e6, label='model', s=15, marker='s',color='k')
        # ax.scatter(sample_times, Qfluids_samples/1e6, label='interpolated', s=15, marker='^', color='orangered')
        # # ax2=ax.twinx()

        # # # overlay the dni of clear sky and seasonal for reference
        # # ax2.plot(model.results.time, model.results.dni, linewidth=1.5, label='model DNI', color='k')
        # # ax2.plot(sample_times, sample_DNIs, linewidth=1.5, label='seasonal DNI', color='orangered')
        # # ax2.set_ylabel('Direct Normal Irradiance (W/m2)', fontsize=fontsize)
        # ax.set_xlabel('time (hr)', fontsize=fontsize)
        # ax.set_ylabel('power to fluid (MWth)', fontsize=fontsize)
        # ax.tick_params('both', labelsize=fontsize)
        # ax.legend(fontsize=fontsize-2)
        # # ax2.legend(fontsize=fontsize-2)
        # ax.grid(True, alpha=0.5)
        # ax.set_xlim(0,24)
        # fig.savefig('imgs/default_save_fit_of_Qfluid.png',dpi=300)
        # plt.show()
        # plt.close()


        ## interpolate pump values at the DNI levels dictated by the summer solstice
        pump_ps, pcovs                   =curve_fit(parabolic_fun, Qfluids_model, Qpumps_model)
        pump_p1, pump_p2, pump_p3        =pump_ps

        Qpumps_samples                      =parabolic_fun(Qfluids_samples, pump_p1, pump_p2, pump_p3)
        Qpumps_samples[Qfluids_samples < 1] =0
        

        # ## check pump power plot to ensure validity
        # fig,ax  =plt.subplots()
        # fontsize=14
        # ax.scatter(Qfluids_model/1e6, Qpumps_model/1e6, label='model',color='k',marker='s',s=15)
        # ax.scatter(Qfluids_samples/1e6, Qpumps_samples/1e6, label='from fit',color='orangered',s=15,marker='^')
        # ax.plot(np.linspace(0,Qfluids_model.max())/1e6, 
        #         parabolic_fun(np.linspace(0,Qfluids_model.max()),pump_p1, pump_p2, pump_p3)/1e6, label=' model fit', linestyle='--',color='gray',linewidth=1.5)

        # ax.set_xlabel('power to fluid (MWth)', fontsize=fontsize)
        # ax.set_ylabel('required pumping power (MWth)', fontsize=fontsize)
        # ax.tick_params('both', labelsize=fontsize)
        # ax.legend(fontsize=fontsize)
        # ax.set_ylim(-0.1,3)
        # ax.set_xlim(-0.1,model.receiver.Qdes)
        # ax.grid(True,alpha=0.5)
        # fig.savefig('imgs/default_save_pump_power_curve.png', dpi=300)
        # plt.show()
        # plt.close()

        return Qfluids_samples, Qpumps_samples
