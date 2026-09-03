"""
Created by jwenner on 11/17/2025 to evaluate the annual LCOH for a range of aspect ratios
"""
## solve the thermal model at each of the 4 seasons

## calculate the total heating and pumping powers for every hour of the year

## calculate the total number of heliostats and required field

## calculate LCOH


## import stuff
import sys
sys.path.append('./LTE/') # thermal model is in upper level folder
import os
import steady_state_analysis_jwenn
import steady_state_plots
import settings
import timeit
import numpy as np
import matplotlib.pyplot as plt
import tube_jwenner
import numpy as np
import pandas as pd
import helpers_thermal_model
import json
import damage_tool
import matplotlib.ticker as tck
from scipy.optimize import curve_fit
import tower_model
from mpl_toolkits.axes_grid1 import make_axes_locatable


## adapted from helpers_thermal_model
def plot_results(W, H, Npanels, ntubesim, nz, results, label_name, savename='default_receiver_heatmap', vmin=0, vmax=1):
    """
    makes a heatmap using imshow, based on receiver width and height. Resolution is dependent on number of tubes/panel
    --
    results - a nz x npanels x ntubes/panel size array of whatever result you want to plot
    """
    cbr_fontsize =12
    fig,ax=plt.subplots(tight_layout=True)
    if label_name == 'lifetimes (yrs)' or label_name == 'log10(lifetimes (yrs))':
        cmap_choice ='plasma_r'
    else: 
        cmap_choice ='inferno'
    im = ax.imshow(results.reshape(nz,Npanels*ntubesim), extent=[-W/2, W/2, 0, H], vmin=vmin, vmax=vmax, cmap=cmap_choice )
    ax.set_xlabel('x position (m)',fontsize=cbr_fontsize+2)
    ax.set_ylabel('y position (m)',fontsize=cbr_fontsize+2)
    divider = make_axes_locatable(ax)  
    cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
    cbr = fig.colorbar(im, cax=cax)
    cbr.set_label(label=label_name, size=cbr_fontsize)
    cbr.ax.tick_params(labelsize=cbr_fontsize)
    fig.savefig(f'{savename}',dpi=300)
    #plt.show()
    plt.close(fig)
    return 
##

if __name__ == '__main__':
    # ARs =[0.83,]
    ## initialize the damage model
    matl    ='A230'
    dmg_obj =damage_tool.damageTool(mat_string=matl, interp_mode='LNDI')    # set interpolation mode to LNDI - improves accuracy and performance

    ## set design lifetime
    N_life =30  # (yrs)
    
    

    ## set up the full steady state analysis object
    model =steady_state_analysis_jwenn.SteadyStateAnalysis()

    AR         =0.83
    input_name =f'receivers/AR_study/billboard_220Qdes_{AR:.2f}AR_spt_demo'
    model.receiver.load_from_json(input_name)

    model.receiver.disc                     =settings.Discretization(5, 99, model.receiver.n_flux_y)   # standard is 5, 79, 50. number of r, theta, z nodes 
    model.receiver.options.wall_detail      ='2D' 
    model.receiver.options.calculate_stress =False   # Calculate elastic thermal stress distributions?
    model.receiver.flow_control_mode        =0             # 0 = Control each path independently
    model.receiver.ntubesim                 =3    # Number of tubes simulated per panel

    model.analysis_mode         ='selected_days' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
    model.analysis_days         =[172] # can manually set this but analysis_mode will default set this as well
    substeps                    =1                      # number of model substeps per hour
    model.delta_hour            =1/substeps                # Time spacing between simulated time points (hr)
    model.is_half_day           =False           # Simulate only half of the day?
    model.dni_setting           ='user-defined-noon'  
    model.user_defined_dni      ={172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
    model.ambient_setting       ='user-defined'
    model.user_defined_Tamb     =25+273.15    # Constant user-defined ambient T (K)
    model.user_defined_RH       =25             # Constant user-defined relative humidity
    model.user_defined_vwind10  =0         # Constant user-defined wind speed (m/s at 10m height)

    ## configure to run from a saved flux profile
    model.check_for_existing_flux_distributions =False  # Check if flux distributions for this configuration have already been saved
    model.save_flux_distributions               =False
    model.SF.saved_flux_direc                   ='./flux_sims/'  # Directory to store flux distributions

    # directly set the aiming file
    # model.receiver.aiming_file          ='aiming/demo_informed_spt_method/ideal_fluxgrid_220MWth_0.83AR_270area_raw.csv'
    # model.receiver.aiming_file          ='aiming/demo_informed_spt_method/ideal_fluxgrid_220MWth_0.83AR_270area_offset.csv'
    # model.receiver.aiming_file          ='aiming/demo_informed_spt_method/ideal_fluxgrid_220MWth_0.83AR_270area_maxLTE.csv'

    model.receiver.use_aiming_scheme    =0  # use these two lines in tandem if you want to enforce uniform aiming
    #model.receiver.aiming_file          ="C:\\Users\\deani\\OneDrive - UW-Madison\\heliocon\\outputs\\mma\\data\\flux_map_iter_1.csv"
    #model.receiver.aiming_file          ="C:\\Users\\deani\\OneDrive - UW-Madison\\heliocon\\outputs\\cloudy\\flux_t02.csv"
    #model.receiver.aiming_file = 'None'
    # #

    ###
    case_string =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results'

    ## get the seasonal DNI reference
    with open('seasonal_DNI',) as f:
        DNI_seasons=json.load(f)
    t_samples   =np.arange(0,24)+0.5

    ## include the tower pressure when caclulating pumping power
    model.receiver.include_tower_in_Qpump =True

    ## determine number of required panel replacements by solving the model at design point
    model.analysis_mode = 'design_point' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
    model.analysis_days = [172] # can manually set this but analysis_mode will default set this as well
    for i in range(40):
        model.receiver.aiming_file      =f"C:\\Users\\deani\\OneDrive - UW-Madison\\heliocon\\outputs\\mma\\data\\flux_map_iter_{i+1}.csv"
        model.solve(verbosity=1)

        ## extract thermal points for damage tool
        dTs_design, Tfs_design, qabs_design, Rs_design =helpers_thermal_model.get_thermal_results(model.receiver)

        print(f"max dT = {dTs_design.max():.1f} C at {np.unravel_index(np.argmax(dTs_design), dTs_design.shape)}")
        print(f"max Tf = {Tfs_design.max():.1f} C at {np.unravel_index(np.argmax(Tfs_design), Tfs_design.shape)}")
        print(f"min dT = {dTs_design.min():.1f} C at {np.unravel_index(np.argmin(dTs_design), dTs_design.shape)}")
        print(f"min Tf = {Tfs_design.min():.1f} C at {np.unravel_index(np.argmin(Tfs_design), Tfs_design.shape)}")

        bad = (dTs_design > 300) | (Tfs_design > 575)
        print(f"out-of-range nodes: {bad.sum()} / {bad.size}")
        print("first failing locations [axial node, panel, simulated tube]:")
        print(np.argwhere(bad)[:10])

        ## calculate lifetimes
        LTEs                            =dmg_obj.get_LTEs(dTs_design.flatten(), Tfs_design.flatten(), Rs_design.flatten())
        min_panel_LTEs, min_tube_LTEs   =dmg_obj.calc_minimum_panel_LTEs(model.receiver, LTEs)
        print('minimum panel LTE is:',min_panel_LTEs.min())

        # # ## show lifetime profiles and op points if desired
        dmg_obj.plot_dmg_map(include_ratios=False, op_dTs=dTs_design.flatten(), op_Tfs=Tfs_design.flatten(), savename=f'outputs/mma/LTEs/contours/LTE_contour_{i+1:02d}.png' )
        plot_results(model.receiver.D, model.receiver.H, model.receiver.Npanels, model.receiver.ntubesim, model.receiver.disc.nz, LTEs, 'lifetimes (yrs)', savename=f'outputs/mma/LTEs/maps/LTE_heatmap_{i+1:02d}.png', vmin=0, vmax=100) # use for A617. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'


        ## calculate the number of replacements
        panel_replacements =np.floor(N_life/min_panel_LTEs)
        N_repl             =np.sum(panel_replacements)

        ## 
        print(f'number of required panel replacements over a 30 year lifetime:{N_repl}')
