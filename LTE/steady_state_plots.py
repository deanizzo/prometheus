from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  
import seaborn as sns
import pandas as pd
plt.rcParams['font.size'] = 7.75
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5



#==========================================================================
def _setup_subplots(nsub = None, nrow = None, ncol = None, wsub = 2.0, hsub = 2.0, wspace = 0.2, hspace = 0.2, left = 0.45, right = 0.15, bot = 0.45, top = 0.45, wr = None):   
    if wr is not None and len(wr) != ncol:
        wr = None
    
    if nsub is None:
        nsub = nrow*ncol
    else:
        nrow = int(floor(nsub**0.5))
        ncol = int(ceil(nsub / float(nrow)))    
        
    width = wsub * ncol + (ncol-1)*wspace + left + right
    height = hsub * nrow + (nrow-1)*hspace + top + bot
    fig = plt.figure(figsize = (width, height))     
    gs = gridspec.GridSpec(nrow, ncol, bottom = bot/height,  top = 1.0-(top/height), left = left/width, right = 1.0-(right/width), wspace = wspace/wsub , hspace = hspace/hsub, width_ratios = wr)    
    ax = []
    for row in range(nrow):
        for col in range(ncol):
            i = row*ncol + col
            if i < nsub:
                newax = fig.add_subplot(gs[row,col])
                ax.append(newax)    
    return [fig, ax, nrow, ncol]
     

def _set_common_y_limits(axs):
    ylims = [min([ax.get_ylim()[0] for ax in axs]), max([ax.get_ylim()[1] for ax in axs])]
    for ax in axs:
        ax.set_ylim(ylims)
    return


#==========================================================================
# Plot profiles of temperature, flux, and stress against position along each flow path of the receiver
def plot_profiles_in_path(model, day, hour_offset, use_all_tubes = True, include_allow_stress = False, savename = None):
    tpt = np.intersect1d(np.where(model.results.day==day)[0], np.where(model.results.time_offset==hour_offset)[0])[0]
    nz, npan, ntube, ntime = model.results.Tf.shape
    plottubes = np.arange(ntube) if use_all_tubes else [int((ntube-1)/2)]  # Tube indicies to include in the plot
    
    npath = model.receiver.npaths
    nrow = include_allow_stress + 1  # Row 1 plots wall T, fluid T, absorbed flux, von Mises stress; Row 2 plots wall T and stress/allowable stress
    [fig, ax, nrow, ncol] = _setup_subplots(nrow = nrow, ncol = npath, wsub = 2.4, hsub = 1.8, wspace = 1.25, hspace = 0.5, left = 0.5, right = 0.7, bot = 0.4, top = 0.4) 
    
    axtwins = []
    for p in range(npath):
        panels = model.receiver.flow_paths[p]
        n = len(panels)
        zpts = [j*model.receiver.H + i*model.receiver.H/(nz-1) for j in range(n) for i in range(nz)]
        data = {k:np.zeros((n*nz, ntube)) for k in ['Tw', 'Tf', 'abs_flux', 'stress', 'allow_thermal_stress_fraction_prof']}
        for j in range(len(panels)):
            for k in data.keys():
                data[k][j*nz:(j+1)*nz,:] = getattr(model.results, k)[:,panels[j],:,tpt]

        ax1 = ax[p]        
        ax2 = ax[p].twinx()
        axtwins.append(ax2)
        for t in plottubes:
            alpha = 1 if t == (ntube-1)/2  else 0.4
            islabel = (t == (ntube-1)/2)
            ax1.plot(zpts, data['Tf'][:,t]-273.15, ls = '-', lw = 0.75, color = 'midnightblue', alpha = alpha, label = 'Fluid T' if islabel else '') 
            ax1.plot(zpts, data['Tw'][:,t]-273.15, ls = '-', lw = 0.75, color = 'darkgreen', alpha = alpha, label = 'Wall T' if islabel else '')
            ax2.plot(zpts, data['abs_flux'][:,t], ls = '-', lw = 0.75, color = 'maroon', alpha = alpha, label = 'Absorbed flux' if islabel else '') 
            ax2.plot(zpts, data['stress'][:,t], ls = '-', lw = 0.75, color = 'gray', alpha = alpha, label = 'von Mises stress' if islabel else '')                               
        ax1.set_xlabel('Flow path length (m)')
        ax1.set_ylabel('Wall or fluid temperature ($^{\circ}$C)')
        ax2.set_ylabel('Absorbed flux (kW/m$^2$) or\n von Mises elastic stress (MPa)')  
        ax1.legend(loc='upper center', bbox_to_anchor=(0.18, 1.245), ncol=1)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.73, 1.245), ncol=1)
        ax1.annotate('Flow path %d'%p, xycoords = 'axes fraction', xy = (0.5, 0.98), ha = 'center', va = 'top', annotation_clip = False)
        
        
        if include_allow_stress:   
            for t in plottubes:
                alpha = 1 if t == (ntube-1)/2 else 0.4
                ax[npath+p].plot(zpts, data['allow_thermal_stress_fraction_prof'][:,t], ls = '-', lw = 0.75, color = 'k', alpha = alpha)                 
            ax[npath+p].set_xlabel('Flow path length (m)')
            ax[npath+p].set_ylabel('Fraction of allowable stress')   
            ax[npath+p].annotate('Flow path %d'%p, xycoords = 'axes fraction', xy = (0.5, 0.98), ha = 'center', va = 'top', annotation_clip = False)
    
    _set_common_y_limits([ax[j] for j in range(npath)])
    _set_common_y_limits(axtwins)
    if include_allow_stress:
        _set_common_y_limits([ax[npath+j] for j in range(npath)])
    
    if savename is not None:             
        plt.savefig(savename, dpi = 500)  

    return    

     





#==========================================================================
# Plot solution quantities per panel (max over simulated tubes) for all time points in a given day
def plot_panels_vs_time(model, day, combine_tubes = 'max', savename = None):  
    use_profiles = ['Tw', 'Tf', 'abs_flux', 'stress']
    labels = ['Twall ($^{\circ}$C)', 'fluid T ($^{\circ}$C)', 'absorbed flux (kW/m$^2$)', 'total stress (MPa)']
    inds = np.where(np.logical_and(model.results.day == day, model.results.Qfluid > 0))[0]  # Time points on this day where solution was completed
    hours = model.results.time_offset[inds]             
    nsub = len(use_profiles)
    npan = model.receiver.Npanels
    colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue', 'C1', 'C2']
    [fig, ax, nrow, ncol] = _setup_subplots(nsub = 4, wsub = 1.385, hsub = 1.385, wspace = 0.5, hspace = 0.5, left = 0.6, right = 1.0, bot = 0.4, top = 0.05) 
    # ax.set_prop_cycle(color=colors)
    for j in range(nsub):
        data_per_tube = getattr(model.results, use_profiles[j]).max(0)   # Max over axial position for each (panel, tube, time point)
        data = data_per_tube.max(1) if combine_tubes == 'max' else data_per_tube.mean(1)  # Maximum or average over simulated tubes per panel
        if use_profiles[j] in ['Tw', 'Tf']:
            data -= 273.15
        for p in range(npan):
            color = colors[p] if p < npan/2 else colors[npan-1-p] # commented out because doesn't work for large number of panels
            alpha = 1.0 if p < npan/2 else 0.4
            ax[j].plot(hours, data[p,inds], '-o', linewidth = 0.75, markersize = 2.0, color = color, alpha = alpha, label = 'Panel ' + str(p)) # commented out because doesn't work for large number of panels
            ax[j].plot(hours, data[p,inds], '-o', linewidth = 0.75, markersize = 2.0, alpha = alpha, label = 'Panel ' + str(p))    
        ax[j].set_ylabel(labels[j])
        ax[j].set_xlabel('Hour (relative to solar noon)')
    ax[ncol-1].legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
 
    if savename is not None:             
        plt.savefig(savename, dpi = 500)  
    return
    
#==========================================================================
# Plot solution quantities per panel (max over simulated tubes) for all time points in a given day
def plot_panels_vs_time_jwenn(model, day, combine_tubes = 'max', savename = None):  
    use_profiles = ['Tw', 'Tf', 'abs_flux', 'stress']
    labels = ['Twall ($^{\circ}$C)', 'fluid T ($^{\circ}$C)', 'absorbed flux (kW/m$^2$)', 'total stress (MPa)']
    inds = np.where(np.logical_and(model.results.day == day, model.results.Qfluid > 0))[0]  # Time points on this day where solution was completed
    hours = model.results.time_offset[inds]             
    nsub = len(use_profiles)
    npan = model.receiver.Npanels
    colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue', 'C1', 'C2']
    [fig, ax, nrow, ncol] = _setup_subplots(nsub = 4, wsub = 1.385, hsub = 1.385, wspace = 0.5, hspace = 0.5, left = 0.6, right = 1.0, bot = 0.4, top = 0.05) 
    # ax.set_prop_cycle(color=colors)
    for j in range(nsub):
        data_per_tube = getattr(model.results, use_profiles[j]).max(0)   # Max over axial position for each (panel, tube, time point)
        data = data_per_tube.max(1) if combine_tubes == 'max' else data_per_tube.mean(1)  # Maximum or average over simulated tubes per panel
        if use_profiles[j] in ['Tw', 'Tf']:
            data -= 273.15
        for p in range(npan):
            index_color= p % len(colors)
            color = colors[index_color]
            # color = colors[p] if p < npan/2 else colors[npan-1-p] # commented out because doesn't work for large number of panels
            alpha = 1.0 if p < npan/2 else 0.4
            ax[j].plot(hours, data[p,inds], '-o', linewidth = 0.75, markersize = 2.0, color = color, alpha = alpha, label = 'Panel ' + str(p)) # commented out because doesn't work for large number of panels
            ax[j].plot(hours, data[p,inds], '-o', linewidth = 0.75, markersize = 2.0, alpha = alpha, label = 'Panel ' + str(p))    
        ax[j].set_ylabel(labels[j])
        ax[j].set_xlabel('Hour (relative to solar noon)')
    ax[ncol-1].legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
 
    if savename is not None:             
        plt.savefig(savename, dpi = 500)  
    return
    


#==========================================================================
# 2D heat map of solution quantities (x-axis = position around circumference, y-axis = receiver height)
def plot_receiver_heatmap(model, key, day, hour_offset = 'all', plot_all_tubes = True, title = None, ylims = None, savename = None):    
    inds = np.where(np.logical_and(model.results.day == day, model.results.Qfluid > 0))[0]  # Time points on this day where solution was completed
    if hour_offset != 'all':
        inds = np.intersect1d(inds, np.where(model.results.time_offset==hour_offset)[0])


    offset = -273.15 if key in ['Tf', 'Tw'] else 0.0
    soln = getattr(model.results, key)[...,inds] + offset  # Axial position (from tube inlet), panel, tube per panel, last time point
    nz,npan,ntube,npt = soln.shape
    if not plot_all_tubes:  # Plot only center most tube
        ntube = 1
        t = int((ntube-1)/2)
        soln = soln[:,:,t,:].reshape((nz,npan,ntube,npt))
    
    vmin = soln.min() if ylims is None else ylims[0]
    vmax = soln.max() if ylims is None else ylims[1]

    xpts = np.linspace(-180, 180.0, npan*ntube)
    zpts = np.flip(np.linspace(0.0, 1.0, nz), 0)
    def _get_data_labels(nlabels, pts, form = '%.1f'):
        lab = []
        n = len(pts)
        xint = int(floor(n / float(nlabels-1)))
        for i in range(n):
            if n <= nlabels or (i%xint) == 0:
                lab.append((form % pts[i]))
            else:
                lab.append('')
        return lab
    xlab = _get_data_labels(6, xpts, form = '%.0f')
    zlab = _get_data_labels(5, zpts) 


    [fig, ax, nrow, ncol] = _setup_subplots(npt, wsub = 1.8, hsub = 1.5, wspace = 0.45, hspace = 0.65, left = 0.4, right = 0.15, bot = 0.6, top = 0.4) #wspace=0.45
    for j in range(len(inds)):  # Subplots for each time point
        label = ('Day: %d, Time: %d h'% (day, model.results.time_offset[inds[j]]))
        
        data = soln[:,:,:,j]
        for p in range(npan):
            if model.receiver.tubes[p][0].flow_against_gravity:  # Tube inlet to outlet is bottom to top, flip axial points so that top of receiver is at the top of the array
                data[:,p,:] = data[::-1,p,:]   
        data = soln[:,:,:,j].reshape((nz, npan*ntube)) # Combine tubes and panels: left to right in the array is south - east - north - west - south ()
        data = np.flip(data, 1)  # Flip columns so that receiver panel on the west-side of the receiver are on the left-hand side of the array

        sns.heatmap(data, ax = ax[j], vmin = vmin, vmax = vmax, xticklabels = xlab, yticklabels = zlab)        
        ax[j].set_xlabel('Circumferential position\n(0 = north, + = east)')
        ax[j].set_ylabel('Height')     
        ax[j].annotate(label, xycoords = 'axes fraction', xy = (0.5, 1.025), ha = 'center', annotation_clip = False)

    if title is not None:
        if npt == 1:
            ax[0].annotate(title, xycoords = 'axes fraction', xy = (0.5, 1.025), ha = 'center', annotation_clip = False)
        else:
            plt.suptitle(title)
                
    if savename is not None:             
        plt.savefig(savename+'.png', dpi = 500)  

    return


#==========================================================================
# Plot wall temperature and stress profiles in tube
def plot_tube_wall_vs_theta(rec, panels, axial_loc = 0.5, use_all_tubes = False, include_inner_wall = True, include_stress = True, savename = None):
    colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue']
    ntube = len(rec.tubes[0])
    plottubes = np.arange(ntube) if use_all_tubes else [int((ntube-1)/2)]  # Tube indicies to include in the plot
    k = int(axial_loc * rec.disc.nz)  # Axial point to use in plots
    thetapts = rec.disc.thetapts * 180/np.pi
    ncol = include_stress + 1 # First column = wall temperatures, second colum = stress
    [fig, ax, nrow, ncol] = _setup_subplots(nrow = 1, ncol = ncol, wsub = 2.2, hsub = 1.8, wspace = 0.5, hspace = 0.5, left = 0.5, right = 0.05, bot = 0.4, top = 0.05) 
    for c in range(ncol):
        name = 'Tw' if c == 0 else 'stress_equiv'  # Plot wall T in first column total equivalent stress (thermal + pressure) in second column
        for i in range(len(panels)):
            p = panels[i]
            for t in plottubes:
                alpha = 1 if t == (ntube-1)/2 else 0.4
                islabel = (t == (ntube-1)/2)
                data = getattr(rec.tubes[p][t], name)-273.15 if name == 'Tw' else getattr(rec.tubes[p][t], name)
                ax[c].plot(thetapts, data[k,:,-1], lw = 1.0, ls = '-', color = colors[i], alpha = alpha, label = 'Outer wall, panel %d'%p if islabel else '')
                if include_inner_wall:
                    ax[c].plot(thetapts, data[k,:,0], lw = 1.0, ls = '--', color = colors[i], alpha = alpha, label = 'Inner wall, panel %d'%p if islabel else '')
        ax[c].set_xlabel('Tube circumferential position ($^{\circ}$)')
        ax[c].set_ylabel('Wall temperature ($^{\circ}$C)' if name == 'Tw' else 'Equivalent stress (MPa)')
    #ax[ncol-1].legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
    ax[ncol-1].legend()
    # ax[ncol-2].set_ylim(300,535)
    if savename is not None:             
        plt.savefig(savename+'.png', dpi = 500)  
    return
#=====================================================================
# post processing functions #
#=====================================================================
def get_maximum_dTs(model, plot=True,save_img=True,save_csv=True,imgName='max_dTs_defaultSave',csvName='max_dTs_defaultSave' ):
    """
    -analyzes thermal results to obtain each tube's maximum temperature difference and the corresponding Tf
    -note: dTs vary with position and time -- this function either looks at one timepoint or all, depending on input parameters
    -function will append to a dataframe for csv saving if option selected
    """
    Tw_outer_K=getattr(model.results, 'Tw')
    Tw_outer=Tw_outer_K-273.15    #convert to [C]

    Tf_K=getattr(model.results, 'Tf')
    Tf=Tf_K-273.15  #convert to [C]

    Tw_inner_high_K=getattr(model.results, 'Tw_inner_high')
    Tw_inner_high=Tw_inner_high_K-273.15   #convert to [C]

    
    Tw_inner_low_K=getattr(model.results, 'Tw_inner_low')
    Tw_inner_low=Tw_inner_low_K-273.15    #convert to [C]


    allTubeTempInfo=pd.DataFrame(data=None, index=None, columns=["Tf","dT"], dtype=None, copy=None)  #just need the strainRange2 to be formatted in a way I can add to master DF
    npts=Tw_outer.shape[3]
    dTs=np.zeros((model.receiver.Npanels, model.receiver.ntubesim, model.receiver.disc.nz) )
    
    for panel in range(model.receiver.Npanels):
        for tube in range(model.receiver.ntubesim):
            for time in range(npts):
                single_Tw_outer=Tw_outer[:,panel,tube,time]
                single_Tw_inner_low=Tw_inner_low[:,panel,tube,time]
                single_Tw_inner_peak=Tw_inner_high[:,panel,tube,time]
                dT=single_Tw_outer-single_Tw_inner_low
                dTs[panel,tube,:]=np.maximum(dT,dTs[panel,tube,:])  #this overwrites previous timepoints' max if a new one is found
            dTmax=np.max(dTs[panel,tube,:]) #takes the maximum along all axial positions for all timepoints
            locDTmax=np.argmax(dTs[panel,tube,:]) # get the index of maximum dT to find corresponding Tf
            Tf_single=np.average(Tf[locDTmax,panel,tube,:]) #take the average Tf overall timepoints at whatever axial location it is...
            tubeTempInfoDF=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the strainRange2 to be formatted in a way I can add to master DF
            tubeTempInfoDF['Tf']=np.array([Tf_single])
            tubeTempInfoDF['dT']=np.array([dTmax])
            allTubeTempInfo=pd.concat([allTubeTempInfo,tubeTempInfoDF],ignore_index=True)
    if plot:
        [fig,ax] = plt.subplots()
        ax.scatter(allTubeTempInfo['dT'],allTubeTempInfo['Tf'])
        ax.set_ylabel('$T_f$ (C)')
        ax.set_xlabel('$\Delta T$ (C)')
        # ax.set_ylim(bottom=295)
        if save_img:
            imgName = "./imgs/"+ imgName +'.png'
            plt.savefig(imgName, dpi = 500) 
        plt.show()   
    if save_csv:
        fname = './' + csvName + '.csv'
        allTubeTempInfo.to_csv( (fname) )
    return