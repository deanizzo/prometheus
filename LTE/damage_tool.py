"""
damage tool created by jwenner on 7/10/2025 

goal is to make SRLIFE-generated FEA results accessible and useful for predicting solar receiver tube creep-fatigue damage and lifetime
"""
import numpy as np
import math
import pandas as pd
import scipy.interpolate 
from scipy.spatial import Delaunay
import timeit
import sys
import material_limits
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from pathlib import Path

class damageTool():
    def __init__(self, mat_string, interp_mode='LNDI', choose_ratios=False):
        
        self.mat            =mat_string # material type. Can be any of the materials in SRLIFE
        
        ## make sure mat_string input was correct
        if self.mat in ['316H', '740H', '800H', 'A230', 'A282', 'A617']:
            print(f'dmg model material is:{self.mat}')
        else: 
            print('material string not recognized')

        self.file_string        =None       # filestring location for dataframe loading
        self.interp_dT_limits   =None       # dataset limits for temperature difference
        self.interp_Tf_limits   =None       # dataset limits for fluid temperature

        ## either supply a simple set of ratios or a multi-part set depending on whether user desires ratio selection detail
        if choose_ratios==False:    # fastest method. Damage dataset is selected from subset of available ones
            simple_R_dict       ={'316H':0.43, '740H':0.75, '800H':0.46, 'A230':0.75, 'A282':0.75, 'A617':0.50}
            self.R              =simple_R_dict[self.mat]
            print(f'assuming constant ratio of dT_cond/dT_conv of:{self.R}')
        else:                       # slower but considered to be higher accuracy. predicts damage based on most applicable data set R
            print('no code for consideration of multiple resistance ratios')

        ## assign a contour cutoff based on recommended cutoffs
        self.rec_cutoffs        ={'316H':175, '740H':295, "800H":132, 'A230':270, 'A282':295, 'A617':295}
        self.cutoff =self.rec_cutoffs[self.mat]
            
        ## determine where this script is running and see if there's a database for this material
        py_loc =Path(__file__).resolve()    # this line gets the current path's file string
        potential_filestring    =py_loc.parent / f'dmg_tool_data/{self.mat}_dmgMap_data_R{self.R:.2f}.csv'
        if os.path.exists(potential_filestring):
            self.file_string = potential_filestring
            print('NOTE: Damage tables provide LTE in years')
        else:
            print('no database found')

        self.materialLimits = material_limits.materialLimits(self.mat)
        self.interpolator   =None
        self.df_LTE         =pd.DataFrame()
        self.ctf            =None
        self.interp_mode    =interp_mode    # sets the interpolation mode. Current options include: rbf (radial basis function) and LNDI (linear nd interpolator)

        
    def plot_dmg_map(self,include_ratios=False, op_dTs=np.array([]), op_Tfs=np.array([]), LTEs=np.array([]), savename=None ):
        ## load dataframe
        fig,ax      =plt.subplots(tight_layout=True)
        fontsize    =14
        if self.df_LTE.empty == True:
            self.load_lookup_table()

        ### get the maximum damage ratio for all parametric cases
        dratsDF_local   =self.df_LTE.pivot(index="Tf",columns='dT',values='ratio_dmgs') #re-organize data for countour plots
        drats_local     =pd.DataFrame.to_numpy(dratsDF_local)

        ### make pivot table for ratio
        if include_ratios:
            self.df_LTE["drat_log"] =np.log10(self.df_LTE["drat"])
            df_HeatMap              =self.df_LTE.pivot(index="Tf",columns='dT',values='drat_log') #re-organize data for countour plots
            df_HeatMap              =df_HeatMap.sort_index(ascending=False) #flip it
            drats                   =pd.DataFrame.to_numpy(df_HeatMap)
            if self.mat in ['A617', '740H', '800H']:
                levels =[-6,-5,-4,-3,-2,-1] # this is for A617 and 740H and 800H
            elif self.mat == '316H':
                levels =[-3,-2,-1,0,1,2] # this is for 316H
            else:
                levels =[-6,-5,-4,-3,-2,-1,0,1,2] #can be used in ctf line

            DesRegionY = df_HeatMap.index.to_numpy()
            DesRegionX = df_HeatMap.columns.to_numpy()
            ctf=ax.contourf(DesRegionX, DesRegionY, drats,cmap=cm.YlGn,levels=levels) # removed levels=levels
            cb1=fig.colorbar(ctf)
            cb1.set_label(label='log($d_f$/$d_c$)',fontsize=fontsize)
            cb1.ax.tick_params(labelsize=fontsize)

        # ### add a set of material limits
        x_LIM_points, y_LIM_points = self.materialLimits.get_plot_pts(R=self.R)
        # ax.plot(x_LIM_points,y_LIM_points,color='slateblue')
        y_line_pts=np.ones(x_LIM_points.size)*600 # just need an upper bound to fill between
        ax.fill_between(x_LIM_points,y_LIM_points,y_line_pts,color='rosybrown',zorder=2)

        ## plot SR zone
        self.materialLimits.make_SR_interpolator()
        if self.materialLimits.df_SR.empty == False:
            SRzone      =self.materialLimits.df_SR
            SRzone      =SRzone.sort_index(ascending=False) #flip it
            SRzoneVals  =pd.DataFrame.to_numpy(SRzone)
            SRzoneVals  =SRzone.iloc[:,1:].values
            SRlevels    =(2.00,40.00)
            DesRegionY  =SRzone['Tf'].values
            DesRegionX  =SRzone.columns[1:].to_numpy().astype(float)
            ctfSR=ax.contourf(DesRegionX, DesRegionY, SRzoneVals,levels=SRlevels,cmap=cm.inferno,zorder=2) # removed levels=levels


        ### make another pivot table for total damage
        if (self.mat == '316H') or (self.mat == '800H'): 
            ax.set_xlim(100,180)
            ax.set_ylim(bottom=300,top=500)
        elif self.mat=='A230':
            ax.set_xlim(50,300)
            ax.set_ylim(bottom=275,top=575)
        elif (self.mat=='A282') or (self.mat=='740H'):
            ax.set_xlim(50,300)
            ax.set_ylim(bottom=275,top=575)
        else:
            ax.set_xlim(100,220)
            ax.set_ylim(bottom=300,top=550)

        df_HeatMap2=self.df_LTE.pivot(index="Tf",columns="dT",values="LTE")
        df_HeatMap2=df_HeatMap2.sort_index(ascending=False) #flip it
        lifetimes=pd.DataFrame.to_numpy(df_HeatMap2) # make a countourf plot using matplotlib
        levels=(5,10,15,20,30,40,50,60,70,80)#range(0,80,10) #can be used in ctf line
        DesRegionY2 = df_HeatMap2.index.to_numpy()
        DesRegionX2 = df_HeatMap2.columns.to_numpy()
        ctf=ax.contour(DesRegionX2, DesRegionY2, lifetimes,levels=levels,cmap=cm.copper_r,zorder=1) #was using cm.cool or colors='black', levels=levels
        # cb2=plt.colorbar()


        ### manually place contour
        ax.axes.clabel(ctf,inline=True,inline_spacing=1, fontsize=6, fmt=self.fmt,manual=False,levels=levels) #left click: place. center click: done

        ### add tube temperatures
        if op_dTs.any() == True:
            ax.scatter(op_dTs,op_Tfs,marker='^',s=15,color='r',zorder=3)
            if LTEs.any() == True:
                for i in range(LTEs.size):
                    ax.annotate(f'{LTEs[i]:.2f}', (op_dTs[i], op_Tfs[i]), textcoords="offset points", xytext=(0, 5), ha='left', va='bottom', fontsize=10)

        ### final touches on plot
        ax.grid(visible=False)
        ax.set_facecolor('white')
        ax.spines['left'].set_color('black')  #use this line if plotting only one at a time
        ax.spines['bottom'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.tick_params(bottom=True, top=False, left=True, right=False, labelsize=fontsize)
        ax.set_xlabel('temperature difference (C)',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        ax.set_ylabel('fluid temperature (C)',fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savename != None:
            plt.savefig(fname=f'{savename}',dpi=300)
        #plt.show()
        plt.close()

        return

    def make_interpolator(self):
        """
        oversight function
        """
        self.load_lookup_table()
        if self.interp_mode == 'rbf':
            self.generate_RBF_interpolator()
        elif self.interp_mode == 'LNDI':
            self.generate_LNDI_interpolator()
        
        return

    def load_lookup_table(self):

        self.df_LTE = pd.read_csv(self.file_string)

        ## dataset range limits
        self.interp_Tf_limits   =np.array([self.df_LTE['Tf'].values.min(),self.df_LTE['Tf'].values.max()])
        self.interp_dT_limits   =np.array([self.df_LTE['dT'].values.min(),self.df_LTE['dT'].values.max()])

        return 

    # def grid_interp_LTEs(self,xi):
    #     """
    #     NOTE: untested
    #     """
    #     finite_mask = np.isfinite(self.df_LTE['LTE'].values) # infinite lifetime values do not effect interpolation
    #     # make a subselection for lookup table
    #     dTs = self.df_LTE['dT'].values[finite_mask]
    #     Tfs = self.df_LTE['Tf'].values[finite_mask]
    #     LTEs= self.df_LTE['LTE'].values[finite_mask]

    #     interp_grid = scipy.interpolate.griddata(np.dstack( (dTs,Tfs) ).squeeze(), LTEs, xi, method='linear')

    def generate_RBF_interpolator(self):
        """
        creates a radial basis function interpolator based on finite lifetime data
        -tested on 7/28/25 and returned accurate values. Note that transitiion to infinite LTE as seen on maps is not captured because of nature of RBF...
            ... this means that the data may predict infnite lifetime but this interpolator will still provide a finite (albeit extremely high) number
        """
        if self.df_LTE.empty == True:
            self.load_lookup_table()

        finite_mask = np.isfinite(self.df_LTE['LTE'].values) # infinite lifetime values do not effect interpolation
        # make a subselection for lookup table
        dTs = self.df_LTE['dT'].values[finite_mask]
        Tfs = self.df_LTE['Tf'].values[finite_mask]
        LTEs= self.df_LTE['LTE'].values[finite_mask]

        interp_f = scipy.interpolate.RBFInterpolator(np.dstack(  (dTs,Tfs)).squeeze(), LTEs, neighbors=100, kernel='linear')
        
        self.interpolator = interp_f # returns time in years

    def generate_LNDI_interpolator(self):
        """
        creates a linear non-dimensional interpolator based on finite lifetime data
        
        """
        if self.df_LTE.empty == True:
            self.load_lookup_table()

        finite_mask = np.isfinite(self.df_LTE['LTE'].values) # infinite lifetime values do not effect interpolation
        # make a subselection for lookup table
        dTs = self.df_LTE['dT'].values[finite_mask]
        Tfs = self.df_LTE['Tf'].values[finite_mask]
        LTEs= self.df_LTE['LTE'].values[finite_mask]

        # get the triangulation done
        tri =Delaunay(np.dstack((dTs,Tfs)).squeeze() )

        # run the interpolator once to get initialization out of the way
        interp_f = scipy.interpolate.LinearNDInterpolator(tri, LTEs, fill_value=100000)
        
        self.interpolator = interp_f # returns time in years

    def make_contour_function_from_interpolator(self,LTE_desired=30, cutoff=None, show_LTE_ctr=False):
        """
        creates contour levels using the interpolator - this ensures that the desired and predicted LTE agree
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        # create and plot a damage map using LTEs from the interpolator
        map_fig,map_ax   =self.make_dmg_map_from_interpolator(LTE_desired)
        # select a desired contour, get vertices, show on plot
        if (LTE_desired < 30) or (self.mat=='316H'): # my method isn't made for problematic contours, need to limit user
            raise ValueError('Design lower than 30 yrs and 316H contours not recommended. Many contours are unstable')
        elif self.mat in ['740H','A282']:
            print('NOTE: design contours 30-100 years have known conflicts with crit. corrosion temps.')
        # overwrite cutoff if present
        if cutoff is not None:
            self.cutoff=cutoff
        # get the desired LTE index from the level list
        LTE_index =self.level_list.index(LTE_desired)
        # grab the contour's vertices
        all_dT_verts    =self.ctf.get_paths()[LTE_index].vertices[:,0] # if you see a "has no attribute 'get_paths' " error then check your python version
        all_Tf_verts    =self.ctf.get_paths()[LTE_index].vertices[:,1]
        # # make subselection, assuming you want the left most contour
        # Tf_data_max =self.df_LTE['Tf'].max()
        # slice_indices     =np.where(all_Tf_verts == Tf_data_max)[0]
        # if slice_indices.size > 1:
        #     dTs_LTE             =all_dT_verts[0:slice_indices[1]]
        #     Tfs_LTE             =all_Tf_verts[0:slice_indices[1]]
        # else: 
        #     dTs_LTE             =all_dT_verts[0:slice_indices[0]]
        #     Tfs_LTE             =all_Tf_verts[0:slice_indices[0]]
        # dTs_LTE             =np.where(dTs_LTE < cutoff, dTs_LTE,cutoff) # prevent dTs from venturing into stress reset zone
        
        ## make subselection assuming you want the left most
        Tf_data_max =self.df_LTE['Tf'].max()
        slice_indices     =np.where(all_Tf_verts == Tf_data_max)[0]
        if slice_indices.size > 1:  # ase that the contour traverses the top of the range multiple times
            dTs_LTE             =all_dT_verts[0:slice_indices[1]]
            Tfs_LTE             =all_Tf_verts[0:slice_indices[1]]
        elif slice_indices.size == 1:  # case that the contour only traverses the top of the range once
            dTs_LTE             =all_dT_verts[0:slice_indices[0]]
            Tfs_LTE             =all_Tf_verts[0:slice_indices[0]]
        else:   # case that the contour never traverses the top of the Tf range
            dTs_LTE =all_dT_verts
            Tfs_LTE =all_Tf_verts
        

        
        dTs_LTE             =np.where(dTs_LTE < self.cutoff, dTs_LTE,self.cutoff) # prevent dTs from venturing into stress reset zone

        dT_function =scipy.interpolate.interp1d(Tfs_LTE, dTs_LTE, fill_value='extrapolate')
        self.dT_function = dT_function
        print('dT function created')

        # plot the contour function
        ctr_Tf_samples  =np.linspace(self.df_LTE['Tf'].values.min(), self.df_LTE['Tf'].values.max())
        ctr_dT_samples  =self.dT_function(ctr_Tf_samples)
        map_ax.plot(dTs_LTE, Tfs_LTE, label='selected contour data',color='gray',linestyle='--')
        map_ax.plot(ctr_dT_samples, ctr_Tf_samples, label='contour function',color='b')
        map_ax.legend()

        if show_LTE_ctr:
            plt.show()
        plt.close()

        return 
    
    @staticmethod
    def fmt(x):
        # for formatting the damage map
        s = f"{x:.0f}" + ' (yrs)'
        return s

    def make_dmg_map_from_interpolator(self, user_level):
        """
        makes a damage map for contour access and visualization. Note: based on RBF interpolator, so slightly different values than previously done
        ---
        user_level - desired lifetime contour in years
        """
        print('making damage map and returning contours')
        # initialize plot
        fig, ax=plt.subplots()

        # make a meshgrid of x,y values for interpolation use
        Tf_data_min =self.df_LTE['Tf'].min()
        Tf_data_max =self.df_LTE['Tf'].max()
        dT_data_min =self.df_LTE['dT'].min()
        dT_data_max =self.df_LTE['dT'].max()
        Tf_range  =np.linspace(Tf_data_min,Tf_data_max,250)
        dT_range  =np.linspace(dT_data_min, dT_data_max, 250)
        dT_samples, Tf_samples  =np.meshgrid(dT_range, Tf_range)
        # get the LTE values based on the RBF interpolator
        LTE_samples =self.get_LTEs_no_limits(dT_samples.flatten(), Tf_samples.flatten(), 0.5*np.ones(Tf_samples.size))
        # make the contour plot
        levels=[10.1, 30.1, 100.1]
        # levels=[5,10,15,20,25,30,40,50,60,70,80]
        levels.append(user_level)
        levels.sort()
        ctf=ax.contour(dT_range, Tf_range, LTE_samples.reshape(dT_samples.shape), levels=levels,cmap=cm.copper_r,zorder=1) #was using cm.cool or colors='black', levels=levels
        self.ctf=ctf
        self.level_list=levels
        ax.axes.clabel(ctf,inline=True,inline_spacing=1, fontsize=6, fmt=self.fmt,manual=False,levels=levels) #left click: place. center click: done
        fontsize=14
        ax.set_ylabel('fluid temperature (C)', fontsize=fontsize)
        ax.set_xlabel('total temperature difference (C)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
      
        return fig,ax
    
    def get_LTEs_no_limits(self,dTs,Tfs,Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in cycles or years, depending on the material's database. LTE only. Ignores corrosion and SR region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print(f'current interpolator is {self.interpolator}')

        ## check the limits
        self.check_data_limits(dTs, Tfs)

        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs = self.interpolator(interp_coords)

        return interp_LTEs

    def get_LTEs(self,dTs,Tfs,Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in years, depending on the material's database. Returns a 0 for a coordinate if that coordinate is in the corrosion or stress reset region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print(f'interpolator is:{self.interpolator}')

        ## check the limits
        self.check_data_limits(dTs, Tfs)

        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs = self.interpolator(interp_coords)

        ## change any values that violate the film temperature limits
        # get all the crown temperatures
        Tcrowns = Tfs + dTs
        # get all the film temperatures
        Tfilms = self.materialLimits.estimate_Tfilm(Tcrowns,Tfs,Rs)
        # find the points that have T > Tcrit:
        toasty_array = self.materialLimits.check_T_pts(Tfilms)
        # assign the T > Tcrit locations lifetimes of 0 years
        interp_LTEs[toasty_array] = 0

        ## change any values that are in the stress reset region
        ## passes on 740H and A282 - these materials have no SR region in current data range
        # get an array of booleans
        if self.mat not in ['740H', 'A282']:
            resets = self.materialLimits.check_SR(dTs, Tfs)
            # set the coordinates that are resetting to have a LTE of 0
            interp_LTEs[resets] = 0

        return interp_LTEs
        
    def get_LTEs_w_penalties(self, dTs, Tfs, Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate and penalizes any stress reset cases
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in cycles or years, depending on the material's database. Returns a 0 for a coordinate if that coordinate is in the corrosion region, and a negative value for those in the SR region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        ## check the limits
        self.check_data_limits(dTs, Tfs)
        
        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs =self.interpolator(interp_coords)
        
        # initialize some lists
        interp_LTEs_pre_penalties         =np.array([interp_LTE for interp_LTE in interp_LTEs]) # Need to make independent copies of the list
        interp_LTEs_w_corrosion_zero      =np.array([interp_LTE for interp_LTE in interp_LTEs]) 
        interp_LTEs_w_SR_zero             =np.array([interp_LTE for interp_LTE in interp_LTEs]) 

        ## change any values that violate the film temperature limits and penalize them!
        # get all the crown temperatures
        Tcrowns = Tfs + dTs
        # get all the film temperatures
        Tfilms = self.materialLimits.estimate_Tfilm(Tcrowns,Tfs,Rs)
        # find the points that have T > Tcrit:
        toasty_array = self.materialLimits.check_T_pts(Tfilms)
        # assign the T > Tcrit locations lifetimes of 0 years
        interp_LTEs_w_corrosion_zero[toasty_array] = 0
        # now penalize those 0 LTE locations
        interp_LTEs_w_corrosion_penalty =self.materialLimits.penalize_corrosion(interp_LTEs_w_corrosion_zero, dTs, Tfs, Rs)
        corrosion_penalty               =interp_LTEs_w_corrosion_penalty-interp_LTEs_w_corrosion_zero

        ## change any values that are in the stress reset region. Penalize them!
        ## doesn't do this for 740H and A282 - no SR region in current dataset range
        if self.mat not in ['740H', 'A282']:
            # get an array of booleans
            resets =self.materialLimits.check_SR(dTs, Tfs)
            # set the coordinates that are resetting to have a LTE of 0
            interp_LTEs_w_SR_zero[resets] =0
            # for the coordinates that have LTE of zero, assign a penalty
            interp_LTEs_w_SR_penalty      =self.materialLimits.penalize_SR(interp_LTEs_w_SR_zero, dTs, Tfs)
            SR_penalty                    =interp_LTEs_w_SR_penalty-interp_LTEs_w_SR_zero
            
            ## need to add back in the corrosion penalty for points that got overwritten by SR checker
            interp_LTEs_post_penalties    =interp_LTEs_pre_penalties+SR_penalty+corrosion_penalty
        else: # only add corrosion penalty to 740H and A282
            interp_LTEs_post_penalties    =interp_LTEs_pre_penalties+corrosion_penalty

        return interp_LTEs_post_penalties
    
    def check_data_limits(self,dTs, Tfs):
        """
        make sure that all thermal operating points are within the dataset range. The extrapolation function
        is programmed to return large LTEs for low Tf and dTs. However, high Tf and dTs will also yield large
        LTEs. This isn't physically true so we want to alert the user.
        """
        if (dTs.max() > self.interp_dT_limits[1]) or (Tfs.max() > self.interp_Tf_limits[1]):
            print('WARNING: a Tf or dT exists that exceeds the range of the data set')
            #raise ValueError('WARNING: a Tf or dT exists that exceeds the range of the data set')
        
        return 

    def calc_minimum_panel_LTEs(self, receiver, LTEs):
        """
        outputs the LTEs per panel, based on the discretization of the receiver model
        ---
        receiver    - NREL thermal model object
        LTes        - single, flattened array with length of nz*ntubes_sim
        ---
        returns:
        min_panel_LTEs  - array with length of nPanels
        tube_min_LTEs   - array with length of ntubes_sim*Npanels
        """
        ## find out the number of tubes and axial discretization
        ntubes_sim  =receiver.ntubesim
        axial_nodes =receiver.tubes[0][0].disc.nz
        Npanels     =receiver.Npanels

        ## reshape, find the minimum for each tube
        LTEs_reshaped=LTEs.reshape(axial_nodes, Npanels, ntubes_sim)
        tube_min_LTEs=LTEs_reshaped.min(axis=0)

        ## find the minimum of each panel
        pmods           =np.mod(np.arange(Npanels*ntubes_sim),ntubes_sim)
        inds            =list(np.where(pmods == 0)[0])
        min_panel_LTEs  =np.array([])
        for i in range(Npanels):
            if i < Npanels-1:                
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:inds[i+1]])
            else:
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:])

            min_panel_LTEs  =np.concatenate((min_panel_LTEs,np.array([min_panel_LTE]) ))

        return min_panel_LTEs, tube_min_LTEs
    
    def calc_minimum_panel_LTEs_simple_inputs(self, ntubes_sim, axial_nodes, Npanels, LTEs):
        """
        outputs the LTEs per panel, based on the discretization of the receiver model. Same functionality as above, but doesn't require entire receiver object
        ---
        ntubes_sim  - how many tubes the model simulated per panel
        axial_nodes - (int)
        Npanels     - total number of panels on the receiver
        LTes        - single, flattened array with length of nz*ntubes_sim
        ---
        returns:
        min_panel_LTEs  - array with length of nPanels
        tube_min_LTEs   - array with length of ntubes_sim*Npanels
        """

        ## reshape, find the minimum for each tube
        LTEs_reshaped=LTEs.reshape(axial_nodes, Npanels, ntubes_sim)
        tube_min_LTEs=LTEs_reshaped.min(axis=0)

        ## find the minimum of each panel
        pmods           =np.mod(np.arange(Npanels*ntubes_sim),ntubes_sim)
        inds            =list(np.where(pmods == 0)[0])
        min_panel_LTEs  =np.array([])
        for i in range(Npanels):
            if i < Npanels-1:                
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:inds[i+1]])
            else:
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:])

            min_panel_LTEs  =np.concatenate((min_panel_LTEs,np.array([min_panel_LTE]) ))

        return min_panel_LTEs, tube_min_LTEs

if __name__ == "__main__":
    ## test basic functions for a starter material
    matl='A617'
    dmg_obj = damageTool(mat_string=matl)
    dmg_obj.load_lookup_table()
    xi = np.array([[102,300]])

    dmg_obj.generate_RBF_interpolator()
    t_interp_start = timeit.default_timer()
    LTE_xi = dmg_obj.interpolator(xi)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')
    print(f'interpolated LTE was:{LTE_xi} years')

    ### test ability to look up lifetimes for a second material
    t_interp_start = timeit.default_timer()
    matl2 ='A617'
    dmg_obj2 = damageTool(mat_string=matl2,interp_mode='LNDI')
    # LTEs_returned = dmg_obj2.get_LTEs(np.array([160,200]),np.array([550,500]),np.array([0.5,0.5]) ) # use this line for all the others
    LTEs_returned = dmg_obj2.get_LTEs(np.array([170,150]),np.array([450,475]),np.array([0.5,0.5]) ) # use this line for lower temp materials 
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    ### test ability to look up lifetimes, but rerun now that interpolator is made
    t_interp_start = timeit.default_timer()
    LTEs_returned = dmg_obj2.get_LTEs(np.array([170,170,170,180]),np.array([500,525,549,549]),np.array([0.5,0.5,0.5,0.5]) )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    ### test ability to look up lifetimes for 20 tubes
    t_interp_start = timeit.default_timer()
    npts = 100
    LTEs_returned = dmg_obj2.get_LTEs(np.linspace(100,220,npts),np.linspace(300,550,npts),np.ones(npts)*0.5 )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    ### test ability to look up lifetimes for 20 tubes for rbf method
    t_interp_start = timeit.default_timer()
    npts = 100
    LTEs_returned_rbf = dmg_obj.get_LTEs(np.linspace(100,220,npts),np.linspace(300,550,npts),np.ones(npts)*0.5 )
    print('the returned LTEs are:')
    print(LTEs_returned_rbf)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')
   

    ### test ability to look up lifetimes for 100 tubes but using new SR penalty function
    LTE_dummy = dmg_obj2.get_LTEs_w_penalties(np.array([100,120]),450*np.ones(2),np.ones(2)*0.5 ) # just something to initialize the interpolator so we get accurate measurement of time
    t_interp_start = timeit.default_timer()
    npts =100
    xs   =np.linspace(100,299,npts)
    LTEs_returned = dmg_obj2.get_LTEs_w_penalties(xs,450*np.ones(npts),np.ones(npts)*0.5 )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate w SR penalty was: {t_interp_end-t_interp_start}')

    fig, ax=plt.subplots(tight_layout=True)
    ax.plot(xs, LTEs_returned)
    ax.set_xlabel('dT (C)')
    ax.set_ylabel('LTE (years)')
    # fig.savefig('imgs/new_SR_penalty_325')
    plt.show()
    plt.close()


    # make a damage map using RBF interpolator
    dmg_obj2.make_contour_function_from_interpolator(LTE_desired=30, show_LTE_ctr=True)