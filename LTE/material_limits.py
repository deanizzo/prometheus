"""
created by: jwenner on 4/28/2025
class for identifying corrosion velocity and temperature limits for solar receiver materials
temperatures are in C
Tcrit is defined as the maximum allowable film temperature

source for H230, 740H, 800H, A625, 316H: material selection for solar central receiver tubes by Laporte et al. (2021), which compiled other sources

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
from matplotlib import cm
import os
from pathlib import Path

class materialLimits:
    def __init__(self,mat):
        self.mat = mat  # receiver alloy material. Can be: 'A230', 'A617', '316H', '800H', '740H', 'A282', 'A625'
        self.interpolator       =None
        self.border_function    =None
        self.df_SR              =pd.DataFrame()
        # assing a material
        if self.mat == 'A230':
            self.Tcrit = 650 #(C) og source: McConohy & Kruizenga (2014)
        elif self.mat == '740H': 
            self.Tcrit = 650 #(C) Laporte (2021) decided on 650 because comparable alloy composition to other materials that have proven limit of 650 (C)
        elif self.mat == '800H':
            self.Tcrit = 650 #(C) og source: Bradshaw & Goods (2001)
        elif self.mat == 'A625':
            self.Tcrit = 630 #(C) og source: Miliozzi et al. (2001)
        elif self.mat == '316H':
            self.Tcrit = 600 #(C) og source: Miliozzi et al. (2001)
        elif self.mat == 'A617':
            self.Tcrit = 650 #(C) no source. Judgement call based on common comparison to A230
        elif self.mat =='A282':
            self.Tcrit = 650 #(C) Mark Messner doesn't know of any sources for this yet. He recommended keeping it the same as 740H 
        else:
            print('material not recognized')

        py_loc =Path(__file__).resolve()    # this line gets the current path's file string        
        file_string_option =f'{py_loc}/../dmg_tool_data/{self.mat}_SR_region.csv'  
        if os.path.exists(file_string_option):
            self.file_string_SR = file_string_option 
        elif self.mat in ('A282', '740H'):
            print('SR region does not exist for this material')
        else:
            raise ValueError('materialLimits class error: material not recognized')     
        return
        
    def estimate_Tfilm(self,Tcrown,Tf,R):
        """
        estimates the film temperature, where "film" is defined as surface inner crown surface temeprature similar to Laporte (2024)
        Tcrown - (C) crown temperature
        Tf     - (C) fluid temperature
        R      - resistance ratio (deltaT_cond/deltaT_conv)
        """
        deltaT_total = Tcrown - Tf # total temperature difference
        Tfilm = Tf + deltaT_total/(1+R) #based on definition of R and definition of deltaT_total
        return Tfilm
    
    def check_T_pts(self,T_pts):
        """
        compares the given temperature 
        Tpts - array (C)
        returns True if Tcheck is above the limit
        returns False if Tcheck is below the limit
        """
        toasty_array = np.array( T_pts > self.Tcrit)

        return toasty_array
    
    def penalize_corrosion(self, LTEs_input, dTs, Tfs, Rs):
        """
        applies the same magnitude of penalty to a corrosion region violation as SR penalty
        --
        LTEs         - array of preprocessed LTEs, with 0 values at corrosion violation locations
        dTs, Tfs     - (C) corresponding arrays of thermal operating points
        """
        def corrosion_penalty(dT, Tf, R):
            '''
            inputs are a single dT and Tf
            '''
            Tcrown          =dT+Tf
            dT_corrosion    =(self.Tcrit-Tf)*(1+R)  # the maximum dT above which corrosion occurs
            
            if (dT_corrosion-dT) > 0:
                print('materialLimits class warning: penalized an operating point that does not actually violate corrosion guidelines')
            
            return 100*(dT_corrosion-dT)
        
        LTEs_processed  =np.array([LTE_processed for LTE_processed in LTEs_input])      # make a new list to prevent modififying the old one
        # find the indices where LTEs have been set to zero
        indices_zero    =np.where(LTEs_processed == 0)[0]
        for index in indices_zero:
            LTEs_processed[index]=corrosion_penalty(dTs[index], Tfs[index], Rs[index])

        return LTEs_processed
    
    def get_plot_pts(self,R,xLow=100,xHigh=600,res=100): 
        """
        returns plot pts for corrosion limit based on thermal assumptions
        R      - resistance ratio (deltaT_cond/deltaT_conv)
        res    - number of points along the line
        """
        deltaT_totals = np.linspace(xLow,xHigh,res)
        Tfs = self.Tcrit - deltaT_totals/(1+R)
        return deltaT_totals, Tfs
    
    def make_SR_interpolator(self):
        if self.mat not in ['740H', 'A282']:    # the other 4 materials have SR interpolated regions. 
                                                # These two alloys have no SR region within dataset because their Sy is very high
            self.load_lookup_table()
            self.generate_RBF_interpolator()
        else:
            print('no SR region for 740H/A282 in Dataset Range')

    def load_lookup_table(self):
        self.df_SR =pd.read_csv(self.file_string_SR)

    def generate_RBF_interpolator(self):


        dTs_og  =self.df_SR.columns[1:].values.astype(float)
        Tfs_og  =self.df_SR['Tf'].values

        dTs     =np.tile(dTs_og, Tfs_og.shape)
        Tfs     =np.repeat(Tfs_og, dTs_og.shape)
        data_coords =np.dstack( (dTs,Tfs)).squeeze()
        data_vals   =self.df_SR.iloc[:,1:].values.flatten()
        # interp_f = scipy.interpolate.RectBivariateSpline(dTs,Tfs,data_vals.transpose(),kx=1,ky=1)
        interp_f    =scipy.interpolate.RBFInterpolator( data_coords, data_vals, neighbors=20, kernel='linear')
        self.interpolator = interp_f
        return

    def make_SR_border_function(self, show_function=False):
        """
        creates a stress ratio=2 contour for use in the penalty method
        
        """
        ratio =2    # the critical ratio of elastic to plastic stresses above which stress reset is predicted to occur

        # create and plot an SR border from the interpolator
        map_fig,map_ax   =self.make_SR_map_from_interpolator(ratio)

        # select a desired contour, get vertices, show on plot

        # get the desired LTE index from the level list
        ratio_index     =self.level_list.index(ratio)
        # grab the contour's vertices
        all_dT_verts    =self.ctf.get_paths()[ratio_index].vertices[:,0]
        all_Tf_verts    =self.ctf.get_paths()[ratio_index].vertices[:,1]
        
        # plot the contour function
        map_ax.plot(all_dT_verts, all_Tf_verts)

        # make 1D interpolation function that returns the dT that causes S.R. for any given Tf
        border_function         =scipy.interpolate.interp1d(all_Tf_verts, all_dT_verts, fill_value='extrapolate')
        self.border_function    =border_function
        print('border function created')
        if show_function:
            plt.show()
        plt.close()

        return 
    
    def make_SR_map_from_interpolator(self, user_level):
        """
        makes a stress rest ratio map for contour access and visualization. Note: based on RBF interpolator, so slightly different values than previously done
        ---
        user_level - desired ratio contour
        """
        print('making damage map and returning contours')
        # initialize plot
        fig, ax=plt.subplots()

        # get the SR curve coordinates
        dTs_SR = self.df_SR.columns[1:].values.astype(float)
        Tfs_SR = self.df_SR['Tf'].values

        # make a meshgrid of x,y values for interpolation use
        Tf_data_min =Tfs_SR.min()
        Tf_data_max =Tfs_SR.max()
        dT_data_min =dTs_SR.min()
        dT_data_max =dTs_SR.max()
        Tf_range  =np.linspace(Tf_data_min,Tf_data_max,250)
        dT_range  =np.linspace(dT_data_min, dT_data_max, 250)
        dT_samples, Tf_samples  =np.meshgrid(dT_range, Tf_range)
        # get the LTE values based on the RBF interpolator

        ratio_samples =self.interpolator( np.dstack( (dT_samples.flatten(),Tf_samples.flatten() )).squeeze() )

        # make the contour plot
        levels=[1, 3]
        levels.append(user_level)
        levels.sort()
        ctf=ax.contour(dT_range, Tf_range, ratio_samples.reshape(dT_samples.shape), levels=levels,cmap=cm.copper_r,zorder=1) #was using cm.cool or colors='black', levels=levels
        self.ctf=ctf
        self.level_list=levels
        ax.axes.clabel(ctf,inline=True,inline_spacing=1, fontsize=6, fmt=self.fmt,manual=False,levels=levels) #left click: place. center click: done
        fontsize=14
        ax.set_ylabel('fluid temperature (C)', fontsize=fontsize)
        ax.set_xlabel('total temperature difference (C)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)


        return fig,ax

    @staticmethod
    def fmt(x):
        # for formatting the damage map
        s = f"{x:.0f}"
        return s

    def check_SR(self, dTs, Tfs):
        """
        Returns True if the ratio of elastic to peak plastic stress is greater than 2
        dT and Tf arrays must be the same size and in C
        """
        if self.interpolator == None:
            self.make_SR_interpolator()
        interp_SRs = self.interpolator(np.dstack( (dTs,Tfs)).squeeze() )
        resets = np.array( interp_SRs > 2)
        return resets
    
    def penalize_SR(self, LTEs_input, dTs, Tfs):
        """
        further augments LTEs by changing each zero to a penalty value equal to the pythagorean distance between the closest SR point. Bigger penalty for more likely S.R.
        --
        LTEs - a nz*ntubes long flat array of lifetimes in years
        dTs  - a size like LTEs in celsius
        Tfs  - a size like LTEs in celsius
        """
        # get the SR curve coordinates
        dTs_SR = self.df_SR.columns[1:].values.astype(float)
        Tfs_SR = self.df_SR['Tf'].values

        # make the border function which returns the stress reset border
        if self.border_function == None:
            self.make_SR_border_function(show_function=False)

        # make an independent copy of the LTE list
        LTEs_output =np.array([LTE_input for LTE_input in LTEs_input])


        # make a function to calculate the distance for a single dT, Tf combo
        def SR_penalty(dT,Tf):
            '''
            inputs are a single dT and Tf
            '''
            # calculate the dT at which SR occurs, using the border function
            dT_SR   =self.border_function(Tf)
            if (dT-dT_SR) < 0:
                print('materialLimits class warning: penalized an operating point that is not expected to experience SR')
            # return -np.exp((dT-dT_SR)/10)    # we want the minimum distance because this is most accurate, being the closest SR dT, Tf point
            return 100*(dT_SR-dT)   # we want the minimum distance because this is most accurate, being the closest SR dT, Tf point
        
        # find the indices of where LTEs are set to zero
        indices_zero =np.where(LTEs_output == 0)[0]
        for index in indices_zero:
            LTEs_output[index]=SR_penalty(dTs[index], Tfs[index])

        return LTEs_output

if __name__ == "__main__":
    mat = materialLimits('A230')
    print('film temp is',mat.estimate_Tfilm(Tcrown=520,Tf=400,R=0.5) )
    xPts, yPts = mat.getPlotPts(R=0.5)
    
    fig,ax=plt.subplots()
    ax.plot(xPts,yPts)
    plt.show()