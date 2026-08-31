import numpy as np

        
class Discretization:
    def __init__(self, nr = 5, ntheta = 79, nz = 30, OD = None, twall = None, length = None):
    
        self.nr = nr           # Number of radial discretization points
        self.ntheta = ntheta   # Number of circumferential discretization points per half-tube
        self.nz = nz           # Number of axial discretization points per tube

        self.dr = float('nan')  
        self.dtheta = float('nan')
        self.dz = float('nan')
        self.rpts = []
        self.thetapts = []
        self.zpts = []
        
        if OD is not None and twall is not None and length is not None:
            self.update(OD, twall, length)

        return
    
    def update(self, OD, twall, length):
        ID = OD - 2*twall
        self.dr = twall / (self.nr-1)   
        self.rpts = np.linspace(0.5*ID, 0.5*OD, self.nr)
        self.dtheta = np.pi/(self.ntheta-1) 
        self.thetapts = np.linspace(0, np.pi, self.ntheta)            
        self.dz = length / (self.nz-1)
        self.zpts = np.linspace(0, length, self.nz)       
        return
        
                


class SolutionOptions:
    def __init__(self):

        #--- Solution options that apply to both transient and steady state models
        self.wall_detail = '2D'               # Conduction model in tube wall: '0D' = solve for 1D radial conduction at each axial position for front/back tube surfaces, with average flux at the front surface
                                              #                                '1D' = solve for 1D radial conduction at each axial/circumferential position, 
                                              #                                '2D' = solve for 2D radial/circumferential conduction at each axial position    
        
        self.crosswall_avg_k = False          # Use constant cross-wall average thermal conductivity (computed at each circumferential/axial position) and neglect derivatives of thermal conductivity in conduction solution   
        self.is_adjacent_tubes = True         # Are there adjacent tubes (changes view factors to ambient)
        self.use_full_rad_exchange = True     # Solve radiosity model for IR exchange between adjacent tubes
        self.entry_length_correction = False  # Use entry length correction?
        self.calculate_stress = False
        
        
        #--- Transient solution options
        self.is_transient = False              # Is model transient?
        self.transient_soln_method = 'CN'      #'CN' (Crank-Nicolson), 'explicit', or 'implicit'.  Crank-Nicolson is most accurate, but implicit (backwards Euler) may reduce oscillations at large time steps
        self.is_separate_Tf = True             # If False, solve for fluid temperature at next time point simultaneously with wall temperature.
                                               # If True, solve for wall temperature first (using only fluid temperature at previous time point), then solve for fluid temperature
        self.is_pseudo_steady_state = False    # Solve model with pseudo-steady-state temperature gradients in wall    
        
        self.is_transient_sequential_panel_soln = False  # Solve each panel (or header) at all time points before moving to next panel.  Better for parallelization, but requires the wall temperature at which external heat transfer coefficients are evaluated to be fixed at all time points

        return


        
        
        
