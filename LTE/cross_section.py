import numpy as np
from scipy.sparse import diags as sparse_diags
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sparse_linalg
import scipy.linalg as splinalg



class CurrentConditions:
    def __init__(self):
        self.Tw = float('nan')          # Wall temperature profile (theta, r) (K)
        self.Tf = float('nan')          # Fluid temperature profile (k)
        self.Tf_prevz = float('nan')    # Fluid temperature at previous axial position and current time point (K)
        self.Tf_prevzt = float('nan')   # Fluid temperature at previous axial position and previous time point (K)
        
        self.qabsnet = float('nan')     # Net absorbed solar flux (W/m2): f(theta)
        self.hin = float('nan')         # Internal heat transfer coefficient (W/m2/K): f(theta)
        self.hext = float('nan')        # External heat trasnfer coefficient (W/m2/K): f(theta)
        self.Tamb = float('nan')        # Extenral ambient temperature (K)
        self.mflow = float('nan')       # Tube mass flow (kg/s)
        self.tstep = float('nan')       # Current time step (s)
        return
    
    def populate(self, Tw, Tf, qabsnet, hin, hext, Tamb, mflow, tstep = None, Tf_prevz = None, Tf_prevzt = None):
        self.Tw = Tw
        self.Tf = Tf
        self.Tf_prevz = Tf_prevz
        self.Tf_prevzt = Tf_prevzt
        self.qabsnet = qabsnet
        self.hin = hin
        self.hext = hext
        self.Tamb = Tamb
        self.mflow = mflow
        self.tstep = tstep

        return
        
        
        

class CrossSection:
    def __init__(self, wall, htf, discretization, options):
                 
        self.wall = wall   # Tube wall material
        self.htf = htf     # Tube HTF material
        
        self.disc = discretization          # Tube wall discretization 
        self.ro = discretization.rpts[-1]   # Outer radius (m)
        self.ri = discretization.rpts[0]    # Inner radius (m)

        self.options = options                  # Solution options
        self.conditions = CurrentConditions()   # Current conditions 
         
        self.A = None           # Coefficient matrix
        self.Alocs = None       # Locations of nonzero entries for sparse matricies   
        self.bvec = None
        self.matrix_type = None
        self.phi = float('nan')
        
        self.dkdr = None
        self.dkdtheta = None
        self.tmp = None
        self.Tw2 = None
        self.rptsm = None

        self.initialize()
        return


    #=========================================================================
    def initialize(self):
        nr = self.disc.nr
        ntheta = self.disc.ntheta
        n = self.disc.ntheta*self.disc.nr
        ntot = n
        if self.options.is_transient and not self.options.is_separate_Tf:
            ntot += 1        
        
        phis = {'explicit':0.0,'implicit':1.0, 'CN': 0.5}
        self.phi = phis[self.options.transient_soln_method]
        
        self.bvec = np.zeros((ntot))
        self.dkdr = np.zeros((ntheta,nr))
        self.dkdtheta = np.zeros((ntheta,nr))
        self.b = np.zeros((ntheta,nr))
        self.tmp = np.zeros((ntheta,nr))
        self.Tw2 = np.zeros((ntheta+2,nr+2))
        self.k2 = np.zeros((ntheta,nr+2)) 
        self.rptsminv = 1.0/np.tile(self.disc.rpts,ntheta).reshape((ntheta,nr))

        
        #--- Set matrix type
        tcdim = int(self.options.wall_detail[0]) 
        self.is_sequential = False
        if not self.options.is_transient or self.options.is_pseudo_steady_state:  # Matrix is only used for steady-state solution in wall, banded and diagonal forms are possible
            if self.options.wall_detail == '1D' and self.options.crosswall_avg_k:
                self.matrix_type = None
            else:
                self.is_sequential = (tcdim <= 1 and self.disc.ntheta < 20)    # Solve equations sequentially for each theta node
                self.matrix_type = 'dense' if (self.is_sequential or n<50) else 'banded' 
        elif self.options.is_transient:
            if self.options.transient_soln_method == 'explicit':
                self.matrix_type = None
            elif self.options.is_separate_Tf: # Matrix is only used for wall, banded and diagonal forms are possible
                self.is_sequential = (tcdim <= 1 and self.disc.ntheta < 20)    # Solve equations sequentially for each theta node
                self.matrix_type = 'dense' if (self.is_sequential or n<50) else 'banded' 
            else:
                self.matrix_type = 'dense' if n<150 else 'sparse_csr'

        #--- Initialize coefficient matrix
        self.A = None
        if self.matrix_type == 'dense':
            self.A = np.zeros((ntot,ntot))
        elif self.matrix_type == 'banded':
            nrow = 3 if tcdim == 1 else 2*nr+1
            self.A = np.zeros((nrow, n))     
        elif self.matrix_type == 'sparse_diags':
            diags = [np.ones(n), np.ones(n-1), np.ones(n-1)] if tcdim < 2  else [np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-nr), np.ones(n-nr)]
            offsets = (0,1,-1) if tcdim < 2 else (0,1,-1,nr,-nr)
            self.A = sparse_diags(diags, offsets, (n,n)) 
        elif self.matrix_type == 'sparse_csr':  
            inds = np.arange(n)
            rows = inds.tolist() + inds[:-1].tolist() + (inds[:-1]+1).tolist()
            cols = inds.tolist() + (inds[:-1]+1).tolist() + inds[:-1].tolist()
            if tcdim > 1:
                rows += inds[:-nr].tolist() + (inds[:-nr]+nr).tolist()
                cols += (inds[:-nr]+nr).tolist() + inds[:-nr].tolist()
            if self.ntot > n:  # Fluid terms are included     
                rows += np.arange(0, n, nr).tolist() + [ntot-1] + [(ntot-1) for j in range(ntheta-1)]
                cols += [(ntot-1) for j in range(ntheta)] + [ntot-1] + np.arange(0, n-nr, nr).tolist()
            n_nonzero = len(rows)
            self.A = csr_matrix((np.ones(n_nonzero), (rows,cols)), shape = (ntot, ntot), dtype = float)  # csr slightly faster than csc   
            self.Alocs = [rows, cols]   

        return
    

    #=========================================================================
    def calculate_wall_coefficient_matrix(self, wall_properties = None, fluid_properties = None, Tfin = None, Twin = None):
        '''
        Populate coefficient matrix and RHS vector for linear equations used for tube wall temperature solution
        wall_properties, fluid_properties = pre-computed wall and fluid properties (optional)
        Tfin = fluid inlet temperature (optional, only needed for first axial node)
        Twin = fixed inner wall temperature (optional, only needed for pseudo-steady-state solution)
        '''
        #---- Stacked arrays are set up as:    
        #|____r0____| |____r1_____|... |____rN-1_____| |____r0____| |____r1_____|... |____rN-1_____| ...etc
        #|_________________t0________________________| |_________________t1________________________| ...etc


        nr = self.disc.nr
        nt = self.disc.ntheta
        tcdim = int(self.options.wall_detail[0])
        n = nt * nr
        roffset = 1         # Number of points in stacked T array from current to next/previous r position
        toffset = nr        # Number of points in stacked T array from current to next/previous theta position 
        is_transient = self.options.is_transient and Twin is None

        #--- Evaluate properties
        k = self.wall.k(self.conditions.Tw) if wall_properties is None else wall_properties['k']
        if self.options.crosswall_avg_k:
            k[:,:] = k.mean(1, keepdims = True)
            
        if self.conditions.tstep is not None:
            if wall_properties is None:
                rho = self.wall.density(self.conditions.Tw)
                cp = self.wall.cp(self.conditions.Tw)
            else:
                rho = wall_properties['density']
                cp = wall_properties['cp']
            alpha = k / rho / cp  
            
            
        #--- Calculate groups of parameters that are used multiple times
        cin = 2*self.disc.dr*self.conditions.hin/k[:,0]
        cout1 = 2*self.disc.dr*self.conditions.qabsnet/k[:,-1]
        cout2 = 2*self.disc.dr*self.conditions.hext/k[:,-1]
        
        #--- Set up wall temperature array including "imaginary" nodes (~9% of time in this function)
        self.Tw2.fill(0.0)
        self.Tw2[1:-1,1:-1] = self.conditions.Tw
        self.Tw2[1:-1,0] = self.conditions.Tw[:,1] - cin*(self.conditions.Tw[:,0] - self.conditions.Tf) if Twin is None else Twin   
        self.Tw2[1:-1,-1] = self.conditions.Tw[:,-2] + cout1 - cout2*(self.conditions.Tw[:,-1] - self.conditions.Tamb)
        self.Tw2[0,1:-1] = self.conditions.Tw[1,:]
        self.Tw2[-1,1:-1] = self.conditions.Tw[-2,:]            

        #--- Evaluate derivatives of thermal conductivity with respect to position (~30% of time in this function, ~11% of solution time)    
        self.dkdr.fill(0.0)
        self.dkdtheta.fill(0.0)
        if not self.options.crosswall_avg_k:
            #k2 = np.zeros((nt, nr+2))
            self.k2.fill(0.0)
            self.k2[:,1:-1] = k
            self.k2[:,0] = self.wall.k(self.Tw2[1:-1,0])  # 62% of time to evaluate derivates comes from evaluation wall k at these nodes..
            self.k2[:,-1] = self.wall.k(self.Tw2[1:-1,-1])
            self.dkdr = (self.k2[:,2:]-self.k2[:,:-2])/(2*self.disc.dr)
            if tcdim > 1:
                self.dkdtheta[1:-1, :] = (k[2:,:] - k[:-2,:]) / (2*self.disc.dtheta)         


        #--- Set coefficients
        beta1 = 1.0 / self.disc.dr**2 
        gamma1 = 1.0  / 2 / self.disc.dr  
        beta2 = 1.0  / self.disc.dtheta**2  
        gamma2 = 1.0  / 2 / self.disc.dtheta  
        ntot = n
        if not is_transient:
            a1 = beta1*k
            a2 = gamma1*k*(self.rptsminv + self.dkdr/k)
            C11 = -2*a1
            C12 = 0.0 
            D11 = a1+a2
            D21 = a1-a2
            D12 = 0.0
            D22 = 0.0
            if tcdim > 1:
                v = k*self.rptsminv**2
                a1 = v * beta2
                a2 = v * gamma2 * self.dkdtheta/k             
                C11 -= 2*a1
                E11 = a1+a2
                E21 = a1-a2
        else:
            ntot = ntot+1 if not self.options.is_separate_Tf else ntot
            a1 = beta1*alpha
            a2 = gamma1*alpha*(self.rptsminv + self.dkdr/k)
            asum = a1+a2
            adiff = a1-a2
            C11 = 1+2*self.phi*a1
            C12 = -2*a1 + C11  # 1-2*(1-phi)*a1
            D11 = -self.phi*asum
            D21 = -self.phi*adiff
            D12 = asum + D11 #(1-phi)*asum
            D22 = adiff + D21 #(1-phi)*adiff
            if tcdim > 1:
                v = alpha*self.rptsminv**2
                a1 = v * beta2
                a2 = v * gamma2 * self.dkdtheta/k             
                asum = a1+a2
                adiff = a1-a2
                C11 += 2*self.phi*a1
                C12 -= 2*(1-self.phi)*a1
                E11 = -self.phi*asum
                E21 = -self.phi*adiff
                E12 = asum + E11 #(1-sefl.phi)*asum
                E22 = adiff + E21 #(1-self.phi)*adiff            
        
            

        #--- Set RHS  (~17% of time in this function, ~6% of solution time)
        term1 = cin*D21[:,0]
        term2a = cout1*D11[:,-1]
        term2b = cout2*D11[:,-1]

        b = self.tmp
        if not is_transient: # Steady state
            #b = np.zeros((nt,nr))
            b.fill(0.0)
            b[:,0] = -term1*self.conditions.Tf if Twin is None else Twin #-cin*self.conditions.hin * D21[:,0]*self.conditions.Tf if Twin is None else Twin
            b[:,-1] = -term2a - term2b*self.conditions.Tamb  #-cout* D11[:,-1]*(self.conditions.qabsnet + self.conditions.hext*self.conditions.Tamb) 
        
        else:  # Transient
            b = C12*self.Tw2[1:-1,1:-1] + D12*self.Tw2[1:-1,2:] + D22*self.Tw2[1:-1,:-2]
            b[:,-1] -= term2a + term2b*self.conditions.Tamb #cout*D11[:,-1]*(self.conditions.qabsnet + self.conditions.hext*self.conditions.Tamb)
            if ntot == n:  # Fluid temperature not solved simultaneously
                b[:,0] -= term1*self.conditions.Tf
            if tcdim > 1:
                b += E12*self.Tw2[2:,1:-1] + E22*self.Tw2[:-2,1:-1]
        self.bvec[0:n] = b.ravel()


        if is_transient and self.options.transient_soln_method == 'explicit':
            return 
        

        #--- Calculate diagonals (~16% of time in this function)
        self.tmp.fill(0.0)
        self.tmp[:,:] = C11
        self.tmp[:,0] -= term1
        self.tmp[:,-1] -= term2b
        if Twin is not None:
            self.tmp[:,0] = 1.0
        diag = np.copy(self.tmp.ravel())
        
        self.tmp[:,:] = D11
        self.tmp[:,0] += D21[:,0]
        self.tmp[:,-1] = 0.0
        if Twin is not None:
            self.tmp[:,0] = 0.0        
        diag_r_up = np.copy(self.tmp.ravel()[:-1])
        
        self.tmp[:,:] = D21
        self.tmp[:,0] = 0.0
        self.tmp[:,-1] += D11[:,-1] 
        diag_r_down = np.copy(self.tmp.ravel()[1:])

        if tcdim > 1:
            self.tmp[:,:] = E11
            self.tmp[0,:] += E21[0,:]
            self.tmp[-1,:] = 0.0
            if Twin is not None:
                self.tmp[:,0] = 0.0
            diag_t_up = np.copy(self.tmp.ravel()[:-nr])
            
            self.tmp[:,:] = E21
            self.tmp[0,:] = 0.0     
            self.tmp[-1,:] += E11[-1,:]
            if Twin is not None:
                self.tmp[:,0] = 0.0
            diag_t_down = np.copy(self.tmp.ravel()[nr:])



        
        #--- Fill in coefficient matrix  (~2% of time in this function)
        #fluid_lhs, wall_coeffs = [None, None]
        if ntot > n: # Fluid and solid nodes considered simultaneously
            fluid_lhs, fluid_rhs, wall_coeffs = self.get_fluid_coefficients(fluid_properties, Tfin)
            self.bvec[-1] = fluid_rhs

        if self.matrix_type == 'banded':
            if tcdim == 1:
                self.A[0,1:] = diag_r_up
                self.A[1,:] = diag
                self.A[2,:-1] = diag_r_down
            elif tcdim == 2:
                self.A[0,nr:] = diag_t_up
                self.A[nr-1,1:] = diag_r_up
                self.A[nr,:] = diag
                self.A[nr+1,:-1] = diag_r_down
                self.A[2*nr, :-nr] = diag_t_down  
        elif self.matrix_type == 'sparse_diags':
            self.A.setdiag(diag,0)
            self.A.setdiag(diag_r_up, 1)
            self.A.setdiag(diag_r_down, -1)
            if tcdim >1:
                self.A.setdiag(diag_t_up, nr)
                self.A.setdiag(diag_t_up, -nr)
                    
        elif self.matrix_type == 'sparse_csr':
            coeffs = np.zeros(len(self.Alocs[0]))
            coeffs[0:n] = diag
            coeffs[n:2*n-1] = diag_r_up
            coeffs[2*n-1:3*n-2] = diag_r_down
            i = n+2*(n-1)
            if tcdim > 1:
                coeffs[i:i+n-nr] = diag_t_up
                coeffs[i+n-nr:i+2*(n-nr)] = diag_t_down
                i+= 2*(n-nr)
            if ntot > n:  # Fluid terms are included     
                coeffs[i:i+nt] = cin*self.conditions.hin * D21[:,0] # LHS coefficients on Tf in equation for inner tube boundary nodes
                coeffs[i+nt] = fluid_lhs
                coeffs[i+nt+1:i+2*nt] = wall_coeffs
            self.A[self.Alocs[0], self.Alocs[1]] = coeffs
            
        elif self.matrix_type == 'dense':
            inds = np.arange(n)
            self.A[inds, inds] = diag
            self.A[inds[:-roffset], inds[:-roffset]+roffset] = diag_r_up
            self.A[inds[:-roffset]+roffset, inds[:-roffset]] = diag_r_down
            if tcdim > 1:
                self.A[inds[:-toffset], inds[:-toffset]+toffset] = diag_t_up
                self.A[inds[:-toffset]+toffset,inds[:-toffset]] = diag_t_down   
            if ntot > n:  # Fluid and solid nodes considered simultaneously
                rlow = np.arange(0, n, nr) 
                self.A[rlow,-1] = cin*self.conditions.hin * D21[:,0]  
                self.A[-1,-1] = fluid_lhs
                self.A[-1,rlow[:-1]] = wall_coeffs

        return

    
    #==========================================================================
    def get_fluid_coefficients(self, properties = None, Tfin = None):
        Acf = 0.5*np.pi*self.ri**2          # Cross-sectional area for half-tube
        mflow = 0.5*self.conditions.mflow   # Mass flow in half-tube
        if properties is None:
            rhof = self.htf.density(self.conditions.Tf)
            cpf = self.htf.cp(self.conditions.Tf)
        else:
            rhof = properties['density']
            cpf = properties['cp']

        c1 = mflow/rhof/Acf * self.conditions.tstep/self.disc.dz
        eta11 = self.phi*c1
        eta12 = (1-self.phi)*c1
        c2 = self.ri*self.disc.dtheta*self.conditions.hin[:-1]/rhof/cpf/Acf * self.conditions.tstep  
        eta21 = self.phi*c2 
        eta22 = c2-eta21 #(1-self.phi)*c2  
        wall_coeffs = None

        if Tfin is not None:
            lhs = 1
            rhs = Tfin
        else:
            lhs = 1+eta11+eta21.sum()  # LHS coefficients on Tf in equation for Tf
            rhs = self.conditions.Tf*(1-eta12-eta22.sum()) + eta12*self.conditions.Tf_prevzt + eta11*self.conditions.Tf_prevz + (c2*self.conditions.Tw[:-1,0]).sum() 
            if not self.options.is_separate_Tf:
                wall_coeffs = -eta21         # LHS coefficients on boundary wall T nodes
                rhs -= (eta21*self.conditions.Tw[:-1,0]).sum()  
 
        return lhs, rhs, wall_coeffs   
    


    #==========================================================================
    def get_coefficient_matrix_for_pseudo_steady_state(self, wall_properties = None, fluid_properties = None, Tfin = None):
        # Stacked arrays are set up as (theta0, theta1, ...thetaN, Tf)  

        if wall_properties is None:
            rho = self.wall.density(self.conditions.Tw).mean(1)
            cp = self.wall.cp(self.conditions.Tw).mean(1)
        else:
            rho = wall_properties['density'].mean(1)
            cp = wall_properties['cp'].mean(1)
        psi = self.conditions.tstep/rho/cp/(0.5*(self.ro**2-self.ri**2))
        n = self.disc.ntheta if self.options.is_separate_Tf else self.disc.ntheta+1

        A = np.zeros((n,n))
        bvec = np.zeros((n))

        inds = np.arange(self.disc.ntheta)  
        A[inds,inds] = 1+psi*self.phi*self.conditions.hin*self.ri
        bvec[inds] = self.conditions.Tw[:,0] \
                     + psi*self.ro*(self.conditions.qabsnet - self.conditions.hext*(self.conditions.Tw[:,-1]-self.conditions.Tamb)) \
                     - psi*(1-self.phi)*self.conditions.hin*self.ri*(self.conditions.Tw[:,0] - self.conditions.Tf)
              
        if self.options.is_separate_Tf:
            bvec[inds] += psi*self.phi*self.conditions.hin*self.ri*self.conditions.Tf 
        else:
            A[inds,-1] = -psi*self.phi*self.conditions.hin*self.ri  # LHS coefficients on Tf in equation for inner tube boundary nodes
            A[-1,-1], bvec[-1], wall_coeffs = self.get_fluid_coefficients(fluid_properties, Tfin)
            A[-1,inds[:-1]] = wall_coeffs  # LHS coefficients on boundary wall T nodes
        return A, bvec        
    
    
    #==========================================================================
    def set_up_banded_matrix(self, offsets):  # Set up banded matrix from either dense or sparse matrix
        noff = len(offsets)
        Aband = np.zeros((noff, self.A.shape[1]))
        for j in range(noff):
            i = offsets[j]
            if i>=0:
                Aband[j,i:] = self.A.diagonal(i)  
            else:
                Aband[j,:i] = self.A.diagonal(i)
        return Aband
    
    
    def solve_linear_eqns(self):
        if self.matrix_type == 'banded':
            tcdim = int(self.options.wall_detail[0])
            noff = self.disc.nr if tcdim == 2 else 1   
            Tvec = splinalg.solve_banded((noff,noff), self.A, self.bvec) 
        elif self.matrix_type == 'sparse_csr':
            Tvec = sparse_linalg.spsolve(self.A, self.bvec)
        else:
            Tvec = np.linalg.solve(self.A, self.bvec)
        return Tvec

            

    #=========================================================================
    #--- Solve for steady state tube wall temperature profiles in a given axial cross-section 
    def solve_wall_cross_section_steady_state(self, wall_properties = None, Twin = None):
        tcdim = int(self.options.wall_detail[0])    # Set number of dimensions of thermal conduction to use in the solution
        nr = len(self.disc.rpts)
        nt = len(self.disc.thetapts) 
        hext = np.maximum(self.conditions.hext, 1.e-6)   
        
        #--- 1D radial conduction and constant tube k (can be solved sequentially at each theta position)
        if tcdim <= 1 and self.options.crosswall_avg_k:  
            Tsoln = np.zeros((nt,nr))  
            if wall_properties is None:
                kwallavg = self.wall.k(self.conditions.Tw).mean(1)
            else:
                kwallavg = wall_properties('k').mean(1)
            Rcond = np.log(self.ro/self.ri) / kwallavg         
            Rext = 1. / self.ro / hext  
            if Twin is None:
                Rin = 1. / self.ri / self.conditions.hin 
                Rtot = Rin + Rcond + Rext
                for r in range(nr):
                    Tsoln[:, r] = self.conditions.Tf + 1/hext/Rtot * (self.conditions.qabsnet - hext*(self.conditions.Tf-self.conditions.Tamb)) * (Rin + np.log(self.disc.rpts[r]/self.ri) / kwallavg) 
            else:
                Rtot = Rcond + Rext
                for r in range(nr):
                    Tsoln[:, r] = Twin + 1/hext/Rtot * (self.conditions.qabsnet - hext*(self.conditions.Tf-self.conditions.Tamb)) * (np.log(self.disc.rpts[r]/self.ri) / kwallavg) 
            return Tsoln
        
        #--- 2D conduction, or 1D conduction without crosswall-average k
        self.calculate_wall_coefficient_matrix(wall_properties, Twin = Twin) 
        if not self.is_sequential:   
            Tvec = self.solve_linear_eqns()
            Tsoln = np.reshape(Tvec, (nt,nr)) 
        else:
            Tsoln = np.zeros((nt,nr))  
            for t in range(nt):  # Solve 1D models consecutively for each circumferential position
                Asec = self.A[t*nr:(t+1)*nr, t*nr:(t+1)*nr] if self.matrix_type == 'dense' else self.A[t*nr:(t+1)*nr, t*nr:(t+1)*nr].todense()
                Tsoln[t,:] = np.linalg.solve(Asec, self.bvec[t*nr:(t+1)*nr])         
        
        return Tsoln

    
    #=========================================================================
    def solve_cross_section_at_time_point(self, wall_properties = None, fluid_properties = None, Tfin = None):
        nr = self.disc.nr
        nt = self.disc.ntheta

        if not self.options.is_pseudo_steady_state:
            self.calculate_wall_coefficient_matrix(wall_properties, fluid_properties, Tfin)
            if not self.options.is_separate_Tf:   # Solve wall/fluid T simultaneously
                Tvec = self.solve_linear_eqns() if self.options.transient_soln_method != 'explicit' else self.bvec
                Twsoln = np.reshape(Tvec[0:-1], (nt,nr))
                Tfsoln = Tvec[-1]
                
            else:  # Solve wall/fluid T sequentially.
                if self.options.transient_soln_method == 'explicit':
                    Tvec = self.bvec
                elif not self.is_sequential: # not solve_sequentially:
                    Tvec = self.solve_linear_eqns()
                else:
                    Tvec = np.zeros_like(self.bvec)
                    for t in range(nt):  # Solve for each theta-position sequentially
                        Asec = self.A[t*nr:(t+1)*nr, t*nr:(t+1)*nr] if self.matrix_type == 'dense' else self.A[t*nr:(t+1)*nr, t*nr:(t+1)*nr].todense()
                        Tvec[t*nr:(t+1)*nr] = np.linalg.solve(Asec, self.bvec[t*nr:(t+1)*nr])   # Note, banded method is inefficient for small matricies  
                Twsoln = np.reshape(Tvec, (nt,nr))
                self.conditions.Tw = Twsoln         # Solve for fluid using updated wall T
                fluid_lhs, fluid_rhs, wall_coeffs = self.get_fluid_coefficients(fluid_properties, Tfin)  
                Tfsoln = fluid_rhs / fluid_lhs

        
        else:  # Pseudo-steady-state solution
            A, bvec = self.get_coefficient_matrix_for_pseudo_steady_state(wall_properties, fluid_properties, Tfin)

            if not self.options.is_separate_Tf:   # Solve wall/fluid T simultaneously.  Uses dense matricies -> might still be a way to speed up with sparse?
                Tvec = np.linalg.solve(A, bvec) if self.options.transient_soln_method != 'explicit' else bvec
                Tfsoln = Tvec[-1]
                Twsoln = self.solve_wall_cross_section_steady_state(Twin = Tvec[:-1])  

            else:  # Solve wall/fluid T sequentially.
                Tinner = bvec / np.diagonal(A) if self.options.transient_soln_method != 'explicit' else bvec  # Inner wall T
                Twsoln = self.solve_wall_cross_section_steady_state(Twin = Tinner)  
                self.conditions.Tw = Twsoln    # Solve for fluid using updated wall T
                fluid_lhs, fluid_rhs, wall_coeffs = self.get_fluid_coefficients(fluid_properties, Tfin)  # Solve for fluid using updated wall T
                Tfsoln = fluid_rhs / fluid_lhs

        return Twsoln, Tfsoln
    

