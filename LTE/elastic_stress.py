
import numpy as np
import materials
 
# ELASTIC STRESS CALCULATIONS

class TubeElasticStress:
    def __init__(self, ri, ro, material, props = None):
        self.material_name = material   # Tube wall material
        self.ri = ri                    # Inner radius (m)
        self.ro = ro                    # Outer radius (m)
        
        self.wall = materials.create_material(self.material_name)
        if self.material_name != 'GenericConstantProp':
            self.wall = materials.create_material(self.material_name)
        else:
            self.wall = materials.create_material(self.material_name, props)

        return
    

    def calculate_pressure_stress(self, P, nr): # Pressure can be either P[z] or P[z,t]
        ci = self.ri**2/(self.ro**2-self.ri**2)
        rpts = np.linspace(self.ri, self.ro, nr)
        shape = list(P.shape)
        shape.insert(1,nr)
        shape.append(3)
        stress = np.zeros(shape)
        for r in range(nr):
            stress[:,r,...,0] = P * ci * (1.0 - self.ro**2/rpts[r]**2)   # Radial direction
            stress[:,r,...,1] = P * ci * (1.0 + self.ro**2/rpts[r]**2)   # Tangential direction (hoop stress)
            stress[:,r,...,2] = P * ci           # Axial direction
        return stress
        
    
    # Thermal stress (MPa): Tw can be (z,theta,r) or (z,theta,r,t)
    def calculate_thermal_stress(self, Tw, use_property_temperature_variation = False): #True):   
        ri = self.ri
        ro = self.ro
        nz, ntheta, nr = list(Tw.shape[0:3])
        rpts = np.linspace(self.ri, self.ro, nr)
        thetapts = np.linspace(0, np.pi, ntheta) 
        rvals = np.tile(rpts, ntheta)
        thetavals = np.repeat(thetapts, nr)
        
        # Calculate parameters
        order = 4
        X = np.ones((ntheta*nr, 2*order))
        for j in range(order):
            X[:,2*j] = np.cos((j+1)*thetavals) * (rvals**(j+1))
            X[:,2*j+1] = np.cos((j+1)*thetavals) * (rvals**-(j+1))
        XtXinvXt = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X), X)), np.matrix.transpose(X))    
        
        ci = ri**2/(ro**2-ri**2)
        R1 = (-np.log(ro/rpts) - ci*(1.0 - ro**2/rpts**2)*np.log(ro/ri))
        R2 = rpts/(ri**2+ro**2) * (1-ri**2/rpts**2) * (1-ro**2/rpts**2)
        T1 = (1.0 - np.log(ro/rpts) - ci*(1.0 + ro**2/rpts**2)*np.log(ro/ri))
        T2 = rpts/(ri**2+ro**2) * (3.0 - (ri**2+ro**2)/rpts**2 - (ri**2 * ro**2)/rpts**4)
        
        

        # Re-order columns and re-shape input array for convenience
        Tw2 = np.copy(Tw)
        nt = 1 if len(Tw.shape) == 3 else Tw.shape[3]
        Tw2 = np.moveaxis(np.copy(Tw), [0,1,2], [2,0,1])  # Change input array from (z,theta,r,t) to (theta, r, z, t)
        Tw2 = np.reshape(Tw2, (ntheta, nr, nz*nt)) # (ntheta, nr, nz*nt)
         
        # Calculate properties and average over cross-section
        rwts = np.reshape(rpts/rpts.sum(), (nr,1))
        Tref = 25 + 273.15  # Reference temperature
        E = self.wall.youngs_modulus(Tw2)  # (ntheta, nr, nz*nt)
        alpha = self.wall.cte(Tw2) 
        corr = E*alpha*(Tw2-Tref)
        E_avg = (E.mean(0) * rwts).sum(0)  # Average Young's modulus in each cross section (1, nz*nt)
        alpha_avg = (alpha.mean()*rwts).sum(0)
        corr_avg = (corr.mean(0)*rwts).sum(0)
        Tavg = (Tw2.mean(0) * rwts).sum(0)   # Average T over cross-sectional area (1,nz*nt)
        mu = self.wall.poisson_ratio(Tavg)
            
        # Fit coefficients to temperature solution
        Toavg = Tw2[:,-1,:].mean(0)   # Average outer wall T over theta positions (1, nz*nt)
        Tiavg = Tw2[:,0,:].mean(0)    # Average inner wall T over theta positions (1, nz*nt)
        rvals = np.reshape(rvals, (ntheta*nr,1))
        Ttheta = np.reshape(Tw2, (ntheta*nr, nz*nt)) - Toavg - (Tiavg-Toavg) * (np.log(ro/rvals) / np.log(ro/ri))
        coeffs = np.matmul(XtXinvXt, Ttheta)
        B1 = coeffs[1,:]     #(nz*nt)
        K = (Tiavg-Toavg) / np.log(ro/ri)
        
        # Calculate thermal stress
        stress = np.zeros((ntheta, nr, nz*nt, 4))
        thetapts2 = np.reshape(np.repeat(thetapts, nz*nt), (ntheta,nz*nt))
        for r in range(nr):
            if not use_property_temperature_variation:
                stress[:,r,:,0] = E_avg*alpha_avg/2/(1-mu) * (K*R1[r] + B1*np.cos(thetapts2)*R2[r])   # Radial [MPa]
                stress[:,r,:,1] = E_avg*alpha_avg/2/(1-mu) * (K*T1[r] + B1*np.cos(thetapts2)*T2[r])   # Tangential [MPa]
                stress[:,r,:,3] = E_avg*alpha_avg/2/(1-mu) * (B1*np.sin(thetapts2)*R2[r])  # Shear stress (r/theta), typically small                            
    
                # Axial stress [MPa] from generalized plane strain (zero axial force) without correction for temperature-dependent properties
                stress[:,r,:,2] = mu * (stress[:,r,:,0]+stress[:,r,:,1]) - E_avg*alpha_avg*(Tw2[:,r,:]-Tavg)   
            else:
                term1 = E[:,r,:]*alpha[:,r,:]/2/(1-mu)
                stress[:,r,:,0] = term1 * (K*R1[r] + B1*np.cos(thetapts2)*R2[r])   # Radial [MPa]
                stress[:,r,:,1] = term1 * (K*T1[r] + B1*np.cos(thetapts2)*T2[r])   # Tangential [MPa]
                stress[:,r,:,3] = term1 * (B1*np.sin(thetapts2)*R2[r])  # Shear stress (r/theta), typically small                            
                
                # Axial stress [MPa] from generalized plane strain (zero axial force) including correction for temperature-dependent properties
                stress[:,r,:,2] = mu * (stress[:,r,:,0]+stress[:,r,:,1]) - E[:,r,:]*(alpha[:,r,:]*(Tw2[:,r,:]-Tref) - corr_avg / E_avg)   
                
        # Reshape stress back to original shape of Tw
        stress = np.reshape(stress, (ntheta, nr, nz*nt*4))
        stress = np.reshape(stress, (ntheta, nr, nz, nt, 4))
        if nt == 1:
            stress = np.reshape(stress, (ntheta, nr, nz, 4))
        
        stress = np.moveaxis(stress, [0,1,2], [1,2,0])

        return stress
    
    
    

    def calculate_total_stress(self, P, Tw, use_property_temperature_variation = True):
        nr = Tw.shape[2]
        pressure = self.calculate_pressure_stress(P, nr)
        thermal = self.calculate_thermal_stress(Tw, use_property_temperature_variation)
        
        
        pressure = np.moveaxis(pressure, [0], [1])          # Change from (z,r,t) to (r,z,t)
        thermal = np.moveaxis(thermal, [0,1,2], [2,0,1])    # Change from (z,theta,r,t) to (theta, r, z, t)
        total = np.copy(thermal)
        for j in range(3):
            total[...,j] += pressure[...,j]   
            
        # Change back to original column order (z, theta, r, t) for thermal/total, (z,r,t) for pressure
        pressure = np.moveaxis(pressure, [1], [0])
        thermal = np.moveaxis(thermal, [0,1,2], [1,2,0])
        total = np.moveaxis(total, [0,1,2], [1,2,0])
       
        thermal_equiv = self.vonMises_equiv(thermal)
        pressure_equiv = self.vonMises_equiv(pressure)
        total_equiv = self.vonMises_equiv(total)

        return thermal_equiv, pressure_equiv, total_equiv
    
    
    def vonMises_equiv(self, stress):
        ncomp = stress.shape[-1] 
        equiv = (stress[...,0]-stress[...,1])**2 + (stress[...,0]-stress[...,2])**2 + (stress[...,1]-stress[...,2])**2
        for j in range(3,ncomp):
            equiv += 6*(stress[...,j]**2)
        equiv = (0.5*equiv)**0.5  # Equivalent stress  
        return equiv
    
    def intensity(self, stress):
        i0 = stress[...,0] - stress[...,1]
        i1 = stress[...,0] - stress[...,2]
        i2 = stress[...,1] - stress[...,2]
        intensity = np.maximum(np.maximum(i0,i1), i2)     
        return intensity
    
    