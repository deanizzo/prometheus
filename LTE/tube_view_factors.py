'''
View factor between differential element on surface to tube to finite element on surface of adjacent tube
Note angles alpha (element 1) and beta (element 2) are defined with alpha = 0 and beta = 0 at the front tip of each tube
'''

from math import pi, sin, cos, asin, acos, atan, floor
import numpy as np


# Critical angles for energy emitted from angular position alpha. View factor to all beta below/above these limits is zero
def critical_angles(alpha):
    s = sin(alpha)
    c = cos(alpha)
    beta_low = asin( ( 2 - s - 2*c*(1.-s)**0.5) / (5 - 4*s) )
    beta_high = asin( 2*s**2 - s + 2*c*(s-s**2)**0.5)
    return [beta_low, beta_high]
    



#---------------------------------------------------------------------------
# View factor from differential element at angular position alpha to infinite length between angular positions (beta +/- 0.5*delta_beta)
# Note this method uses numerical integration between beta +/- delta_beta  
def vf_differential_to_infinite_length(alpha, beta, delta_beta, n = 100):   

    if (alpha > 80*pi/180 and beta + 0.5*delta_beta > 80*pi/180):  # Function is sharply peaked as alpha and beta approach 90 degrees, use more discretization points for better resolution
        n = max(n, 500)
        
    sa = sin(alpha)
    ca = cos(alpha)       
    beta1, beta2 = critical_angles(alpha)    
    vf = 0.0
    if beta+0.5*delta_beta > beta1 and beta-0.5*delta_beta < beta2:    
        bmin = max(beta1, beta-0.5*delta_beta)   
        bmax = min(beta2, beta + 0.5*delta_beta)
        db = (bmax-bmin)/n    
        for i in range(n):
            b = bmin + (i+0.5)*db
            sb = sin(b)
            cb = cos(b)
            vf += db * (2*sa-sa*sb+ca*cb-1) *(2*sb-sa*sb+ca*cb-1) / 2 /  ((2-sa-sb)**2 + (ca-cb)**2)**1.5
    
    vf = max(vf, 0.0)    # Ensure limits are realistic--> only needed as alpha and beta approach 90 degrees
    vf = min(vf, 1.0)   
    return vf



#---------------------------------------------------------------------------
# View factor from finite angular element centered at angle alpha to infinite length between angular positions (beta +/- 0.5*delta_beta)
# Note this method uses numerical integration between alpha +/- delta_alpha and beta +/- delta_beta  
def vf_to_infinite_length(alpha, delta_alpha, beta, delta_beta, nalpha = 25, nbeta = 100):    
    
    if (alpha+0.5*delta_alpha > 80*pi/180 and beta + 0.5*delta_beta > 80*pi/180):  # Function is sharply peaked as alpha and beta approach 90 degrees, use more discretization points for better resolution 
        nbeta = max(nbeta, 500)
    
    vf = 0.0
    da = delta_alpha/nalpha
    for j in range(nalpha):
        a = (alpha - 0.5*delta_alpha) + (j+0.5)*delta_alpha/nalpha
        sa = sin(a)
        ca = cos(a)       
        beta1, beta2 = critical_angles(a)    
        
        vfloc = 0.0
        if beta+0.5*delta_beta > beta1 and beta-0.5*delta_beta < beta2:    
            bmin = max(beta1, beta-0.5*delta_beta)
            bmax = min(beta2, beta + 0.5*delta_beta)
            db = (bmax-bmin)/nbeta    
            for i in range(nbeta):
                b = bmin + (i+0.5)*db
                sb = sin(b)
                cb = cos(b)
                vfloc += db * (2*sa-sa*sb+ca*cb-1) *(2*sb-sa*sb+ca*cb-1) / 2 /  ((2-sa-sb)**2 + (ca-cb)**2)**1.5
        
        vf += vfloc * da        
                
    vf /= delta_alpha
    
    vf = max(vf, 0.0)    # Ensure limits are realistic--> only needed as alpha and beta approach 90 degrees
    vf = min(vf, 1.0)   
    return vf




#---------------------------------------------------------------------------
# View factor from differential element at angular position alpha to finite height between angular positions (beta +/- 0.5*delta_beta)
# Axial position and element size is defined relative to the tube radius.  The axial distance between the element centroid is eta*rtube and element size is delta_eta*rtube
# Note this method uses numerical integration between beta +/- delta_beta  
def vf_differential_to_finite_length(alpha, beta, delta_beta, eta, delta_eta, n = 100):    
    
    if (alpha > 80*pi/180 and beta + 0.5*delta_beta > 80*pi/180):  # Function is sharply peaked as alpha and beta approach 90 degrees, use more discretization points for better resolution
        n = max(n, 500)
        
    sa = sin(alpha)
    ca = cos(alpha)
    beta1, beta2 = critical_angles(alpha)   
    vf = 0.0
    if beta+0.5*delta_beta > beta1 and beta-0.5*delta_beta < beta2:
        bmin = max(beta1, beta - 0.5*delta_beta)
        bmax = min(beta2, beta + 0.5*delta_beta)
        db = (bmax-bmin)/n
        vf = 0.0
        for i in range(n):
            b = bmin + (i+0.5)*db
            sb = sin(b)
            cb = cos(b)        
            absq = (2-sb-sa)**2 + (cb-ca)**2
            vfloc = 0.0
            vfloc += (eta + 0.5*delta_eta)*absq**0.5/(absq+(eta + 0.5*delta_eta)**2)
            vfloc -= (eta - 0.5*delta_eta)*absq**0.5/(absq+(eta - 0.5*delta_eta)**2)
            vfloc += atan((eta + 0.5*delta_eta)/absq**0.5)
            vfloc -= atan((eta - 0.5*delta_eta)/absq**0.5)
            vfloc *= db*(2*sa-sa*sb+ca*cb-1)*(2*sb-sa*sb+ca*cb-1)/(2*pi*absq**1.5)        
            vf += vfloc
            
    vf = max(vf, 0.0)    # Enforce limits (degeneracies near alpha, beta = 90 can cause minor inconsistencies because of numerical integration)
    vf = min(vf, 1.0)   
    return vf


#---------------------------------------------------------------------------
# Ray tracing for view factors from differential element at angular position alpha
def ray_trace_differential_elem(alpha, rtube = 0.02, delta_angle = 2*pi/180, delta_eta = 0.1, nray = 1000000, rng_seed = 123):    
    nz = 100
    dz = delta_eta * rtube
    dtheta = delta_angle
    angle = alpha
    ntheta = int(pi/2/delta_angle)
    np.random.seed(rng_seed)  
    nhit = np.zeros((ntheta, nz))
    nloss = 0.0
    
    for r in range(nray):
        rand = np.random.random_sample(2)
        
        # Emission position
        sp = [rtube*sin(angle), rtube*cos(angle), 0.0]
        
        # Surface normal vector at emission position
        sn = [sin(angle), cos(angle), 0.0]
    
        # Emission direction in surface-normal coordinate system
        sin_alpha = rand[0]**0.5
        cos_alpha = (1.0-sin_alpha**2)**0.5
        beta = 2*pi*rand[1]
        rsn = [sin_alpha*cos(beta), sin_alpha*sin(beta), cos_alpha]  
        
        # Emission direction in global coordinate system
        rg = np.zeros(3)
        if(abs(sn[2]) == 1):
            rg[0] = rsn[0]
            rg[1] = sn[2]*rsn[1]
            rg[2] = sn[2]*rsn[2]
        else:
            den = (1.0-sn[2]**2)**0.5
            rg[0] = sn[1]*rsn[0]/den + sn[0]*sn[2]*rsn[1]/den + sn[0]*rsn[2]
            rg[1] = -sn[0]*rsn[0]/den + sn[1]*sn[2]*rsn[1]/den + sn[1]*rsn[2]
            rg[2] = -den*rsn[1] + sn[2]*rsn[2]
    
        # Check if ray hits adjacent tube
        if rg[0] < 0: # Only rays travelling in +x direction have the possibility of hitting adjacent tube
            nloss +=1
        else: 
            a = rg[0]**2 + rg[1]**2
            b = 2*rg[0]*(sp[0]-2*rtube)+2*rg[1]*sp[1]
            c = (sp[0]-2*rtube)**2+sp[1]**2 - rtube**2
            test = b**2-4*a*c
            if (test < 0):
                nloss +=1
            else:
                dist = (-b-test**0.5)/(2*a)  
                hp = [sp[0]+dist*rg[0], sp[1]+dist*rg[1], sp[2]+dist*rg[2]]  # Point where ray strikes tube
                theta = acos(hp[1]/rtube)
                thetabin = int(floor(theta/dtheta))
                if hp[2] > 0:
                    zbin = int(floor((hp[2]+0.5*dz)/dz))
                else:
                    zbin = int(floor(abs(hp[2]-0.5*dz)/dz))
                
                if thetabin <0 or zbin < 0:
                    print ('Error: angular bin index = %d, height bin index = %d'%(thetabin, zbin))
                    break
                
                if zbin < nz:
                    nhit[thetabin, zbin] += 1
    
    
    eta_pts = np.arange(nz, dtype = float) * delta_eta + 0.5*delta_eta
    beta_pts = np.arange(ntheta, dtype = float)*dtheta + 0.5*dtheta 
    vf = np.zeros_like(nhit)
    vf[:,0] = nhit[:,0] / nray
    for j in range(1,nz):
        vf[:,j] = 0.5*nhit[:,j] / nray  # All axial bins other than first count rays hitting both above and below differential emitting element
    
    return {'vf_tube': vf, 'vf_amb':nloss/nray, 'eta_pts':eta_pts, 'beta_pts':beta_pts }





#===================================================================================================================
# Function to calculate view factors to ambient and view factors between discretized tube elements for each angular position on the tube (assumes infinite length and neglects end effects)  
# rtube = outer tube radius
# Nt = number of circumferential elements per quarter-tube
# dz = axial element spacing (only include to check applicability of infinite-length assumption)
def calculate_view_factors(rtube, Nt, dz = None):

    vf_to_elem = np.zeros((Nt+1, Nt+1)) # Only considering non-zero view factors between elements at the same axial position for now... ok unless axial position spacing > 2*rtube

    if dz is not None and dz / rtube < 2:
        print ('Warning: axial tube spacing %.1f times the tube radius. May need to consider nonzero view factors between elements at different axial positions' %(dz/rtube))
    
    dtheta = 0.5*pi/Nt
    for t in range(Nt):   # Loop over all elements except
        theta = (t+0.5) * dtheta
        if (theta < pi/2 - 0.001):
            vf_to_elem[t,-1] = 1.0 - vf_differential_to_infinite_length(theta, 45*pi/180, 90*pi/180, n = 200)    # View factor to ambient (last element)
            for b in range(Nt):
                beta = (b+0.5) * dtheta
                vf_to_elem[t,b] = vf_differential_to_infinite_length(theta, beta, dtheta, n = 25)  
                
        else:  # Manual limit at point where tubes connect due to discontinuities. Element only sees opposing element
            vf_to_elem[t,:] = 0.0
            vf_to_elem[t,t]= 1.0
    
    
    # View factors from aperture to tube elements = (view factor from tube element to aperture) * (ratio of tube element area to aperture area)
    elem_to_aper_area = dtheta
    vf_to_elem[-1,:] = elem_to_aper_area * vf_to_elem[:,-1]  
        
    return vf_to_elem



#===================================================================================================================
# Function to calculate view factors between each tube element and ambient
# rtube = outer tube radius
# Nt = number of circumferential elements per quarter-tube      
def calculate_view_factors_to_ambient(rtube, Nt):    
    vf_to_amb = np.zeros(Nt)
    dtheta = (pi/2)/(Nt-1)
    for t in range(Nt):  
        theta = t * dtheta
        if (theta < pi/2 - 0.001):
            vf_to_amb[t] = 1.0 - vf_differential_to_infinite_length(theta, 45*pi/180, 90*pi/180, n = 200)  
        else:
            vf_to_amb[t] = 0.0
    return vf_to_amb







#===================================================================================================================    
def test_differential_elem(alpha):
    
    import timeit
    import matplotlib.pyplot as plt
  
    delta_angle = 2*pi/180
    delta_eta = 0.1
    nangle = int(pi/2/delta_angle)
    neta = 20
    

    print ('Calculating view factor to infinite height and full adjacent tube')
    start = timeit.default_timer()
    vf_full_inf = vf_differential_to_infinite_length(alpha, 45*pi/180, 90*pi/180, n = 100)
    print ('Elapsed time = %.1es' % (timeit.default_timer() - start))
    
    
    print ('Calculating view factors to infinite height and evenly spaced angular positions')
    start = timeit.default_timer()
    vf_inf = np.zeros(nangle)
    for i in range(nangle):
        beta = (i+0.5)*delta_angle       
        vf_inf[i] = vf_differential_to_infinite_length(alpha, beta, delta_angle, n = 3)
    print ('Elapsed time = %.1es' % (timeit.default_timer() - start))
    
    
    print ('Calculating view factors to evenly spaced axial positions and full adjacent tube')
    start = timeit.default_timer()
    vf_full = np.zeros(neta)
    for j in range(neta):
        eta = j*delta_eta
        vf_full[j] = vf_differential_to_finite_length(alpha, 45*pi/180, 90*pi/180, eta, delta_eta, n = 100)
    print ('Elapsed time = %.1es' % (timeit.default_timer() - start)) 
        
    
    print ('Calculating view factors to evenly spaced axial positions and evenly spaced angular positions')
    start = timeit.default_timer()
    vf = np.zeros((nangle, neta))
    for j in range(neta):
        eta = j*delta_eta
        for i in range(nangle):
            beta = (i+0.5)*delta_angle
            vf[i,j] = vf_differential_to_finite_length(alpha, beta, delta_angle, eta, delta_eta, n = 3)      
    print ('Elapsed time = %.1es' % (timeit.default_timer() - start))
    
    
    print ('Calculating view factors by ray tracing')
    start = timeit.default_timer()
    ret = ray_trace_differential_elem(alpha, rtube = 0.02, delta_angle = delta_angle , delta_eta = delta_eta, nray = 1000000)
    vf_rt = ret['vf_tube']
    vf_rt_amb = ret['vf_amb']
    print ('Elapsed time = %.2fs' % (timeit.default_timer() - start))
    

    
    print ('View factor to full infinite-length adjacent tube: Analytical = %.4f, Ray-tracing = %.4f' %(vf_full_inf, vf_rt.sum()))
    print ('View factor to ambient: Analytical = %.4f, Ray-tracing = %.4f' % (1.-vf_full_inf, vf_rt_amb) )
    

    eta_pts = np.arange(neta, dtype = float) * delta_eta
    beta_pts = np.arange(nangle, dtype = float)*delta_angle + 0.5*delta_angle
    

    # Plots vs beta for infinite height
    plt.figure()
    plt.plot(beta_pts*180/pi, vf_inf, label = 'Analytical')
    plt.plot(beta_pts*180/pi, vf_rt[:,0] + 2*vf_rt[:,1:].sum(1), '--', label = 'Ray tracing')
    plt.legend()
    plt.xlabel('beta (deg)')
    plt.ylabel('View factor to infinite height')
    
    
    # Plots vs height
    plt.figure()
    plt.plot(eta_pts, vf_full, label = 'Analytical')
    plt.plot(eta_pts, vf_rt.sum(0)[0:neta], '--', label = 'Ray tracing')
    plt.legend()
    plt.xlabel('height / tube radius')
    plt.ylabel('View factor to full adjacent tube')
    
    
    plt.figure()
    for j in range(10):
        plt.plot(beta_pts*180/pi, vf[:,j], label = str(j) + ', Analytical')
        plt.plot(beta_pts*180/pi, vf_rt[:,j], '--', label = str(j) + ', Ray tracing')
    #plt.legend()
    plt.xlabel('beta (deg)')
    plt.ylabel('View factor to specified axial segment')
    
    return 
    
