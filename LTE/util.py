from math import pi, cos
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
from pvlib.location import Location
from pvlib import solarposition



#=============================================================================
# Calculate wind velocity at receiver midpoint
def calculate_wind_velocity(vwind10, Htower, Hhel):  
    Htot = Htower + 0.5*Hhel
    vwind = vwind10 * (Htot/ 10.)**0.14
    return vwind


#=============================================================================
# Calculate dew point from dry bulb T (K) and relative humidity (%)
def calculate_dew_point(Tdry, relative_humidity):  
    b = 17.271
    c = 237.7
    rh = np.minimum(np.maximum(relative_humidity, 0.01), 100) / 100
    tdryC = Tdry - 273.15
    gamma = b*tdryC / (c+tdryC) + np.log(rh)
    tdew = c*gamma / (b-gamma)   # Magnus formula en.wikipedia.org/wiki/Dew_point
    #tdew = ((1./(tdry+273.15) - (1./5423.)*np.log(rh))**-1)-273.15  
    return tdew + 273.15

# Calculate relative humidity from dry bulb T (K) and dew point T (K)
def calculcate_rh(Tdry, Tdew):  
    b = 17.271
    c = 237.7    
    TdryC = Tdry - 273.15
    TdewC = Tdew - 273.15
    rh = 100 * np.exp(b*TdewC/(c+TdewC) - b*TdryC/(c+TdryC))
    return rh

# Calculate sky temperature from dry bulb T (K), relative humidity (%) and time of day relative to solar noon
def calculate_sky_temperature(Tamb, relative_humidity, noon_offset = 0.0):
    Tdp = calculate_dew_point(Tamb, relative_humidity)
    TdpC = Tdp - 273.15
    time = (12.0 + noon_offset) * 15 * pi/180.
    Tsky = Tamb * (0.711 + 0.0056*TdpC + 0.000073*(TdpC**2)+0.013*cos(time))**0.25        
    return Tsky       

#==============================================================================
# Functions to read weather data from file

# Get location information from weather file header
def read_weatherfile_header(file):
    header = np.genfromtxt(file, delimiter = ',', dtype = 'str', skip_header = 0, max_rows = 2)
    i = np.where(header[0,:] == 'Latitude')[0][0]
    lat = float(header[1,i])
    i = np.where(header[0,:] == 'Longitude')[0][0]
    lon = float(header[1,i])
    i = np.where(header[0,:] == 'Time Zone')[0][0]
    tz = float(header[1,i])
    i = np.where(header[0,:] == 'Elevation')[0][0]
    elev = float(header[1,i])    
    return {'lat':lat, 'lon':lon, 'tz':tz, 'elevation':elev}
 
   
def read_weather(file, rows = None):
    data_cols = np.genfromtxt(file, delimiter = ',', dtype = 'str', skip_header = 2, max_rows = 1)
    if rows is None:
        weatherdata = np.genfromtxt(file, delimiter = ',', skip_header = 3)    
    else:
        weatherdata = np.genfromtxt(file, delimiter = ',', skip_header = 3+rows[0], max_rows = rows[-1]-rows[0]+1)

    colnames = {'Year':['Year'], 
                'Month':['Month'], 
                'Day':['Day'],
                'DNI':['DNI'],
                'Tdry':['Temperature', 'Tdry'],
                'Tdew':['Dew Point', 'Tdew'],
                'Pres':['Pressure', 'Pres'],
                'RH':['RH'],
                'wspd': ['Wspd']}
    
    weather = {}    
    for k, names in colnames.items():
        for name in names:
            i = np.where(data_cols == name)[0]
            if len(i)>0:
                weather[k] = weatherdata[:,i[0]]
                break
    
    if rows is not None and len(rows) == 1:
        weather = {k:weather[k][0] for k in weather.keys()}

    return weather







#==============================================================================
#--- Functions to calculate clear-sky DNI and solar position

# Return datetime object (in UTC) at current day of year (doy) and either hour of day, or hour offset from solar noon
def get_datetime(site, doy, hour_of_day = None, hour_offset = None):
    start_of_day = datetime.datetime(2019, 1, 1) + datetime.timedelta(days = int(doy))
    if hour_of_day is not None:
        time = start_of_day + datetime.timedelta(milliseconds = hour_of_day*3600*1000)
    elif hour_offset is not None:
        solar_noon = get_hour_at_solar_noon(site, doy)
        time = start_of_day + datetime.timedelta(milliseconds = solar_noon*3600*1000) + datetime.timedelta(milliseconds = hour_offset*3600*1000)
    return time


def get_datetimes(site, doy, hour_of_day = None, hour_offset = None, step_minutes = 60, nsteps = 0, is_midpoint = False):
    start = get_datetime(site, doy, hour_of_day, hour_offset)
    end = start+datetime.timedelta(milliseconds = nsteps*step_minutes*60*1000)
    times = pd.date_range(start, end, freq = '%dms'%(step_minutes*60*1000)) 
    if is_midpoint and nsteps > 0:  # Times at center of step interval
        times = times.shift(0.5*step_minutes*60, 'S')    
    return times


def calculate_solar_position(site, doy, hour_of_day = None, hour_offset = None, step_minutes = 60, nsteps = 0, is_midpoint = False):
    times = get_datetimes(site, doy, hour_of_day, hour_offset, step_minutes, nsteps, is_midpoint)
    timesutc = times - datetime.timedelta(hours = site['tz'])  
    solpos = solarposition.get_solarposition(timesutc, site['lat'], site['lon'], altitude=site['elevation'])   # UTC is assumed for unlocalized datetime objects
    zeniths = solpos['zenith'].values      # 'apparent_zenith' to account for atmospheric refraction
    azimuths = solpos['azimuth'].values
    if nsteps == 0:
        zeniths = zeniths[0]
        azimuths = azimuths[0]
    return zeniths, azimuths

def calculate_clearsky_DNI(site, doy, hour_of_day = None, hour_offset = None, step_minutes = 60, nsteps = 0, is_midpoint = False):
    times = get_datetimes(site, doy, hour_of_day, hour_offset, step_minutes, nsteps, is_midpoint)
    timesutc = times - datetime.timedelta(hours = site['tz'])  
    loc = Location(site['lat'], site['lon'], altitude=site['elevation'], tz='UTC')
    csky = loc.get_clearsky(timesutc, model = 'ineichen')['dni'].values
    if nsteps == 0:
        csky = csky[0]
    return csky


def get_hour_at_solar_noon(site, doy):
    hour_start = 10
    window = 8*60  # minutes
    interval = 30   # minutes
    while interval*60 > 1:
        nsteps = int(window / interval)
        zen, az = calculate_solar_position(site, doy, hour_start, step_minutes = interval, nsteps = nsteps)
        i = np.where(az<180)[0][-1]  
        hour_start += i*interval/60
        window = interval
        interval/=10
    return hour_start

def get_offset(site, doy, hour_of_day):
    hour_at_solar_noon = get_hour_at_solar_noon(site, doy)
    return (hour_of_day - hour_at_solar_noon)

def get_hour_of_day(site, doy, hour_offset):
    hour_at_solar_noon = get_hour_at_solar_noon(site, doy)
    return hour_at_solar_noon + hour_offset    




#=============================================================================
#TODO: update to numpy interpolation?
# Function for 1D interpolation
# x = desired value of indepedent variable
# xvals = 1D array of independent variable points (must be either monotonically increasing or monotonically decreasing)
# yvals = ND-array of dependent varaible values (can be any dimensions as long as 1st dimension matches length of xvals)
# is_extrapolate = Allow extrapolation beyond 1st/last xvals?
def interpolate1D(x, xvals, yvals, is_extrapolate = True, return_indicies = False):

    nx = len(xvals)
    if isinstance(xvals, list):
        xvals = np.array(xvals)
    if isinstance(yvals, list):
        yvals = np.array(yvals)
    
    #--- Find interpolation points
    j = abs(xvals-x).argmin()    # Points in array of x-values that is closest to desired point x
    if j == 0:          # Closest point is first point in data array
        pts = [0, 1]
    elif j == nx-1:      # Closest point is last point in data array
        pts = [j, j-1]
    elif np.sign(xvals[j+1]-xvals[j]) == np.sign(x-xvals[j]):
        pts = [j, j+1]
    else:
        pts = [j, j-1]

    if not is_extrapolate:
        if pts[0] == 0 and np.sign(xvals[1]-xvals[0]) != np.sign(x-xvals[0]):
            pts = [0,0]
        elif pts[1] == nx-1 and np.sign(xvals[-2]-xvals[-1]) != np.sign(x-xvals[-1]):
            pts[1] = [nx-1, nx-1]
    
    if return_indicies:
        return pts
        
    
    #--- Flatten y points into 2D array
    yshape = [v for v in yvals.shape]    
    ny = int(np.prod(yshape[1:]))
    if ny == 1:
        ypts = np.reshape(yvals, (nx,1))
    else:
        yflat = yvals.flatten()
        ypts = np.zeros((nx, ny))
        for j in range(nx):
            ypts[j,:] = yflat[j*ny:(j+1)*ny]

    interp = np.zeros(ny) 
    for i in range(ny):
        if pts[0] == pts[1]:
            interp[i] = ypts[pts[0],i]
        else:
            interp[i] = ypts[pts[0],i] + (ypts[pts[1],i]-ypts[pts[0],i])/(xvals[pts[1]]-xvals[pts[0]]) * (x-xvals[pts[0]])
        

    interp = np.reshape(interp, tuple(yshape[1:]))
    return interp


#=============================================================================
# Radial basis function interpolation: x = ND array of independent variable values (# points, # dimensions), y = 1D array of dependent variable values (# points)
def radial_basis_function_params(x, y, rbf = lambda r: np.exp(-(r**2))):
    npts, nD = x.shape
    A = np.zeros((npts, npts))
    for i in range(npts):
        dist = (((x[i,:] - x)**2).sum(1))**0.5
        A[i,:] = rbf(dist)
    params = np.linalg.solve(A, y) 
    return params

def radial_basis_function_interp(xpt, x, y = None, params = None, rbf = lambda r: np.exp(-(r**2))):
    if params is None:
        params = radial_basis_function_params(x,y,rbf)
    dist = (((xpt - x)**2).sum(1))**0.5
    ypt = (rbf(dist) * params).sum()
    return ypt


# Copied from SAM code...


def gauss_markov_calc_alpha(x, y, beta = 2.0, nug = 0.0):
    npts, nD = x.shape
    N = 0 
    D = 0
    for i in range(npts):
        distsqr = (((x[i,:] - x)**2).sum(1))
        D += (distsqr**(0.5*beta)).sum()
        N += ((distsqr**(0.5*beta)) * (0.5*(y[i]-y)**2 - nug**2)).sum()
    alpha = N/D
    return alpha

def gauss_markov_interp_params(x, y, beta = 2.0, nug = 0.0):
    npts, nD = x.shape
    alpha = gauss_markov_calc_alpha(x,y,beta,nug)
    A = np.ones((npts+1, npts+1))
    b = np.zeros((npts+1))
    b[0:npts] = y
    for i in range(npts):
        dist = (((x[i,:] - x)**2).sum(1))**0.5
        A[i,0:-1] = nug**2 + alpha * (dist**beta)
    A[-1,-1] = 0
    b[-1] = 1
    params = np.linalg.solve(A, b) 
    return params

def gauss_markov_interp(xpt, x, y, params = None, beta = 2.0, nug = 0.0):
    if params is None:
         params, alpha = gauss_markov_interp_params(x, y, beta, nug)
    alpha = gauss_markov_calc_alpha(x,y,beta,nug)
    dist = (((xpt - x)**2).sum(1))**0.5
    vpt = nug**2 + alpha * (dist**beta)
    vpt = np.append(vpt, 1)
    ypt = (params*vpt).sum()
    return ypt

#------ Jacob's addition
def get_month_and_day(doy):
    """
    returns the month and day of month based on a day of the year
    """
    JANUARY = 1
    YEAR = 2025
    date = datetime.datetime(YEAR, JANUARY, 1) + timedelta(doy - 1)
    M = date.month
    D = date.day
    return M, D

    
    
    

    
    
    
    