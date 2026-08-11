"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.
"""

# Loading modules
from __future__ import division
from mmapy import mmasub, kktcheck
import numpy as np
import time

def compute_move(outeriter, max_iter, move_min, mve_max, decay_rate=5., wiggle=0.5):
	
	envelop = np.exp(-decay_rate * outeriter / max_iter)
	oscillation = 1+wiggle * np.cos(2 * np.pi * outeriter / 10)
	scaled = envelop * oscillation
	move = move_min + (mve_max - move_min) * scaled

	return move
	
def runMMA(optimizationFunction,X0,lowerBound,
			 upperBound, fTolerance = 1e-4,gTolerance = 1e-4,
			 maxIterations = 250,minIterations=10,timeLimitSecs = 3600,move_limit = 0.2,
			 kktTol = 1e-6,verbose = False, progress_callback=None,callback=None):
	

	'''
	 Input
		optimizationFunction must return a tuple consisting of
			f0val: a scalar
			df0dx: (N,1) array
			gval: (M,1) array
			dgdx: (M,N) array, i.e., the Jacobian
		X0: (N,1) array
		lowerBound: (N,1) array
		upperBound: (N,1) array
		useGCMMAsub: If True, use gccmmasub else use mmasub
		maxIterations (optional): int
		kktTol (optional):  float
		verbos (optional): Boolean to print
	'''

	def log_message(msg):
		if progress_callback:
			progress_callback(str(msg))
		else:
			print(msg)  
	# Set numpy print options
	np.set_printoptions(precision=4, formatter={'float': '{:0.4f}'.format})
	
	
	_, _, gval, _ = optimizationFunction(X0)
	
	nVariables = len(X0)
	nConstraints = gval.shape[0]
	
	# Initial settings
	n = nVariables #Number of variables
	m = nConstraints #Number of constraints
	xval = X0 # initial values
	eeem = np.ones((m, 1)) # a convenience array
	zerom = np.zeros((m, 1)) # a convenience array
	xmin = lowerBound #lower bound
	xmax = upperBound #upper bound
	maxoutit = maxIterations #  maximum iterations
	kkttol = kktTol
	 
	# Other arrays and params
	low = xmin.copy()
	upp = xmax.copy()
	
	xold1 = xval.copy() 
	xold2 = xval.copy()
	move = move_limit
	c = 100 * eeem
	d = eeem.copy()
	a0 = 1
	a = zerom.copy()
	outeriter = 0
	f0val, df0dx, gval, dgdx = optimizationFunction(xval)
	# The iterations start
	kktnorm = kkttol + 10
	outit = 0
	timeMMA = 0.0
	timeFuncEval = 0.0
	f0Scaling = f0val if abs(f0val) >1e-6 else 1
	f0valPrev = f0val/f0Scaling
	fErr = 1
	gErr = 1
	tStart = time.perf_counter()
	for i in range(maxoutit):
		# Check outer loop convergence: only apply KKT and max iteration checks after min iterations
		if outit >= minIterations:
			if kktnorm <= kkttol or outit >= maxoutit:
				break
		
		# move = compute_move(outit, maxoutit, 1e-1, 0.4, decay_rate=5., wiggle=0.5)
		
		outit += 1
		outeriter += 1
		startTime = time.perf_counter()
		xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
				m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, gval, dgdx, low, upp, a0, a, c, d, move)
		timeMMA += time.perf_counter() - startTime
		# Some vectors are updated:
		xold2 = xold1.copy()
		xold1 = xval.copy()
		xval = xmma.copy()
		
		# Re-calculate function values and gradients of the objective and constraints functions
		startTime = time.perf_counter()
		f0val, df0dx, gval, dgdx = optimizationFunction(xval)
		f0val = f0val / f0Scaling
		df0dx = df0dx / f0Scaling # scale the gradient of the objective
		timeFuncEval += time.perf_counter() - startTime
		
		if callback is not None:
			callback(outit, xval, f0val, gval)
		
		# The residual vector of the KKT conditions is calculated
		startTime = time.perf_counter()
		_, kktnorm, _ = kktcheck(
			m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, gval, dgdx, a0, a, c, d)
		timeMMA += time.perf_counter() - startTime
		fErr = float(np.abs(f0val - f0valPrev) / (1e-10 + np.abs(f0val)))
		gErr = float(np.max(gval))
		if(verbose):
			log_message('iter: {}, f: {:.3e}, max(g): {:.3e}, fErr: {:.3e}'.format(
				outeriter, f0val if np.isscalar(f0val) else float(f0val), 
				gErr, fErr))
		
		# Early convergence check: only allowed after min iterations
		if outit >= minIterations and (fErr < fTolerance and gErr < gTolerance):
			if(verbose):
				log_message(f'Convergence reached with fErr: {fErr:.3e} and max(gval): {gErr:.3e}')
			break
	
	
		f0valPrev = f0val
		if (time.perf_counter() - tStart > timeLimitSecs):
			log_message(f"Time limit of {timeLimitSecs:0.2f} reached, exiting...")
			break	
		if np.isnan(kktnorm):
			log_message("kktnorm is nan, something wrong in the optimization function, exiting...")
			break
			
	if(verbose):
		log_message(f"MMA (secs):  {timeMMA:0.2f}")
		log_message(f"FuncEval (secs): {timeFuncEval:0.2f}")

	return [xval,f0val*f0Scaling, df0dx*f0Scaling, gval, dgdx, outit, timeMMA]