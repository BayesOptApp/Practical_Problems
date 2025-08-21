import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union, Optional

# Constants

# Radii of the planets in km
RPL = [2440,  # Mercury
        6052,  # Venus
        6378,  # Earth
        3397,  # Mars
        71492,  # Jupiter
        60330  # Saturn
        ]

# List of Gravitational parameters and radii for the planets
MU = [22321, # Mercury
        324860, # Venus
            398601.19, # Earth
            42828.3, # Mars
                126.7e6, # Jupiter
                37.93951970883e6 # Saturn
                ]
    
# Gravitational constant of the Sun
MU_SUN = 1.32712428e+11

RAD = np.pi / 180.0
KM = 1.495978706910000e+008
AU = 149597870.66 # km


### ==================
# Spacecraft Trajectory Optimization ==================== ###
# This function is a Python translation of the MATLAB code provided.
def get_xl(_):
    r"""Lower bounds for the design variables."""
    return np.array([
        1900, 2.5, 0, 0,
        *[100]*6,
        *[0.01]*6,
        1.1, 1.1, 1.05, 1.05, 1.05,
        *[-np.pi]*5
    ]).ravel()

def get_xu(_):
    r"""Upper bounds for the design variables."""
    return np.array([
        2300, 4.05, 1, 1,
        *[500]*5, 600,
        *[0.99]*6,
        6, 6, 6, 6, 6,
        *[np.pi]*5
    ]).ravel()

def Spacecraft_Trajectory_OptimizationC1(x):
    r"""Objective function for spacecraft trajectory optimization."""

    # Check input dimensions and size
    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size >= 3, "Input array must have at least 3 elements."

    # Perform a reshape of the initial array
    x = x.ravel()

    n = x.size
    if n < 26:
        raise ValueError("Input array must have at least 25 elements.")
    lb = get_xl(n)
    ub = get_xu(n)
    x = np.abs(ub - lb) * x + lb

    # Compute the function
    y = mga_dsm(x)
    return y if not np.isnan(y) else 1e100

def mga_dsm(t):
    sequence = [3,2,2,1,1,1,1]
    problem = {
        'objective': {'rp': 2640, 'e': 0.704}
    }
    rp_target = problem['objective']['rp']
    e_target = problem['objective']['e']


    # Radii of the planets in km (for the Sun)
    # tdep -> % Departure time (days from the initial epoch)
    # VINF -> % Hyperbolic excess velocity (km/s)
    # udir -> % Departure direction (unit vector)
    # vdir -> % Arrival direction (unit vector)
    tdep, VINF, udir, vdir = t[0:4]


    nn = len(sequence) # Number of planets in the mission sequence

    # Initializing the variables
    tof = np.zeros((nn - 1, )) # Time of flight between planets (days)
    alpha = np.zeros((nn - 1, )) # Fraction of time of flight at which DSM occurs
    rp_non_dim = np.zeros((nn - 2, )) # Non-dimensional periapsis radius for planets P2..Pn-1
    gamma = np.zeros((nn - 2, )) # Rotation of the b-plane-component of the swingby outgoing velocity vector

    # Extract time of flight and DSM fraction from the decision vector
    for i in range(nn - 1):
        tof[i] = t[i+4]; # Planet-to-planet Time of Flight (ToF) (days)
        alpha[i] = t[nn+i+3]; # Fraction of ToF at which the DSM occurs
    
    for i in range(0,nn-2):
        rp_non_dim[i] = t[i+2+2*nn]
        gamma[i] = t[3*nn+i]; # Rotation of the b-plane-component of the swingby outgoing velocity vector

    # Calculate positions, velocities, and gravitational constants for the solar system bodies
    rr = np.zeros((3, nn)) # Position vectors of the planets (km)
    vv = np.zeros((3, nn)) # Velocity vectors of the planets (km/s)
    mu_vec = np.zeros((nn, )) # Gravitational parameters of the planets (km^3/s^2)
    iTime = np.zeros((nn, )) # Time of flight (days)
    dt = np.zeros((nn, )) # Time of flight (seconds)
    seq = np.abs(sequence).tolist() # Sequence of planets (1-indexed)
    tt = deepcopy(tdep) # Departure time (days from the initial epoch)
    dt[0:nn-1] = tof

    #rp = np.array([rp_non_dim[i] * RPL[sequence[i+1] - 1] for i in range(nn-2)])
    for i in range(nn):
        iTime[i] = tt
        if seq[i] < 10:
            rr[:,i], vv[:,i], _ = pleph_an(tt, seq[i]) # Positions and velocities of solar system planets
            mu_vec[i] = MU[seq[i]-1] # Gravitational constants
        else:
            raise NotImplementedError("Planet index out of range.")

        tt += dt[i]
    
    # Calculate flyby radii
    rp = np.zeros((nn-2, 1))
    for i  in range(nn-2):
        rp[i] = rp_non_dim[i] * RPL[seq[i+1]-1]; # Dimensional flyby radii (i=1 corresponds to the second planet)
    
    vtemp= np.cross(rr[:,0],vv[:,0])
    iP1= vv[:,0]/np.linalg.norm(vv[:,0])
    zP1= vtemp/np.linalg.norm(vtemp)
    jP1= np.cross(zP1,iP1)

    theta=2*np.pi*udir #See Picking a Point on a Sphere
    phi=np.arccos(2*vdir-1)-np.pi/2; # In this way: -pi/2<phi<pi/2 so phi can be used as out-of-plane rotation
    vinf=VINF*(np.cos(theta)*np.cos(phi)*iP1+np.sin(theta)*np.cos(phi)*jP1+np.sin(phi)*zP1)

    v_sc_pl_in = np.zeros_like(vv) # Spacecraft absolute incoming velocity at P1 (km/s)
    v_sc_pl_out = np.zeros_like(vv) # Spacecraft absolute outgoing velocity at P1 (km/s)
    v_sc_dsm_in = np.zeros_like(vv) # Spacecraft absolute incoming velocity at P1 (km/s)
    v_sc_dsm_out = np.zeros_like(vv) # Spacecraft absolute outgoing velocity at P1 (km/s)

    v_sc_pl_in[:,0]=vv[:,0] # Spacecraft absolute incoming velocity at P1
    v_sc_pl_out[:,0]=vv[:,0]+vinf # Spacecraft absolute outgoing velocity at P1
    
    tDSM = np.zeros((nn-1, )) # Time of flight at which the DSM occurs (days)
    rd = np.zeros_like(rr) # Position vectors of the planets (km)

    tDSM[0]=alpha[0]*tof[0]


    rd[:,0],v_sc_dsm_in[:,0] =propagateKEP(rr[:,0],v_sc_pl_out[:,0],tDSM[0]*24*60*60,MU_SUN)
    lw=vett(rd[:,0],rr[:,1])
    lw=np.sign(lw[2])
    if lw==1:
        lw=0
    else:
        lw=1
    
    v_sc_dsm_out[:,0] , v_sc_pl_in[:,1],_,_,_,_=lambertI(rd[:,0],
                                                 rr[:,1],
                                                 tof[0]*(1-alpha[0])*24*60*60,
                                                 MU_SUN,
                                                 lw)
    
    DV=np.zeros((nn,1)) # Delta-V at each planet (km/s);
    DV[0]=np.linalg.norm(v_sc_dsm_out[:,0]-v_sc_dsm_in[:,0])
    tDSM=np.zeros((nn-1,1))
    for i in range(nn-2):
        v_rel_in=v_sc_pl_in[:,i+1]-vv[:,i+1]

        e=1+rp[i]/mu_vec[i+1]*np.dot(v_rel_in.T,v_rel_in).ravel(); ##ok<*MHERM>
        beta_rot=2*np.arcsin(1/e);            # velocity rotation

        ix=v_rel_in/np.linalg.norm(v_rel_in)
        iy=vett(ix,vv[:,i+1]/np.linalg.norm(vv[:,i+1])).T

        iy=iy/np.linalg.norm(iy)
        iz=vett(ix,iy).T

        iVout = np.cos(beta_rot) * ix + np.cos(gamma[i])*np.sin(beta_rot) * iy + np.sin(gamma[i])*np.sin(beta_rot) * iz
        v_rel_out=np.linalg.norm(v_rel_in)*iVout
        v_sc_pl_out[:,i+1]=vv[:,i+1]+v_rel_out

        tDSM[i+1]=alpha[i+1]*tof[i+1]
        
        rd[:,i+1],v_sc_dsm_in[:,i+1]=propagateKEP(rr[:,i+1],v_sc_pl_out[:,i+1],tDSM[i+1]*24*60*60,MU_SUN)
        lw=vett(rd[:,i+1],rr[:,i+2])
        lw=np.sign(lw[2])
        if lw==1:
            lw=0
        else:
            lw=1
        v_sc_dsm_out[:,i+1],v_sc_pl_in[:,i+2],_,_,_,_=lambertI(rd[:,i+1],rr[:,i+2],tof[i+1]*(1-alpha[i+1])*24*60*60,MU_SUN,lw)
        DV[i+1]=np.linalg.norm(v_sc_dsm_out[:,i+1]-v_sc_dsm_in[:,i+1])
    

    DVrel=np.linalg.norm(vv[:,nn-1]-v_sc_pl_in[:,nn-1]); # Relative velocity at target planet
    DVper=np.sqrt(DVrel**2+2*mu_vec[nn-1]/rp_target);  # Hyperbola
    DVper2=np.sqrt(2*mu_vec[nn-1]/rp_target-mu_vec[nn-1]/rp_target*(1-e_target)); # Ellipse
    DVarr=np.abs(DVper-DVper2)
    DV[nn-1]=DVarr
    DVtot=np.sum(DV)
    J = DVtot
    
    return J

# The remaining utility functions (IC2par, M2E, par2IC, pleph_an, vett, lambertI, etc.)
# need to be implemented here following the MATLAB logic.

def propagateKEP(r0, v0, t, mu)->Tuple[np.ndarray, np.ndarray]: 

    DD = np.eye(3)
    h = vett(r0, v0); # Specific angular momentum vector
    
    ih = h / np.linalg.norm(h); #Normalized specific angular momentum vector
    
    # Check if the orbit is retrograde (ih = [0, 0, -1]) with a small tolerance.
    if np.abs(np.abs(ih[2]) - 1) < 1e-3:
        # Random rotation matrix to make the Euler angles well defined for retrograde orbits
        DD:np.ndarray = np.asarray([[1.0, 0.0, 0.0],
                                    [ 0.0, 0.0, 1.0],
                                    [ 0.0, -1.0, 0.0]])
        # Rotate the initial position and velocity vectors
        r0 = np.dot(DD , r0)
        v0 = np.dot(DD , v0)
    
    
    # Convert initial conditions to orbital elements
    E = IC2par(r0, v0, mu)
    
    # Calculate the mean anomaly at time t (M)
    M0 = E2M(E[5].ravel(), E[1].ravel())

    if E[1] < 1:
        M = M0 + np.sqrt(mu / E[0]**3) * t
    else:
        M = M0 + np.sqrt(-mu / E[0]**3) * t
    
    
    # Solve for eccentric anomaly (E) at time t
    E[5] = M2E(M, E[1])
    
    # Convert the updated orbital elements back to initial conditions (r0, v0)
    r, v = par2IC(E, mu)
    
    # Rotate the position and velocity vectors back to the original frame
    r = np.dot(DD.T, r)
    v = np.dot(DD.T , v)

    return r.ravel(),v.ravel()

def IC2par(r0:np.ndarray, 
           v0:np.ndarray, 
           mu:float):
    r"""Convert initial position and velocity vectors to orbital elements"""

    from math import pi

    ### TODO: Check the transpose definition of all the vectors in the original MATLAB code.
    
    k = np.asarray([0, 0, 1]).reshape((3,1)) # Unit vector in the z-direction (assuming Earth-centered inertial frame)
    h = (vett(r0, v0).T).flatten() # Specific angular momentum vector
    p = np.dot(h , h.T).ravel()[0] / mu; #  Semi-latus rectum
    
    n = vett(k, h).T # Unit vector in the ascending node direction
    n:np.ndarray = (n / np.linalg.norm(n)).flatten() # Normalize the vector
    
    R0 = np.linalg.norm(r0) # Magnitude of initial position vector
    evett = (vett(v0, h).T/ mu - r0 / R0).flatten() # Eccentricity vector
    e = np.dot(evett.T,evett).flatten()[0] #Eccentricity
    
    E = np.zeros((6, 1)) # Initialize the orbital elements array

    E[0] = p / (1 - e) # Semi-major axis
    E[1] = np.sqrt(e) # Eccentricity
    e = E[1,0] # Eccentricity
    
    E[2] = np.arccos(h[2] / np.linalg.norm(h)) # Inclination
    
    # Argument of periapsis (omega)
    E[4] = (np.arccos(np.dot(n.flatten(), evett) / e))
    
    # Check for quadrant and adjust the angle if necessary
    if evett[2] < 0:
        E[4] = 2 * np.pi - E[4]
    
    # Right ascension of the ascending node (Omega)
    E[3] = np.arccos(n[0])
    if n[1] < 0:
        E[3] = 2 * np.pi - E[3]
    
    # True anomaly (theta)
    ni = np.real(np.arccos(np.dot(evett.T, r0) / e / R0)); # Real is to avoid problems when ni~=pi
    if np.dot(r0, v0).flatten()[0] < 0:
        ni = 2 * np.pi - ni
    
    # Solve for eccentric anomaly (E) from true anomaly (theta)
    E[5] = ni2E(ni, e)

    return E

def ni2E(ni:float, e:float):
    r"""Convert true anomaly 
    (theta) to eccentric anomaly (E) using algebraic Kepler's equation for ellipse or hyperbolic equivalent.
    

    Args
    ---------------
    - ni : `float`: True anomaly in radians.
    - e : `float`: Eccentricity.

    Returns
    ---------------
    - E : `float`: Eccentric anomaly in radians.
    """
    

    if e < 1:
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(ni / 2)); # Algebraic Kepler's equation for ellipse
    else:
        E = 2 * np.arctan(np.sqrt((e - 1) / (e + 1)) * np.tan(ni / 2)); # Algebraic equivalent of Kepler's equation in terms of the Gudermannian for hyperbola

    return E

def par2IC(E:np.ndarray, mu:float):
    r"""
    
    Converts orbital elements (E) to initial position (r0) and velocity (v0) vectors.
    

    Args
    ---------------
    - E : `numpy.ndarray`: Orbital elements (a, e, i, omega, omega_p, EA).
    - mu : `float`: Gravitational parameter.

    Returns
    ---------------
    - r0 : `numpy.ndarray`: Initial position vector.
    - v0 : `numpy.ndarray`: Initial velocity vector.
    """
    
    from math import sqrt, cos, sin, tan, pi

    E = E.ravel().tolist() # Convert to list for easier indexing
    a = E[0]
    e = E[1]
    i = E[2]
    omg = E[3]
    omp = E[4]
    EA = E[5]
    
    # If the orbit is an ellipse
    if e < 1:
        b = a * np.sqrt(1 - e**2)
        n = np.sqrt(mu / a**3)
    
        xper = a * (np.cos(EA) - e)
        yper = b * np.sin(EA)
    
        xdotper = -(a * n * np.sin(EA)) / (1 - e * np.cos(EA))
        ydotper = (b * n * np.cos(EA)) / (1 - e * np.cos(EA))
    else: #If the orbit is a hyperbola
        b = -a * np.sqrt(e**2 - 1)
        n = np.sqrt(-mu / a**3)
        
        # Calculate the denominator for the hyperbolic case
        dNdzeta = e * (1 + np.tan(EA)**2) - (1 / 2 + 1 / 2 * np.tan(1 / 2 * EA + 1 / 4 * np.pi)**2) / np.tan(1 / 2 * EA + 1 / 4 * pi)
        
        xper = a / np.cos(EA) - a * e
        yper = b * np.tan(EA)
        
        xdotper = a * np.tan(EA) / np.cos(EA) * n / dNdzeta
        ydotper = b / np.cos(EA)**2 * n / dNdzeta
    
    
    # Rotation matrix for converting orbital elements to initial conditions
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(omg) * np.cos(omp) - np.sin(omg) * np.sin(omp) * np.cos(i)
    R[0, 1] = -np.cos(omg) * np.sin(omp) - np.sin(omg) * np.cos(omp) * np.cos(i)
    R[0, 2] = np.sin(omg) * np.sin(i)
    R[1, 0] = np.sin(omg) * np.cos(omp) + np.cos(omg) * np.sin(omp) * np.cos(i)
    R[1, 1] = -np.sin(omg) * np.sin(omp) + np.cos(omg) * np.cos(omp) * np.cos(i)
    R[1, 2] = -np.cos(omg) * np.sin(i)
    R[2, 0] = np.sin(omp) * np.sin(i)
    R[2, 1] = np.cos(omp) * np.sin(i)
    R[2, 2] = np.cos(i)
    
    r0 = R @ np.array([xper, yper, 0]).reshape((3, 1)) # Position vector
    v0 = R @ np.array([xdotper, ydotper, 0]).reshape((3, 1)) # Velocity vector

    return r0, v0


def lambertI(r1:np.ndarray, 
             r2:np.ndarray, t:float, mu:float, lw:int, 
             N:Optional[int]=0, 
             branch:Optional[str]="l"):
    from warnings import warn


    #### TODO: Ckeck the transpose definition of all the vectors in the original MATLAB code.

    # r"""Check if the number of input arguments is 5 (N is not provided) and set N to 0 if true."""
    # if nargin == 5:
    #     N = 0
    
    # Check if time t is negative and print a warning message if it is.
    if t <= 0:
        warn('Negative time as input')
        v1 = np.nan* np.ones((3,1))
        v2 = v1.copy()
        return v1, v2
    
    # Set the tolerance for Newton's iterations.
    tol = 1e-11
    
    # Calculate the normalized radius and velocity magnitude.
    R = np.linalg.norm(r1)
    V = np.sqrt(mu / R)
    T = R / V
    
    # Working with non-dimensional radii and time-of-flight.
    r1 = r1 / R
    r2 = r2 / R
    tn = t / T
    
    # Evaluation of the relevant geometry parameters in non-dimensional units.
    r2mod = np.linalg.norm(r2)
    theta = np.arccos(np.dot(r1.T, r2).flatten()[0] / r2mod)
    
    # If lw is true, set theta to its complementary angle.
    if lw:
        theta = 2 * np.pi - theta
    
    c = np.sqrt(1 + r2mod**2 - 2 * r2mod * np.cos(theta)) # Non-dimensional chord
    s = (1 + r2mod + c) / 2 #  Non-dimensional semi-perimeter
    am = s / 2#  Minimum energy ellipse semi-major axis
    lambdda = np.sqrt(r2mod) * np.cos(theta / 2) / s #  Lambda parameter defined in BATTIN's book
    
    # Initialize variables for Newton iterations.
    iterr = 0
    
    # For N = 0, calculate x using Newton iterations.
    if N == 0:
        inn1 = -0.5233; # First guess point
        inn2 = 0.5233; # Second guess point
        x1 = np.log(1 + inn1)
        x2 = np.log(1 + inn2)
        y1 = np.log(x2tof(inn1, s, c, lw, N)) - np.log(tn)
        y2 = np.log(x2tof(inn2, s, c, lw, N)) - np.log(tn)
        
        # Newton iterations
        err = 1
        while err > tol and  y1 != y2:
            iterr += 1
            xnew = (x1 * y2 - y1 * x2) / (y2 - y1)
            ynew = np.log(x2tof(np.exp(xnew) - 1, s, c, lw, N)) - np.log(tn)
            x1 = x2
            y1 = y2
            x2 = xnew
            y2 = ynew
            err = np.abs(x1 - xnew)
        
        x = np.exp(xnew) - 1
    else:
        # For N ~= 0, calculate x using Newton iterations for the given branch (long or short).
        if branch == 'l':
            inn1 = -0.5234
            inn2 = -0.2234
        else:
            inn1 = 0.7234
            inn2 = 0.5234

        x1 = np.tan(inn1 * np.pi / 2)
        x2 = np.tan(inn2 * np.pi / 2)
        y1 = x2tof(inn1, s, c, lw, N) - tn
        y2 = x2tof(inn2, s, c, lw, N) - tn
        
        err = 1
        while err > tol and iterr < 60 and y1 != y2:
            iterr +=  1
            xnew = (x1 * y2 - y1 * x2) / (y2 - y1)
            ynew = x2tof(np.arctan(xnew) * 2 / np.pi, s, c, lw, N) - tn
            x1 = x2
            y1 = y2
            x2 = xnew
            y2 = ynew
            err = np.abs(x1 - xnew)

        x = np.arctan(xnew) * 2 / np.pi
    
    
    a = am / (1 - x**2) # Solution semimajor axis
    
    # For ellipse, calculate beta, alfa, psi, and eta.
    if x < 1:
        beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
        if lw:
            beta = -beta
        alfa = 2 * np.arccos(x)
        psi = (alfa - beta) / 2
        eta2 = 2 * a * np.sin(psi)**2 / s
        eta = np.sqrt(eta2)
    else:
        # For hyperbola, calculate beta, alfa, psi, and eta.
        beta = 2 * np.arcsinh(np.sqrt((c - s) / (2 * a)))
        if lw:
            beta = -beta
        
        alfa = 2 * np.arccosh(x)
        psi = (alfa - beta) / 2
        eta2 = -2 * a * np.sinh(psi)**2 / s
        eta = np.sqrt(eta2)
    
    
    p = r2mod / am / eta2 * np.sin(theta / 2)**2 # Parameter of the solution
    sigma1 = 1 / eta / np.sqrt(am) * (2 * lambdda * am - (lambdda + x * eta))
    ih = vers(vett(r1, r2).T) # Normalized initial unit vector
    
    if lw:
        ih = -ih
    
    # Calculate velocity components.
    vr1 = sigma1
    vt1 = np.sqrt(p)
    v1 = (vr1 * r1 + vt1 * vett(ih, r1).T).flatten()
    
    vt2 = vt1 / r2mod
    vr2 = -vr1 + (vt1 - vt2) / np.tan(theta / 2)
    v2 = (vr2 * r2 / r2mod + vt2 * vett(ih, r2 / r2mod).T).flatten()
    
    # Rescale the velocity vectors to dimensional units.
    v1 = v1 * V
    v2 = v2 * V
    a = a * R
    p = p * R

    return v1, v2, a, p, theta, iterr

def x2tof(x, s, c, lw, N):
    am = s / 2
    a = am / (1 - x**2)
    if x < 1: # Ellipse
        beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
        if lw:
            beta = -beta
        
        alfa = 2 * np.arccos(x)
    else: #Hyperbola
        alfa = 2 * np.arccosh(x)
        beta = 2 * np.arcsinh(np.sqrt((s - c) / (-2 * a)))
        if lw:
            beta = -beta

    t = tofabn(a, alfa, beta, N)

    return t

def tofabn(sigma, alfa, beta, N):
    if sigma > 0:
        t = sigma * np.sqrt(sigma) * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + N * 2 * np.pi)
    else:
        t = -sigma * np.sqrt(-sigma) * ((np.sinh(alfa) - alfa) - (np.sinh(beta) - beta))
    
    return t

def vers(V:np.ndarray)->np.ndarray:
    r"""Calculate the vector of versors."""
    V = V.ravel() # Flatten the input array
    v = V / np.sqrt(np.dot(V.T, V))

    return v

def vett(r1:np.ndarray, r2:np.ndarray)->np.ndarray:
    r"""Calculate the vector between two points."""
    ansd = np.cross(r1.ravel(), r2.ravel()) # Calculate the cross product of r1 and r2

    return ansd.reshape((3,1)) # Reshape the result to a column vector


def pleph_an(mjd2000, planet):
    # Constants
    
    # Time variables
    T = (mjd2000 + 36525.00) / 36525.00
    TT = T * T
    TTT = T * TT

    # Planetary parameters based on the input planet
    E = np.zeros((6,  )) # Initialize the array for planetary parameters
    if planet ==1:
        E[0] = 0.38709860
        E[1] = 0.205614210 + 0.000020460*T - 0.000000030*TT
        E[2] = 7.002880555555555560 + 1.86083333333333333e-3*T - 1.83333333333333333e-5*TT
        E[3] = 4.71459444444444444e+1 + 1.185208333333333330*T + 1.73888888888888889e-4*TT
        E[4] = 2.87537527777777778e+1 + 3.70280555555555556e-1*T +1.20833333333333333e-4*TT
        XM   = 1.49472515288888889e+5 + 6.38888888888888889e-6*T
        E[5] = 1.02279380555555556e2 + XM*T
    elif planet ==2:
        E[0] = 0.72333160
        E[1] = 0.006820690 - 0.000047740*T + 0.0000000910*TT
        E[2] = 3.393630555555555560 + 1.00583333333333333e-3*T - 9.72222222222222222e-7*TT
        E[3] = 7.57796472222222222e+1 + 8.9985e-1*T + 4.1e-4*TT
        E[4] = 5.43841861111111111e+1 + 5.08186111111111111e-1*T -1.38638888888888889e-3*TT
        XM   = 5.8517803875e+4 + 1.28605555555555556e-3*T
        E[5] = 2.12603219444444444e2 + XM*T
    elif planet ==3:
        E[0] = 1.000000230
        E[1] = 0.016751040 - 0.000041800*T - 0.0000001260*TT
        E[2] = 0.00
        E[3] = 0.00
        E[4] = 1.01220833333333333e+2 + 1.7191750*T + 4.52777777777777778e-4*TT + 3.33333333333333333e-6*TTT
        XM   = 3.599904975e+4 - 1.50277777777777778e-4*T - 3.33333333333333333e-6*TT
        E[5] = 3.58475844444444444e2 + XM*T
    elif planet ==4:
        E[0] = 1.5236883990
        E[1] = 0.093312900 + 0.0000920640*T - 0.0000000770*TT
        E[2] = 1.850333333333333330 - 6.75e-4*T + 1.26111111111111111e-5*TT
        E[3] = 4.87864416666666667e+1 + 7.70991666666666667e-1*T - 1.38888888888888889e-6*TT - 5.33333333333333333e-6*TTT
        E[4] = 2.85431761111111111e+2 + 1.069766666666666670*T +  1.3125e-4*TT + 4.13888888888888889e-6*TTT
        XM   = 1.91398585e+4 + 1.80805555555555556e-4*T + 1.19444444444444444e-6*TT
        E[5] = 3.19529425e2 + XM*T
    elif planet ==5:
        E[0] = 5.2025610
        E[1] = 0.048334750 + 0.000164180*T  - 0.00000046760*TT -0.00000000170*TTT
        E[2] = 1.308736111111111110 - 5.69611111111111111e-3*T +  3.88888888888888889e-6*TT
        E[3] = 9.94433861111111111e+1 + 1.010530*T + 3.52222222222222222e-4*TT - 8.51111111111111111e-6*TTT
        E[4] = 2.73277541666666667e+2 + 5.99431666666666667e-1*T + 7.0405e-4*TT + 5.07777777777777778e-6*TTT
        XM   = 3.03469202388888889e+3 - 7.21588888888888889e-4*T + 1.78444444444444444e-6*TT
        E[5] = 2.25328327777777778e2 + XM*T
    elif planet ==6:
        E[0] = 9.5547470
        E[1] = 0.055892320 - 0.00034550*T - 0.0000007280*TT + 0.000000000740*TTT
        E[2] = 2.492519444444444440 - 3.91888888888888889e-3*T - 1.54888888888888889e-5*TT + 4.44444444444444444e-8*TTT
        E[3] = 1.12790388888888889e+2 + 8.73195138888888889e-1*T -1.52180555555555556e-4*TT - 5.30555555555555556e-6*TTT
        E[4] = 3.38307772222222222e+2 + 1.085220694444444440*T + 9.78541666666666667e-4*TT + 9.91666666666666667e-6*TTT
        XM   = 1.22155146777777778e+3 - 5.01819444444444444e-4*T - 5.19444444444444444e-6*TT
        E[5] = 1.75466216666666667e2 + XM*T
    elif planet ==7:
        E[0] = 19.218140
        E[1] = 0.04634440 - 0.000026580*T + 0.0000000770*TT
        E[2] = 7.72463888888888889e-1 + 6.25277777777777778e-4*T + 3.95e-5*TT
        E[3] = 7.34770972222222222e+1 + 4.98667777777777778e-1*T + 1.31166666666666667e-3*TT
        E[4] = 9.80715527777777778e+1 + 9.85765e-1*T - 1.07447222222222222e-3*TT - 6.05555555555555556e-7*TTT
        XM   = 4.28379113055555556e+2 + 7.88444444444444444e-5*T + 1.11111111111111111e-9*TT
        E[5] = 7.26488194444444444e1 + XM*T
    elif planet ==8:
        E[0] = 30.109570
        E[1] = 0.008997040 + 0.0000063300*T - 0.0000000020*TT
        E[2] = 1.779241666666666670 - 9.54361111111111111e-3*T - 9.11111111111111111e-6*TT
        E[3] = 1.30681358333333333e+2 + 1.0989350*T + 2.49866666666666667e-4*TT - 4.71777777777777778e-6*TTT
        E[4] = 2.76045966666666667e+2 + 3.25639444444444444e-1*T + 1.4095e-4*TT + 4.11333333333333333e-6*TTT
        XM   = 2.18461339722222222e+2 - 7.03333333333333333e-5*T
        E[5] = 3.77306694444444444e1 + XM*T
    elif planet ==9:
        T=mjd2000/36525
        TT=T*T
        TTT=TT*T
        TTTT=TTT*T
        TTTTT=TTTT*T
        E[0]=39.34041961252520 + 4.33305138120726*T - 22.93749932403733*TT + 48.76336720791873*TTT - 45.52494862462379*TTTT + 15.55134951783384*TTTTT
        E[1]=0.24617365396517 + 0.09198001742190*T - 0.57262288991447*TT + 1.39163022881098*TTT - 1.46948451587683*TTTT + 0.56164158721620*TTTTT
        E[2]=17.16690003784702 - 0.49770248790479*T + 2.73751901890829*TT - 6.26973695197547*TTT + 6.36276927397430*TTTT - 2.37006911673031*TTTTT
        E[3]=110.222019291707 + 1.551579150048*T - 9.701771291171*TT + 25.730756810615*TTT - 30.140401383522*TTTT + 12.796598193159 * TTTTT
        E[4]=113.368933916592 + 9.436835192183*T - 35.762300003726*TT + 48.966118351549*TTT - 19.384576636609*TTTT - 3.362714022614 * TTTTT
        E[5]=15.17008631634665 + 137.023166578486*T + 28.362805871736*TT - 29.677368415909*TTT - 3.585159909117*TTTT + 13.406844652829 * TTTTT


    # Convert the units of planetary parameters
    E[0] = E[0] * KM
    for ii in range(2,6):
        E[ii] = np.deg2rad( E[ii] )

    #E[5] = np.remainder(E[5], 2 * np.pi)
    E[5] = np.mod(E[5], 2 * np.pi)

    # Calculate the eccentric anomaly (EccAnom) using M2E function
    EccAnom = M2E(E[5], E[1])
    E[5] = EccAnom

    # Calculate the position and velocity vectors (r and v) using the conversion function
    r, v = conversion(E)

    return r.ravel(), v.ravel(), E

def conversion(E:np.ndarray):
    r"""Convert orbital elements to position and velocity vectors.
    
    Args
    ---------------
    - E : `numpy.ndarray`: Orbital elements (a, e, i, omega, omega_p, EA).

    Returns
    ---------------
    - r : `numpy.ndarray`: Position vector.
    - v : `numpy.ndarray`: Velocity vector.
    """

    # Extract orbital elements from the input vector E
    E = E.ravel().tolist()
    a = E[0];   # Semi-major axis
    e = E[1];   # Eccentricity
    i = E[2];   # Inclination
    omg = E[3]; # Longitude of the ascending node (Omega)
    omp = E[4]; # Argument of periapsis (omega)
    EA = E[5];  # Eccentric anomaly (E)

    # Calculate parameters needed for conversion
    b = a * np.sqrt(1 - e**2); # Semi-minor axis
    n = np.sqrt(MU_SUN / a**3); # Mean motion

    # Perifocal coordinates and velocities
    xper = a * (np.cos(EA) - e)
    yper = b * np.sin(EA)

    xdotper = -(a * n * np.sin(EA)) / (1 - e * np.cos(EA))
    ydotper = (b * n * np.cos(EA)) / (1 - e * np.cos(EA))

    R = np.zeros((3, 3)) # Initialize the rotation matrix

    # Transformation matrix from perifocal to ECI (Earth-Centered Inertial) frame
    R[0,0] = np.cos(omg) * np.cos(omp) - np.sin(omg) * np.sin(omp) * np.cos(i)
    R[0,1] = -np.cos(omg) * np.sin(omp) - np.sin(omg) * np.cos(omp) * np.cos(i)
    R[0,2] = np.sin(omg) * np.sin(i)
    R[1, 0] = np.sin(omg) * np.cos(omp) + np.cos(omg) * np.sin(omp) * np.cos(i)
    R[1, 1] = -np.sin(omg) * np.sin(omp) + np.cos(omg) * np.cos(omp) * np.cos(i)
    R[1, 2] = -np.cos(omg) * np.sin(i)
    R[2, 0] = np.sin(omp) * np.sin(i)
    R[2, 1] = np.cos(omp) * np.sin(i)
    R[2, 2] = np.cos(i)

    # Convert perifocal coordinates to ECI coordinates
    r = R @ np.array([xper, yper,0]).reshape((3, 1))
    v = R @ np.array([xdotper,ydotper, 0]).reshape((3, 1))

    return r, v

def M2E(M, e):
    r"""Solve Kepler's equation to find the eccentric anomaly (E) from the mean anomaly (M)
    
    Args
    ---------------
    - M : `float`: Mean anomaly in radians.
    - e : `float`: Eccentricity.

    Returns
    ---------------
    - E : `float`: Eccentric anomaly in radians.
    """
    i = 0
    tol = 1e-10
    err = 1.0
    E = M + e * np.cos(M) # Initial guess for E

    # Iteratively improve the value of E using Newton-Raphson method
    while err > tol and i < 100:
        i += 1
        denom = (1 - e * np.cos(E))
        if denom == 0:
            break
        Enew = E - (E - e * np.sin(E) - M) / denom
        err = abs(E - Enew)
        E = Enew
    
    return E

def E2M(E:np.ndarray, e:float):
    r"""Convert eccentric anomaly (E) to mean anomaly (M)
    

    Args
    ---------------
    - E : `np.ndarray`: Eccentric anomaly in radians.
    - e : `float`: Eccentricity.

    Returns
    ---------------
    - M : `float`: Mean anomaly in radians.

    """
    if e < 1:
        # Ellipse case: E is the eccentric anomaly
        M = E - e * np.sin(E)
    else:
        # Hyperbola case: E is the Gudermannian
        M = e * np.tan(E) - np.log(np.tan(E/2 + np.pi/4))
    
    return M



def CUSTOMeph(jd, epoch, keplerian, flag):
    
    global AU #ok<GVMIS>
    
    a = keplerian(1) * AU; # Semi-major axis in km
    e = keplerian(2);      # Eccentricity
    i = keplerian(3);      # Inclination in degrees
    W = keplerian(4);      # Longitude of the ascending node in degrees
    w = keplerian(5);      # Argument of perigee in degrees
    M = keplerian(6);      # Mean anomaly in degrees

    jdepoch = mjd2jed(epoch); # Convert the epoch from MJD to Julian Ephemeris Date (JED)
    DT = (jd - jdepoch) * 60 * 60 * 24; # Time difference between JD and epoch in seconds

    n = np.sqrt(MU_SUN / a**3); # Mean motion (angular speed) in rad/s
    M = np.deg2rad(M) ;      # Convert mean anomaly to radians
    M = M + n * DT;        # Calculate the mean anomaly at the given JD
    M = np.remainder(M, 2 * np.pi);    # Wrap the mean anomaly to the range [0, 2*pi]

    E = M2E(M, e); # Calculate the eccentric anomaly from the mean anomaly

    # Convert the Keplerian orbital elements to position and velocity vectors
    r, v = par2IC([a, e, np.deg2rad(i), np.deg2rad(W), np.deg2rad(w), E], MU_SUN)

    if flag != 1:
        r = r / AU; # Convert position from km to AU
        v = v * 86400 / AU; # Convert velocity from km/s to AU/day

    return r,v 


def mjd20002jed(mjd2000):
    r"""Convert Modified Julian Date to Julian Ephemeris Date."""
    return mjd2000 + 51544.5

def mjd2jed(mjd):
    r"""Convert Modified Julian Date to Julian Ephemeris Date."""
    return mjd + 2400000.5


# Example usage:
if __name__ == "__main__":
    # Example normalized input (all values in [0,1])
    #x = np.random.rand(25)
    x = np.asarray([2100, 3.27500000000000, 0.500000000000000, 0.500000000000000,
                    300, 300, 300, 300, 300, 350, 
                    0.500000000000000, 0.500000000000000, 0.500000000000000, 0.500000000000000,
                        	0.500000000000000,	0.500000000000000,	3.55000000000000,	3.55000000000000,
                                	3.52500000000000,	3.52500000000000,	3.52500000000000,
                                        	0,	0,	0,	0,	0]).ravel()
    
    # x = np.asarray([1966.66666666667, 3.27500000000000,	0.500000000000000,	0.500000000000000,
    #                 	300, 300,	300,	300,	300,	350,
    #                         	0.500000000000000,	0.500000000000000,	0.500000000000000,	0.500000000000000,
    #                                 	0.500000000000000,	0.500000000000000,	3.55000000000000,	3.55000000000000,
    #                                         	3.52500000000000,	3.52500000000000,	3.52500000000000,
    #                                                 	0,	0,	0,	0,	0]).ravel()
    cost = Spacecraft_Trajectory_OptimizationC1(x)
    print("Delta-V cost:", cost)