import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union, Optional
from Problems.Spacecraft_Trajectory_OptimizationC1 import pleph_an, propagateKEP, vers, vett, lambertI, IC2par, ni2E

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

RAD = np.pi / 180
KM = 1.495978706910000e+008
AU = 149597870.66;# km


def Spacecraft_Trajectory_OptimizationC2(x:np.ndarray)->float:
    # Check input dimensions and size
    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size ==22, "The dimension should be 22"
    # Flatten the array
    x = x.ravel()

    n = x.size

    lb = get_xl(n)
    ub = get_xu(n)
    x = np.abs(ub - lb)*x + lb
    
    y = mga_dsm(x)
    if np.isnan(y):
        # Re-evaluate y if this is nan
        y = 10**100
    return y

def get_xl(n:int)->np.ndarray:
    xl = [-1000,3,0,0,100,100,
          30,400,800,0.01,0.01,
          0.01,0.01,0.01,1.05,1.05,
          1.15,1.7,-np.pi,-np.pi,-np.pi,-np.pi]
    return np.asarray(xl,dtype=np.longdouble)

def get_xu(n:int)->np.ndarray:
    xu = [0,5,1,1,400,
          500,300,1600,2200,0.9,
          0.9,0.9,0.9,0.9,
          6,6,6.5,291,np.pi,np.pi,
          np.pi,np.pi]
    return np.asarray(xu,dtype=np.longdouble)

def mga_dsm(t:np.ndarray)->float:
    sequence = [3,2,2,3,5,6]
    problem = {
        'objective': {'rp': 2640, 'e': 0.704, 'type' :'total DV rndv'}
    }

    # Extract relevant problem parameters based on the objective type
    if problem['objective']['type'] == 'orbit insertion':
        rp_target = problem['objective']['rp']
        e_target = problem['objective']['e']
    elif problem['objective']['type']=='total DV orbit insertion':
        rp_target = problem['objective']['rp']
        e_target = problem['objective']['e']
    elif problem['objective']['type'] =='gtoc1':
        Isp = problem['objective']['Isp']
        mass = problem['objective']['mass']
    elif problem['objective']['type'] == 'time to AUs':
        AUdist = problem['objective']['AU']
        DVtotal = problem['objective']['DVtot']
        DVonboard = problem['objective']['DVonboard']



    tdep = t[0]; # Departure time (days from the initial epoch)
    VINF = t[1]; # Hyperbolic excess velocity (km/s)
    udir = t[2]; # Unit direction vector for the spacecraft's asymptote direction
    vdir = t[3]; # Unit direction vector for the spacecraft's out-of-plane direction
    
    N = len(sequence); 
    tof = np.zeros((N-1, )); 
    alpha = np.zeros_like(tof) 
    
    # Extract time of flight and DSM fraction from the decision vector
    for i in range(N-1):
        tof[i] = t[i+4] # Planet-to-planet Time of Flight (ToF) (days)
        alpha[i] = t[N+i+3] # Fraction of ToF at which the DSM occurs

    if problem['objective']['rp']  =='time to AUs':
        rp_non_dim=np.zeros((N-1,))        # initialization gains speed
        gamma=np.zeros((N-1,))
        for i in range(N-1):
            rp_non_dim[i]=(i+2*N+2) # non-dim perigee fly-by radius of planets P2..Pn (i=1 refers to the second planet)
            gamma[i]=t[3*N+i]        # rotation of the bplane-component of the swingby outgoing
            # velocity  Vector (Vout) around the axis of the incoming swingby velocity vector (Vin)
    
    else:
        rp_non_dim=np.zeros((N-2,))        # initialization gains speed
        gamma=np.zeros_like(rp_non_dim)
        for i in range(N-2):
            rp_non_dim[i]=t[i+2*N+2]; # non-dim perigee fly-by radius of planets P2..Pn-1 (i=1 refers to the second planet)
            gamma[i]=t[3*N+i];        # rotation of the bplane-component of the swingby outgoing
            # velocity  Vector (Vout) around the axis of the incoming swingby velocity vector (Vin)
        
    
    
    nn=len(sequence)
    rr= np.zeros((3,nn))
    vv= np.zeros((3,nn))
    mu_vec=np.zeros((nn,))
    Itime = np.zeros((nn,))
    dT=np.zeros((nn,))
    seq = np.abs(sequence)
    tt=tdep
    dT[0:nn-1]=tof
    
    for i in range(nn):
        Itime[i]=tt
        if seq[i]<10:
            rr[:,i], vv[:,i], _ = pleph_an(tt, seq[i]) # Positions and velocities of solar system planets
            mu_vec[i] = MU[seq[i]-1] # Gravitational constants
        else:
            raise NotImplementedError("Planet index out of range.")
    
        
        tt +=dT[i]


    if problem['objective']['type'] =='time to AUs':
        rp=np.zeros((N-1,))        # initialization gains speed
        for i in range(N-1):
            rp[i]= rp_non_dim[i]*RPL[seq[i+1]-1] # dimensional flyby radii (i=1 corresponds to 2nd planet)
        
    else:
        rp=np.zeros((N-2,))        # initialization gains speed
        for i in range(N-2):
            rp[i]= rp_non_dim[i]*RPL[seq[i+1]-1] # dimensional flyby radii (i=1 corresponds to 2nd planet)
        
    
    
    vtemp= np.cross(rr[:,0],vv[:,0]).flatten()
    iP1= vv[:,0]/np.linalg.norm(vv[:,0])
    zP1= vtemp/np.linalg.norm(vtemp)
    jP1= np.cross(zP1,iP1)
    theta=2*np.pi*udir;         # See Picking a Point on a Sphere
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
    rd[:,0],v_sc_dsm_in[:,0]=propagateKEP(rr[:,0],v_sc_pl_out[:,0],tDSM[0]*24*60*60,MU_SUN)
    lw=vett(rd[:,0],rr[:,1])
    
    lw=np.sign(lw[2])
    if lw==1:
        lw=0
    else:
        lw=1

    v_sc_dsm_out[:,0],v_sc_pl_in[:,1], _,_,_,_= lambertI(rd[:,0],
                                                 rr[:,1],
                                                 tof[0]*(1-alpha[0])*24*60*60,
                                                 MU_SUN,
                                                 lw)
    DV=np.zeros((N-1,))
    DV[0]=np.linalg.norm(v_sc_dsm_out[:,0]-v_sc_dsm_in[:,0])
    tDSM=np.zeros((N-1,))
    
    for i in range(N-2):
        v_rel_in:np.ndarray=v_sc_pl_in[:,i+1]-vv[:,i+1]
        e=1+rp[i]/mu_vec[i+1]*np.dot(v_rel_in.T,v_rel_in) #ok<MHERM>
        beta_rot=2*np.arcsin(1/e) #velocity rotation
        ix=v_rel_in/np.linalg.norm(v_rel_in)
        iy=vett(ix,vv[:,i+1]/np.linalg.norm(vv[:,i+1])).T
        iy=iy/np.linalg.norm(iy)
        iz=vett(ix,iy).T
        iVout = np.cos(beta_rot) * ix + np.cos(gamma[i])*np.sin(beta_rot) * (iy + np.sin(gamma[i])*np.sin(beta_rot) * iz)
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
    
        v_sc_dsm_out[:,i+1],v_sc_pl_in[:,i+2], _,_,_,_=lambertI(rd[:,i+1],rr[:,i+2],tof[i+1]*(1-alpha[i+1])*24*60*60,MU_SUN,lw);
        DV[i+1]=np.linalg.norm(v_sc_dsm_out[:,i+1]-v_sc_dsm_in[:,i+1])

    DVrel=np.linalg.norm(vv[:,nn-1]-v_sc_pl_in[:,nn-1]) # Relative velocity at the target planet
    
    if problem['objective']['type'] =='orbit insertion':
        DVper=np.sqrt(DVrel**2+2*mu_vec[nn-1]/rp_target)  # Hyperbola
        DVper2=np.sqrt(2*mu_vec[nn-1]/rp_target-mu_vec[nn-1]/rp_target*(1-e_target)); # Ellipse
        DVarr=np.abs(DVper-DVper2)
    elif problem['objective']['type'] == 'total DV orbit insertion':
        DVper=np.sqrt(DVrel**2+2*mu_vec[nn-1]/rp_target)  #Hyperbola
        DVper2=np.sqrt(2*mu_vec[nn-1]/rp_target-mu_vec[nn-1]/rp_target*(1-e_target)) # Ellipse
        DVarr=np.abs(DVper-DVper2)
    elif problem['objective']['type'] =='rndv':
        DVarr = DVrel
    elif problem['objective']['type'] =='total DV rndv':
        DVarr = DVrel
    elif problem['objective']['type'] =='gtoc1':
        DVarr = DVrel
    elif problem['objective']['type'] =='time to AUs':  # No DVarr is considered
        DVarr = 0
    
    DV=np.hstack((DV,DVarr))
    
    if problem['objective']['type'] =='gtoc1':
        DVtot=np.sum(DV[0:nn-2])
    elif problem['objective']['type'] =='deflection demo':
        DVtot=np.sum(DV[0:nn-2])
    else:
        DVtot=np.sum(DV)
    
    DVvec=np.hstack((VINF, DV)).ravel()

    if problem['objective']['type'] =='total DV orbit insertion':
        J= DVtot+VINF
    elif problem['objective']['type'] =='total DV rndv':
        J= DVtot+VINF
    elif problem['objective']['type'] =='orbit insertion':
        J= DVtot
    elif problem['objective']['type'] =='rndv':
        J= DVtot
    elif problem['objective']['type'] =='gtoc1':
        mass_fin = mass * np.exp (- DVtot/ (Isp/1000 * 9.80665))
        J = 1/(mass_fin * np.abs((v_sc_pl_in[:,nn-1]-vv[:,nn-1])).T* vv[:,nn-1])
    elif problem['objective']['type'] =='deflection demo':
        mass_fin = mass * np.exp (- DVtot/ (Isp/1000 * 9.80665))
        
        VEL=np.sqrt(MU_SUN/AU)
        TIME=AU/VEL

        raise NotImplementedError()

        # # Calculate the DV due to the impact
        # relV = (V(:,N,1)-v(:,N)); %#ok<NODEF>
        # impactDV = (relV * mass_fin/astmass)/VEL;
        # % Calculate phi 
        # ir = r(:,N)/norm(r(:,N));
        # iv = v(:,N)/norm(v(:,N));
        # ih = vett(ir,iv)';
        # ih = ih/norm(ih);
        # itheta = vett(ih,ir)';
        # impactDV = (impactDV'*ir) * ir + (impactDV'*itheta) * itheta; %projection on the orbital plane
        # phi = acos((impactDV/norm(impactDV))'*ir);
        # if (impactDV'*itheta) < 0
        #     phi = phi +pi;
        # end
        # % Calculate ni0
        # r0 = r(:,N)/AU;
        # v0 = v(:,N)/VEL;
        # E0 = IC2par( r0 , v0 , 1 );
        # ni0 = E2ni(E0(6),E0(2));
        # % Calcuate the deflection projected in covenient coordinates
        # a0 = E0(1);
        # e0 = E0(2);
        # M0 = E2M(E0(6),E0(2));
        # numberofobs = 50;
        # obstime = linspace(0,obstime,numberofobs); #ok<NODEF>
        # M = M0 + obstime*60*60*24/TIME*sqrt(1/a0);
        # theta=zeros(numberofobs,1);
        # for jj = 1:numberofobs
        #     theta(jj) = M2ni(M(jj),E0(2));
        # end
        # theta = theta-ni0;
        # [~,dr] = defl_radial(a0,e0,ni0,phi,norm(impactDV),theta);

        # [~,dtan] = defl_tangential(a0,e0,ni0,phi,norm(impactDV),theta);

        # # Calculate the deflecion on the Earth-asteroid lineofsight
        # defl=zeros(3,numberofobs);
        # temp=zeros(numberofobs,1);
        # for i=1:numberofobs
        #     Tobs = T + obstime(i);
        #     [rast,vast]=CUSTOMeph( mjd20002jed(Tobs) , ...
        #         problem.customobject(seq(end)).epoch, ...
        #         problem.customobject(seq(end)).keplerian , 1);
        #     [rearth,~]=pleph_an( Tobs , 3);
        #     lineofsight=(rearth-rast)/norm((rearth-rast));
        #     defl(:,i) = rast / norm(rast) * dr(i) + ...
        #         vast / norm(vast) * dtan(i);
        #     temp(i) = norm(lineofsight'*(defl(:,i)));

        # end
        # J = 1./abs(temp)/AU;
        # [J,~]=min(J);
    elif problem['objective']['type'] =='time to AUs':
        # Non dimensional units
        V = np.sqrt(MU_SUN/AU)
        T = AU/V
        # Evaluate the state of the spacecraft after the last fly-by
        v_rel_in=v_sc_pl_in[:,nn-1]-vv[:,nn-1]
        e=1+rp[nn-2]/mu_vec[nn-1]*np.dot(v_rel_in.T,v_rel_in)#ok<MHERM>
        beta_rot=2*np.arcsin(1/e);              # velocity rotation
        ix=v_rel_in/np.linalg.norm(v_rel_in)
        # ix=r_rel_in/norm(v_rel_in);  % activating this line and disactivating the one above
        #  shifts the singularity for r_rel_in parallel to v_rel_in
        iy=vett(ix,vv[:,nn-1]/np.linalg.norm(vv[:,nn-1])).T
        iy=iy/np.linalg.norm(iy)
        iz=vett(ix,iy).T
        iVout = np.cos(beta_rot) * ix + np.cos(gamma[nn-2])*np.sin(beta_rot) * iy + np.sin(gamma[nn-2])*np.sin(beta_rot) * iz
        v_rel_out=np.linalg.norm(v_rel_in)*iVout
        v_sc_pl_out[:,nn-1]=vv[:,nn-1]+v_rel_out
        t = time2distance(rr[:,nn-1]/AU,v_sc_pl_out[:,nn-1]/V,AUdist)
        DVpen=0
        if np.sum(DVvec)>DVtotal:
            DVpen=DVpen+(np.sum(DVvec)-DVtotal)
        
        if np.sum(DVvec[1:])>DVonboard:
            DVpen=DVpen+(np.sum(DVvec[1:])-DVonboard)
    

        J= (t*T/60/60/24 + np.sum(tof))/365.25 + DVpen*100
        if np.isnan(J):
            J=100000
    
    
    return J


def time2distance(r0:np.ndarray, 
                  v0:np.ndarray, 
                  rtarget:float):
    
    r0norm = np.linalg.norm(r0)
    if r0norm < rtarget:
        out = np.sign(np.dot(r0.T,v0))
        E = IC2par(r0, v0, 1.0)
        a = E[0]; e = E[1]; E0 = E[5]; p = a * (1 - e**2)
        
        # If the solution is an ellipse 
        if e < 1:
            ra = a * (1 + e)
            if rtarget > ra:
                t = np.nan # Target distance is unreachable.
            else:
                # Find the anomaly where the target distance is reached.
                #ni = np.arccos((p/rtarget - 1) / e); # in the range 0 to pi
                ni = np.arccos(np.clip((p / rtarget - 1) / e, -1.0, 1.0))
                Et = ni2E(ni, e); # in the range 0 to pi
                if out == 1:
                    t = a**(3/2) * (Et - e * np.sin(Et) - E0 + e * np.sin(E0))
                else:
                    E0 = -E0
                    t = a**(3/2) * (Et - e * np.sin(Et) + E0 - e * np.sin(E0))
                
            
        else: # The solution is a hyperbola
            ni = np.arccos((p/rtarget - 1) / e); # in the range 0 to pi
            Et = ni2E(ni, e); # in the range 0 to pi
            if out == 1:
                t = (-a)**(3/2) * (e * np.tan(Et) - np.log(np.tan(Et/2 + np.pi/4)) - e * np.tan(E0) + np.log(np.tan(E0/2 + np.pi/4)))
            else:
                E0 = -E0
                t = (-a)**(3/2) * (e * np.tan(Et) - np.log(np.tan(Et/2 + np.pi/4)) + e * np.tan(E0) - np.log(np.tan(E0/2 + np.pi/4)))

    else:
        t = 12.0 # Arbitrary value when r0norm >= rtarget.

    return t



# Example usage:
if __name__ == "__main__":
    # Example normalized input (all values in [0,1])
    #x = np.random.rand(25)
    # x = np.asarray([2100, 3.27500000000000, 0.500000000000000, 0.500000000000000,
    #                 300, 300, 300, 300, 300, 350, 
    #                 0.500000000000000, 0.500000000000000, 0.500000000000000, 0.500000000000000,
    #                     	0.500000000000000,	0.500000000000000,	3.55000000000000,	3.55000000000000,
    #                             	3.52500000000000,	3.52500000000000,	3.52500000000000,
    #                                     	0,	0,	0,	0,	0]).ravel()
    
    # x = np.asarray([-500, 4, 0.500000000000000, 0.500000000000000,
    #                250, 300, 165, 1000, 1500,
    #                0.455000000000000, 0.455000000000000, 0.455000000000000, 0.455000000000000,
    #                0.455000000000000, 3.52500000000000, 3.52500000000000, 3.82500000000000,
    #                146.350000000000, 0, 0, 0, 0]).ravel()

    #x = np.asarray([-833.333333333333, 4, 0.500000000000000, 0.500000000000000, 250, 300, 165, 1000, 1500, 0.455000000000000, 0.455000000000000, 0.455000000000000, 0.455000000000000, 0.455000000000000, 3.52500000000000, 3.52500000000000, 3.82500000000000, 146.350000000000, 0, 0, 0, 0]).ravel()

    #x = np.asarray([-166.666666666667, 4, 0.500000000000000, 0.500000000000000, 250, 300, 165, 1000, 1500, 0.455000000000000, 0.455000000000000, 0.455000000000000, 0.455000000000000, 0.455000000000000, 3.52500000000000, 3.52500000000000, 3.82500000000000, 146.350000000000, 0, 0, 0, 0]).ravel()

    # x = np.asarray([
    #     -500, 3.33333333333333, 0.5, 0.5, 250, 300, 165, 1000, 1500,
    #     0.455, 0.455, 0.455, 0.455, 0.455,
    #     3.525, 3.525, 3.825, 146.35,
    #     0, 0, 0, 0
    # ]).ravel()

    x = np.asarray([-500,4,5.000000e-01,5.000000e-01,250,1.666667e+02,2.550000e+02,600,1500,4.550000e-01,4.550000e-01,4.550000e-01,4.550000e-01,4.550000e-01,3.525000e+00,5.175000e+00,3.825000e+00,2.427833e+02,0,0,0,2.094395e+00])

    cost = Spacecraft_Trajectory_OptimizationC2(x)
    print("Delta-V cost:", cost)