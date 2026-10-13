
__author__ = "Iván Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### =========================
### CONSTANTS
### =========================

# Define constants and parameters
R1 = 3.0 # CUT-OFF distance
R2 = 0.2 # Cut-off width
A = 3.2647e+3 # Repulsive pair potential pre-factor
B = 9.5373e+1 # Attractive pair potential pre-factor
LEMDA1 = 3.2394 # Decay length for the repulsive potential
LEMDA2 = 1.3258 # Decay length for the attractive potential
LEMDA3 = 1.3258 # Decay length for the angular potential
C = 4.8381 # Angular Coefficient
D = 2.0417 # # Angular Parameter
N1 = 22.956 # Angular Exponent
BETA =  0.33675 # Angular strength parameter
GAMA = 1.00 # Angular Scaling
H = 0 # Cosine of angular parameter
M = 3.00 #Bond order exponent

### Importing libraries
import numpy as np

# This is a handle to avoid the overflow warning 
from scipy.special import expit

### ===============================
### Tersoff potential problem
### ================================

def Tersoff_PotentialC1(x:np.ndarray) -> float:
    r"""Tersoff potential problem.
    The Tersoff potential is a type of interatomic potential used in molecular dynamics simulations.
    It is particularly useful for simulating covalent materials, such as silicon and carbon.
    The potential is defined in terms of the distances between atoms and includes both attractive and repulsive components.
    The potential is a function of the positions of atoms and their neighbors, and it is parameterized by several constants.

    # CHECK -> https://wiki.fysik.dtu.dk/ase/ase/calculators/tersoff.html

    Args
    ------------
    - x (`np.ndarray`): Input array representing the positions of atoms.

    Returns
    ------------
    - `float`: The computed Tersoff potential energy.
    
    """

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    # Flatten the input array to ensure it is one-dimensional
    x = x.flatten()

    n = x.size
    lb = get_xl(n)
    ub = get_xu(n)

    x_eval = np.abs(ub - lb)*x + lb
    #x_eval = np.abs(x - lb) + lb
    
    # Reshape the input vector x into a matrix
    p = n
    NP = int(p / 3)
    x_eval = x_eval.reshape((3, -1), order='F').T
    

    
    # Initialize variables
    E = np.zeros((NP, ))
    r = np.zeros((NP, NP ))
    fcr = np.zeros((NP, NP ))
    VRr = np.zeros((NP, NP ))
    VAr = np.zeros((NP, NP ))

    # Compute pairwise distances and functions for each point pair
    for i in range(NP):
        for j in range(NP):
            r[i, j] = np.sqrt(np.sum(np.power(x_eval[i, :] - x_eval[j, :],2)))
            if r[i, j] < (R1 - R2):
                fcr[i, j] = 1
            elif r[i, j] > (R1 + R2):
                fcr[i, j] = 0
            else:
                fcr[i, j] = 0.5 - 0.5 * np.sin(np.pi / 2 * (r[i, j] - R1) / R2)
            
    
            VRr[i, j] = A * np.exp(-LEMDA1 * r[i, j])
            VAr[i, j] = - B * np.exp(-LEMDA2 * r[i, j])

    # # Compute E
    # for i in range(NP):
    #     for j  in range(NP):
    #         if i == j:
    #             continue
    #         jeta = np.zeros((NP, NP))
    #         for k in range(NP):
    #             if (i == k) or  (j == k):
    #                 continue
    #             rd1 = max(np.sqrt(np.sum(np.power(x_eval[i, :] - x_eval[k, :],2))), 1e-16)
    #             rd3 = max(np.sqrt(np.sum(np.power(x_eval[k, :] - x_eval[j, :],2))), 1e-16)
    #             rd2 = max(np.sqrt(np.sum(np.power(x_eval[i, :] - x_eval[j, :],2))), 1e-16)
    #             ctheta_ijk = (rd1**2 + rd2**2 - rd3**3) / (2 * rd1 * rd2)
    #             G_th_ijk = GAMA*(1 + (C**2) / (D**2) - (C**2) / (D**2 + (H - ctheta_ijk)**2))
    #             jeta[i, j] = jeta[i, j] + fcr[i, k] * G_th_ijk * np.exp(LEMDA3**M * (r[i, j] - r[i, k])**M)

    #         Bij = (1 + (BETA * jeta[i, j])**N1)**(-0.5 / N1)
    #         E[i] = E[i] + fcr[i, j] * (VRr[i, j] + Bij * VAr[i, j]) / 2

    # Compute E using a numerically stable bond-order calculation
    for i in range(NP):
        for j in range(NP):
            if i == j or fcr[i, j] == 0:
                continue

            # log(0) represents an initially empty sum
            log_jeta = -np.inf

            for k in range(NP):
                if i == k or j == k or fcr[i, k] == 0:
                    continue

                rd1 = max(r[i, k], 1e-16)
                rd2 = max(r[i, j], 1e-16)
                rd3 = max(r[k, j], 1e-16)

                # Preserves the formula in the original scripts.
                # NOTE: Use rd3**2 instead to correct the law of cosines formula. However
                # this would change the behavior of the function, so we keep it as is for now.  
                ctheta_ijk = (
                    rd1**2 + rd2**2 - rd3**3
                ) / (2 * rd1 * rd2)

                G_th_ijk = GAMA * (
                    1 + C**2 / D**2
                    - C**2 / (D**2 + (H - ctheta_ijk)**2)
                )

                # log(fcr * G * exp(exponent))
                exponent = (LEMDA3 * (r[i, j] - r[i, k]))**M
                log_term = (
                    np.log(fcr[i, k])
                    + np.log(G_th_ijk)
                    + exponent
                )
                log_jeta = np.logaddexp(log_jeta, log_term)

            # Bij = [1 + (BETA * jeta)**N1]**(-0.5 / N1)
            t = N1 * (np.log(BETA) + log_jeta)
            log_Bij = (-0.5 / N1) * np.logaddexp(0.0, t)
            Bij = np.exp(log_Bij)

            E[i] += (
                fcr[i, j]
                * (VRr[i, j] + Bij * VAr[i, j])
                / 2
            )

    # Sum all the E
    y = np.sum(E)
    
    # Check for NaN value and set a large value if necessary
    if np.isnan(y):
        y = 10**100
    
    return y


def get_xl(nx:int)->np.ndarray:
    r"""Lower bounds for the design variables."""
    xl = np.zeros((nx, ))
    for i in range(3,nx):
        xl[i] = -4 - (0.25)*((i-3)/3); ##ok<*AGROW>
    return xl

def get_xu(nx:int)->np.ndarray:
    r"""Upper bounds for the design variables."""
    xu = np.zeros((nx, ))
    xu[0] = 4
    xu[1] = 4
    xu[2] = 3
    for i in range(3,nx):
        xu[i] = 4 + (0.25)*((i-3)/3)
    
    return xu



if __name__ == "__main__":
    # Test the function with a sample input
    
    #x = np.array([2, 2, 1.50000000000000, 0, 0,	0]).ravel()
    x = np.array([2,2,1.50000000000000,2.66666666666667,0,0,2.83333333333333,0,0,0, 0, 0]).ravel()
    #x = np.array([3.33333333333333, 2, 1.50000000000000, 0, 0, 0]).ravel()
    #x = np.array([0, 0, 3, -4, -4, -4]).ravel()
    #x = np.array([2, 2, 1.50000000000000,	0,
    #              	0, 0 ]).ravel()
    potential = Tersoff_PotentialC1(x)
    print(f"Tersoff Potential: {potential}")