
__author__ = "Ivón Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### Importing libraries
import numpy as np

DTYPE = np.longdouble

### ===============================
### Lennard-Jones potential
### ================================


def LJ_potential(x:np.ndarray) -> np.ndarray:
    """
    Lennard-Jones potential function.


    Args
    ----------------
    - x: np.ndarray: Array of shape (N, ) representing the coordinates of [N-3] particles.

    Returns
    -----------------
    float: Lennard-Jones potential energy.
    """

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size >= 3, "Input array must have at least 3 elements."

    # Perform a reshape of the initial array
    x = x.ravel()

    # Convert the design variables
    x = x.astype(DTYPE)

    # Obtain the size of the array
    n = x.size

    # Extract the lower and uipper bounds of the potential
    xl = get_lower_bound(n)
    xu = get_upper_bound(n)

    # Map the array to the potential bounds
    x_eval = np.abs(xu-xl)* x  + xl

    # compute the number of particles
    n_particles = int(n/3)

    # Reshape the array to obtain the coordinates of the particles
    x_eval = x_eval.reshape((-1, 3))

    r = np.ones((3, n_particles))

    # Compute the potential energy
    y = 0.0
    a = np.ones((n_particles,n_particles))
    b = np.tile(2, (n_particles, n_particles))

    for ii in range(n_particles):
        for jj in range(ii+1,n_particles):
                # Compute the distance between particles
                r[ii, jj] = max(np.sqrt(np.sum((x_eval[ii, :] - x_eval[jj, :])**2)), 1e-16)
                # Compute the potential energy
                y += (a[ii,jj]/r[ii,jj]**12 - b[ii,jj]/r[ii,jj]**6)


    return y



def get_lower_bound(nx:int)->np.ndarray:
    """
    Function to obtain the lower bound of the vector.

    Args
    ----------------
    - nx: int: Number of variables

    Returns
    -----------------
    np.ndarray: Lower bound of the potential.
    """
    
    # Instantiate the lower bound of the potential
    xl = np.zeros((nx, ))

    for ii in range(3,nx):
        xl[ii] = -4 - (0.25)*((ii-3)/3)
    
    return xl.astype(DTYPE)


def get_upper_bound(nx:int)->np.ndarray:
    """
    Function to obtain the upper bound of the vector.

    Args
    ----------------
    - nx: int: Number of variables

    Returns
    -----------------
    np.ndarray: upper bound of the potential.
    """
    # Instantiate the upper bound of the potential
    xu = np.zeros((nx, ))
    xu[0] = 4
    xu[1] = 4
    xu[2] = 3

    for ii in range(3,nx):
        xu[ii] = 4 + (0.25)*((ii-3)/3)

    return xu.astype(DTYPE)




if __name__ == "__main__":
    # Test the function with a sample input
    #x = np.asarray([2, 2,	1.5,	0,	0,	0]).ravel()
    #x = np.asarray([0.666666666666667, 2, 1.50000000000000, 0, 0, 0]).ravel()
    x = np.asarray([2, 2, 1.50000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0]).ravel()
    potential = LJ_potential(x)
    print(f"Lennard-Jones potential: {potential}")