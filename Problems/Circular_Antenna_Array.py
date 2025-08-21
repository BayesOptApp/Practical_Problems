
__author__ = "Ivón Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### Importing libraries
import numpy as np

DTYPE = np.longdouble


# Set the machine precision
EPS = np.finfo(np.float64).eps

### ===============================
### Circular Antenna Array Problem
### ================================

def Circular_Antenna_Array(x:np.ndarray) -> np.float64:
    r"""
    Computes the fitness value for the Circular Antenna Array problem.


    Args
    ----------------
    - x: np.ndarray: Array of shape (N, ) with design variables.

    Returns
    -----------------
    `float`.
    """

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size  == 12, "Input array must be non-empty and equal to 12."

    # Perform a reshape of the initial array
    x = x.ravel()

    # Obtain the size of the array
    n = x.size

    lb = get_xl(n)
    ub = get_xu(n)

    # Convert the design variables 
    x = np.abs(ub - lb)*x + lb

    # Compute the fitness value
    y = Fitness(x)

    return y

def get_xl(n:int) -> np.ndarray:
    r"""Lower bounds for the design variables."""
    xl = np.hstack((np.ones((6, ),dtype=DTYPE)*0.2, -np.ones((6, ),dtype=DTYPE)*180.0))
    return xl

def  get_xu(n:int) -> np.ndarray:
    r"""Upper bounds for the design variables."""
    xu = np.hstack((np.ones((6, ),dtype=DTYPE)*1.0, np.ones((6, ),dtype=DTYPE)*180.0))
    return xu

def Fitness(x:np.ndarray):
    r"""
    Computes the fitness value for the Circular Antenna Array problem.  
    """
    # Constants and parameters
    null = 50
    phi_desired = np.longdouble(180.0)
    distance = np.longdouble(0.5)
    dim = x.size
    
    #phizero = 0
    #[~, num_null] = size(null)
    num_null = 1
    num1 = 300

    # Generate an angular grid for phi
    #phi = np.arange(0.00, 360.00, 1, dtype=DTYPE)
    phi = np.linspace(0.0, 360.0, num1, endpoint=False , dtype=DTYPE)
    phi_rad =np.deg2rad(phi)
    #num1 = phi.size
    #phi[-1] = 0.0
    #phi_rad[-1] = 0.0

    # Initialize yax, sidelobes and sllphi as lists
    yax = np.zeros_like(phi, dtype=DTYPE)
    sidelobes = []
    sllphi = []

    # Calculate array factor for different phi angles and find maximum
    #yax.append(array_factor(x, (pi/180)*phi[0], phi_desired, distance, dim))
    #maxi = yax[0]
    #phi_ref = 0
    for i in range(num1):
        # Calculate the array factor for the current angle
        yax[i] = array_factor(x, phi_rad[i], phi_desired, distance, dim)
        #if i != num1 - 1:
            
        #else:
            #NOTE: This is to compute the circular values better and compatible
            # with the original code because Numpy does not have the circular equivalence
            # thus there are some rounding errors.
            #yax[i] =yax[0]       
        # if maxi < yax[i]:
        #     maxi = yax[i]
        #     phizero = phi[i]
        #     phi_ref = i
    


    # Find the maximum value and its index
    maxi = np.max(yax)
    phizero = phi[np.argmax(yax)]
    phi_ref = np.argmax(yax)


    # Find sidelobes
    for i in range(num1):
        prev_i = (i - 1) % num1
        next_i = (i + 1) % num1
        if (yax[i] - yax[prev_i] > EPS) and (yax[i] - yax[next_i] > EPS):
            sidelobes.append(yax[i])
            sllphi.append(phi[i])

    # Sort the sidelobes
    sidelobes.sort(reverse=True)


    upper_bound = 180.0
    lower_bound = 180.0
    s2 = sidelobes[1] if len(sidelobes) > 1 else sidelobes[0]
    y:float = s2/maxi
    sllreturn = 20.0*np.log10(y)

    # Calculate upper and lower beamwidth bounds
    for i in range(int(num1/2)):
        if (phi_ref + i) > num1-1:
            upper_bound = 180.0
            break
        if yax[phi_ref + i] < yax[phi_ref + i - 1] and yax[phi_ref + i] < yax[phi_ref + i + 1]:
            upper_bound = phi[phi_ref + i] - phi[phi_ref]
            break
        
    
    # Calculate the lower bound
    for i in range(int(num1/2)):
        if (phi_ref - i < 2):
            lower_bound = 180.0
            break
        if yax[phi_ref - i] < yax[phi_ref - i - 1] and yax[phi_ref - i] < yax[phi_ref - i + 1]:
            lower_bound = phi[phi_ref] - phi[phi_ref - i]
            break
        
    bwfn = upper_bound + lower_bound

    # Calculate the objective function components
    y1 = 0.0
    for i in range(num_null):
        # The objective function for null control is calculated here
        y1 += array_factor(x, null, phi_desired, distance, dim)/maxi
    
    y3 = np.abs(phizero - phi_desired)

    if y3 < 5.0:
        y3 = 0.0

    y = 0.0
    if bwfn > 80:
        y += np.abs(bwfn - 80)

    # Combine the components to calculate the final fitness value 'y'
    y += sllreturn + y1 + y3

    # Check for NaN value and set a large value if necessary
    if np.isnan(y):
        y = 10**100

    return y

def array_factor(x1:np.ndarray, 
                 phi:DTYPE, 
                 phi_desired:DTYPE, 
                 distance:float, 
                 dim:int)->DTYPE:
    
    r"""
    Computes the array factor
    """

    y = 0
    y1 = 0

    for i1 in range(1,int(dim/2)+1):
        delphi = 2.0*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(np.radians(phi_desired) - delphi)
        shi = shi * dim * distance
        y += x1[i1-1] * np.cos(shi + np.radians(x1[int(dim/2) + i1 - 1]))
    

    for i1 in np.arange(int(dim/2)+1,dim+1,dtype=int):
        delphi = 2.0*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(np.radians(phi_desired) - delphi)
        shi = shi * dim * distance
        y +=  x1[(i1-1 - int(dim/2))] * np.cos(shi - np.radians(x1[i1-1]))
    

    for i1 in range(1,int(dim/2)+1):
        delphi = 2.0*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(np.radians(phi_desired) - delphi)
        shi = shi * dim * distance
        y1 +=  x1[i1-1] * np.sin(shi + np.radians(x1[int(dim/2-1) + i1]))
    

    for i1 in np.arange(int(dim/2)+1,dim+1,dtype=int):
        delphi = 2.0*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(np.radians(phi_desired) - delphi)
        shi = shi * dim * distance
        y1 += x1[i1 - int(dim/2) -1 ] * np.sin(shi - np.radians(x1[i1-1]))
    

    y = np.hypot(y, y1)

    return y




if __name__ == "__main__":
    # Test the function with a sample input
    #x = np.asarray([2, 2,	1.5,	0,	0,	0]).ravel()
    #x = np.asarray([0.666666666666667, 2, 1.50000000000000, 0, 0, 0]).ravel()
    #x = np.asarray([2, 2, 1.50000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0]).ravel()
    #x = np.asarray([0.600000000000000, 0.600000000000000, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                	0,	0,	0,	0,	0,	0]).ravel()
    #x = np.asarray([0.6000000000000, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                	0,	0,	0,	0,	0,	0]).ravel()
    # x = np.asarray([0.333333333333333, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                	0,	0,	0,	0,	0,	0]).ravel()
    # x = np.asarray([0.866666666666667, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                	0,	0,	0,	0,	0,	0]).ravel()
    # x = np.asarray([0.866666666666667, 0.333333333333333,	0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                	0,	0,	0,	0,	0,	0]).ravel() ### ERROR
    x = np.asarray([0.996566,0.599517,0.330035,0.240970,0.332803,0.599955,-0.000677,0.006097,0.000000,39.993903,0.000000,-0.016258]).ravel() ### ERROR
    potential = Circular_Antenna_Array(x)
    print(f"Circular Antena Function: {potential}")