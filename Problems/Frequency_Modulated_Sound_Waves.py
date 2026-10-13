
__author__ = "Iván Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### Importing libraries
import numpy as np

DTYPE = np.longdouble


def Frequency_Modulated_Sound_Waves(x:np.ndarray):
    assert x.ndim < 2, "Input array must be less than two-dimensional."

    # Perform a reshape of the initial array
    x = x.ravel()

    n = x.size

    lb = get_xl(n)
    ub = get_xu(n)

    # Convert the design variables 
    x_eval = np.abs(ub - lb)*x + lb

    # Define the value of theta
    theta = 2 * np.pi / 10
    
    # Initialize the output value y
    y:float = 0.0
    
    # Determine the number of groups 'g' based on the length of x
    g:int = int(n / 6)
    
    # Iterate over each group 'j'
    for j in range(1,g+1):
        # Iterate over 't' from 1 to 10
        for t in range(1,11):
            y_t = x_eval[6 * (j - 1)] * np.sin(x_eval[1 + 6 * (j - 1)] * t * theta + x_eval[2 + 6 * (j - 1)] * np.sin(x_eval[3 + 6 * (j - 1)] * t * theta + x_eval[4 + 6 * (j - 1)] * np.sin(x_eval[5 + 6 * (j - 1)] * t * theta)))
            y_0_t = 1 * np.sin(5 * t * theta - 1.5 * np.sin(4.8 * t * theta + 2 * np.sin(4.9 * t * theta)))
            y += (y_t - y_0_t)**2

    return y

def get_xl(nx):
    xl = -np.ones((nx,) )*6.4
    return xl

def get_xu(nx):
    xu = np.ones((nx, ))*6.35
    return xu



if __name__ == "__main__":
    # Test the function with a sample input
    # x = np.asarray([-0.0250000000000004, -0.0250000000000004, 
    #                 -0.0250000000000004, -0.0250000000000004, 
    #                 -0.0250000000000004, -0.0250000000000004]).ravel()
    x = np.asarray([-4.27500000000000,-0.0250000000000004,
                    -0.0250000000000004, -0.0250000000000004,
                    -0.0250000000000004, -0.0250000000000004])
    potential = Frequency_Modulated_Sound_Waves(x)
    print(f"Frequency Modulated Sound Waves: {potential}")