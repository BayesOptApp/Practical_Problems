
__author__ = "Iván Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### Importing libraries
import numpy as np

### ===============================
### Spread Spectrum Radar 
### ================================



def Spread_Spectrum_Radar(x:np.ndarray) -> float:
    r"""
    Spread Spectrum Radar function.
    This function is a MATLAB translation of the original code provided.
    """
    assert x.ndim < 2, "Input array must be less than two-dimensional."

    n = x.size
    lb = get_xl(n)
    ub = get_xu(n)
    x_eval = np.abs(ub - lb)*x + lb
    
    # Determine the number of variables and initialize variables
    var = 2 * n - 1 
    hsum = np.zeros((2*var, ))
    
    # Calculate the values of hsum vector
    for kk in range(2 * var):
        if kk % 2 != 0:
            # Odd index
            i = int((kk + 1) / 2)
            hsum[kk] = 0
            for j in range(i,n):
                summ = 0
                for i1 in range(abs(2 * i - j - 1) ,j+1):
                    summ += x_eval[i1] 
                hsum[kk] += np.cos(summ) 

            hsum[kk] +=  0.5
        else:
            # Even index
            i = int(kk / 2)
            hsum[kk] = 0
            for j in range(i,n):
                summ = 0
                for i1 in range((abs(2 * i - j)),j+1):
                    summ += x_eval[i1] 
                
                hsum[kk] += np.cos(summ) 
            
            
        
    
    
    # Calculate the maximum value of hsum as the fitness value y
    y = max(hsum)
    
    # Check for NaN value and set a large value if necessary
    if np.isnan(y):
        y = 10**100
    
    return y


def get_xl(nx:int)->np.ndarray:
    r"""Lower bounds for the design variables."""
    xl = np.zeros((nx, ))
    return xl

def get_xu(nx:int)->np.ndarray:
    r"""Upper bounds for the design variables."""
    xu = np.ones((nx, ))*2*np.pi

    return xu


if __name__ == "__main__":
    # Test the function with a sample input
    # x = np.asarray([3.14159265358980, 3.14159265358980, 3.14159265358980, 3.14159265358980,	3.14159265358980,
    #                 	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,
    #                         	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,
    #                                 	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980]).ravel()
    
    x = np.asarray([1.04719755119660, 3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,
                    	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,
                            	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,	3.14159265358980,
                                    	3.14159265358980, 3.14159265358980, 3.14159265358980, 3.14159265358980, 3.14159265358980]).ravel()
    potential = Spread_Spectrum_Radar(x)
    print(f"Spread Spectrum Radar: {potential}")