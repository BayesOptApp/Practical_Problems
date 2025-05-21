
__author__ = "Ivón Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius

### Importing libraries
import numpy as np

import os

def bench_fun_eps(x:np.ndarray)->float:

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size == 49, "The dimensionality is fixed to 10"

    x = x.ravel()

    dim = x.size

    # Set the bounds
    xL = get_xl(dim)
    xU = get_xu(dim)
    
    # Convert the design variables 
    x_eval = np.abs(xU - xL)*x + xL

    lb = np.zeros((dim,))
    ub = 7.999*np.ones((dim,))

    bounds = [xL,xU]

    # Fix the bounds to fit into the simulation
    y = change_bounds(x_eval,
                      bounds=bounds,
                      lb=lb, 
                      ub=ub)
    
    # Compute the objective
    out = fhd(y)

    return out


def fhd(x:np.ndarray):
    r"""
    This function generates a Docker/Apptainer command to run the simulation and retrieve the objective.
    """
    import subprocess

    format_list = ['{:>3}' for _ in x] 
    s = ','.join(format_list)
    formatted_text = '"' + s.format(*x) +  '"'

    str_ = "docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh ESP " + formatted_text  

    uu = subprocess.run(str_,shell=True, text=True, capture_output=True)

    # Get the output of the process
    out_string = uu.stdout.split("\n")[-2]

    if out_string.find("Error")==-1:
        try:
            out = float(out_string)
        except:
            print("Docker might not be configured in the system!")
    else:
        out = 1
    
    return out

# function for recomputing the bounds
def change_bounds(x:np.ndarray,
                  bounds,
                  lb:list,
                  ub:list):
    

    dim = x.size
    
    y = np.zeros_like(x)
    for i in range(dim):
        range_i = bounds[1][i]-bounds[0][i]
        y[i] = x[i]/range_i*(ub[i]-lb[i])+(lb[i]+ub[i])/2

    return np.floor(y)

def get_xl(nx:int)->np.ndarray:
    xl = -10*np.ones((nx,))

    return xl


def get_xu(nx:int)->np.ndarray:
    xu = 10*np.ones((nx,))
    return xu



if __name__ == "__main__":

    x = np.zeros((49,))
    efficiency = bench_fun_eps(x)
    print(f"Pitz Daily metric: {efficiency}")