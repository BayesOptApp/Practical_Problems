
__author__ = "Ivón Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius

### =================================
### CONSTANTS
### ==================================
LB: list = [0.0063, -0.0465, 0.0019, -0.0484, 0.0167, -0.0116, 0.0046, -0.0439, 0.0034, -0.0471]
UB: list = [0.2339, 0.0093, 0.1333, 0.0078, 0.2300, 0.0103, 0.2267, 0.0046, 0.2408, 0.0100]

### Importing libraries
import numpy as np

import os

def bench_fun_pitz(x:np.ndarray)->float:

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size == 10, "The dimensionality is fixed to 10"

    x = x.ravel()

    dim = x.size

    # Set the bounds
    xL = get_xl(dim)
    xU = get_xu(dim)
    
    # Convert the design variables 
    x_eval = np.abs(xU - xL)*x + xL


    bounds = [xL,xU]

    # Fix the bounds to fit into the simulation
    y = change_bounds(x_eval,bounds=bounds,
                      lb=LB, ub=UB)
    
    # Compute the objective
    out = fhd(y)

    return out


# def fhd(x:np.ndarray):
#     r"""
#     This function generates a Docker/Apptainer command to run the simulation and retrieve the objective.
#     """
#     import subprocess

#     format_list = ['{:>3}' for _ in x] 
#     s = ','.join(format_list)
#     formatted_text = '"' + s.format(*x) +  '"'

#     str_ = "docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh PitzDaily " + formatted_text  

#     uu = subprocess.run(str_,shell=True, text=True, capture_output=True)

#     # Get the output of the process
#     out_string = uu.stdout.replace("\n","")

#     if out_string.find("Error")==-1:
#         try:
#             out = float(out_string)
#         except:
#             print("Docker might not be configured in the system!")
#     else:
#         out = 1
    
#     return out

import subprocess
from typing import Iterable, Union

def fhd(x: Iterable[float]) -> Union[int, float]:
    """
    Run the Docker-based objective function.

    Mirrors the MATLAB behavior:
    - Formats x as comma-separated with 3 decimals.
    - Runs:
        docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh PitzDaily <args>
    - If stdout == 'Error' -> returns 1
    - Else parses stdout as a number and returns it.
    """
    args = ",".join(f"{v:.4f}" for v in x)

    cmd = [
        "docker", "run", "--rm",
        "frehbach/cfd-test-problem-suite",
        "./dockerCall.sh", "PitzDaily", args
    ]

    # Run the command; capture stdout/stderr as text
    res = subprocess.run(cmd, capture_output=True, text=True)

    val = (res.stdout or "").strip()

    if val == "Error":
        return 1

    try:
        return float(val)
    except ValueError:
        # Keep behavior close to MATLAB but give a helpful error if output isn't numeric
        raise RuntimeError(
            f"Unexpected output from dockerCall.sh: {val}\n"
            f"(exit code {res.returncode}, stderr: {res.stderr.strip()})"
        )


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

    return y

def get_xl(nx:int)->np.ndarray:
    xl = -10*np.ones((nx,))

    return xl


def get_xu(nx:int)->np.ndarray:
    xu = 10*np.ones((nx,))
    return xu



if __name__ == "__main__":

    x = np.zeros((10,))
    #x = np.asarray([4.81481481481481, -10, -10, -10, -10, -10, -10, -10, -10, -10]).ravel()  # Example input
    efficiency = bench_fun_pitz(x)
    print(f"Pitz Daily metric: {efficiency}")