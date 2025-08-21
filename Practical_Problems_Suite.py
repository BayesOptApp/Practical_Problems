from Problems import *
from typing import Union, Optional, List
from abc import abstractmethod, ABC
import numpy as np
import ioh


class Practical_Problem(ioh.iohcpp.problem.RealSingleObjective):
    r"""
    This is a template class to hold the Practical Problems
    """
   
    _actual_lower_bound:np.ndarray = np.asarray([])
    _actual_upper_bound:np.ndarray = np.asarray([])
    _name = ""

    def __init__(self,  
                 n_variables:int,
                 prob_id:int):
        
        r"""
        Class initializer

        Args
        -----------
        - n_variables (`int`): Number of variables of the problem
        """
        
        # Set the bounds 
        bounds = ioh.iohcpp.RealBounds(n_variables, -5.0, 5.0) # as BBOB

        self._set_bounds(n_variables)

        # Set a random optimum
        optimum = ioh.iohcpp.RealSolution([0]* n_variables, 0)

        super().__init__(self._name, n_variables, prob_id, True, bounds, [], optimum)
        self.set_id = int(prob_id)
    
    @abstractmethod
    def _set_bounds(self,nn:int)->None:
        r""" 
        Sets the bounds of the actual problem

        Args
        -----------
        - nn (`int`): Integer with some dimensionality
        """

        pass

    def __map2realspace(self,x:np.ndarray)->np.ndarray:
        r"""
        Maps the input array in the range from -5.0 to 5.0 as in BBOB to the actual range.
        """

        x_mod = np.zeros_like(x)

        for ii in range(self.meta_data.n_variables):
            factor = (x[ii]+5)/10
            x_mod[ii] = factor*(self._actual_upper_bound[ii] - self._actual_lower_bound[ii]) + self._actual_lower_bound[ii]
        
        return x_mod.ravel()
    
    @abstractmethod
    def evaluate(self, x:Union[np.ndarray,List[float]])->np.ndarray:
        r"""
        This overrides the default evaluation function from IOH Experimenter.

        Args:
        ----------------
        - x (`Union[np.ndarray,List[float]]`): An array with the corresponding parameter configuration.
        Returns
        ----------------
        - `np.ndarray`: A NumPy array with the actual bounds of the problem
        """
        # Convert to NumPy array
        x = np.asarray(x).ravel()

        x_mod = self.__map2realspace(x)

        return x_mod

class Bench_Fun_Eps(Practical_Problem):

    

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "bench_fun_eps"
        super().__init__(n_variables,1121)
    
    def _set_bounds(self,nn:int)->None:
        if nn == 49:
            self._actual_lower_bound = np.ones((nn,))*-10.0
            self._actual_upper_bound = np.ones((nn,))*10.0
        else:
            raise AttributeError("This problem only allows 49 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Bench_Fun_Eps problem
        """
        x_mod = super().evaluate(x)

        return bench_fun_eps(x_mod)

class Bench_Fun_Pitz(Practical_Problem):

    

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "bench_fun_pitz"
        super().__init__(n_variables,1122)
    
    def _set_bounds(self,nn:int)->None:
        if nn == 10:
            self._actual_lower_bound = np.ones((nn,))*-10.0
            self._actual_upper_bound = np.ones((nn,))*10.0
        else:
            raise AttributeError("This problem only allows 10 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Bench_Fun_Pitz problem
        """
        x_mod = super().evaluate(x)

        return bench_fun_pitz(x_mod)


class Circular_antenna_array(Practical_Problem):

    

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Circular_Antenna_Array"
        super().__init__(n_variables, 1123)
    
    def _set_bounds(self,nn:int)->None:
        if nn == 12:
            self._actual_lower_bound = np.asarray([0.200000000000000, 0.200000000000000,
                                         0.200000000000000, 0.200000000000000,
                                         0.200000000000000, 0.200000000000000,
                                         -180, -180, 
                                         -180, -180, 
                                         -180, -180])
            
            self._actual_upper_bound = np.asarray([1.0, 1.0,
                                                    1.0, 1.0,
                                                    1.0, 1.0,
                                                    180, 180, 
                                                    180, 180, 
                                                    180, 180])
        else:
            raise AttributeError("This problem only allows 12 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Circular Antenna Array problem
        """
        x_mod = super().evaluate(x)

        return Circular_Antenna_Array(x_mod)
    

class Frequency_modulated_sound_waves(Practical_Problem):

   

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Frequency_Modulated_Sound_Waves"
        super().__init__(n_variables,1124)
    
    def _set_bounds(self,nn:int)->None:
        if nn in [6,12,24,48]:
            self._actual_lower_bound = np.ones((nn,))*-6.40000000000000
            
            self._actual_upper_bound =  np.ones((nn,))*6.35000000000000
       
        else:
            raise AttributeError("This problem only allows 6, 12, 24 and 48 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Frequency_Modulated_Sound_Waves problem
        """
        x_mod = super().evaluate(x)

        return Frequency_Modulated_Sound_Waves(x_mod)

class Lennard_Jones_Potential(Practical_Problem):

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Lennard_Jones_Potential"
        super().__init__(n_variables,1125)
    
    def _set_bounds(self,nn:int)->None:
        if nn in [6,12,24,48]:
            self._actual_lower_bound =  np.zeros((nn, ))
            self._actual_upper_bound = np.zeros_like(self._actual_lower_bound)

            self._actual_upper_bound[0:3] = np.asarray([4,4,3])
            for ii in range(3, nn):
                self._actual_lower_bound[ii] = -4 - (0.25)*((ii-3)/3)
                self._actual_upper_bound[ii] = 4 + (0.25)*((ii-3)/3)
        else:
            raise AttributeError("This problem only allows 6, 12, 24 and 48 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Lennard-Jones Potential problem
        """
        x_mod = super().evaluate(x)

        return LJ_potential(x_mod)

class Spacecraft_trajectory_optimizationC1(Practical_Problem):
   

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Spacecraft_Trajectory_OptimizationC1"
        super().__init__(n_variables,1126)
    
    def _set_bounds(self,nn:int)->None:
        if nn ==26:
            self._actual_lower_bound =  np.array([1900, 2.5, 0, 0,*[100]*6,*[0.01]*6,
                                                  1.1, 1.1, 1.05, 1.05, 1.05,*[-np.pi]*5]).ravel()
            self._actual_upper_bound =np.array([2300, 4.05, 1, 1,*[500]*5, 600,*[0.99]*6,
                                                6, 6, 6, 6, 6,*[np.pi]*5]).ravel()

        else:
            raise AttributeError("This problem only allows 26 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the  Spacecraft_Trajectory_OptimizationC1 problem
        """
        x_mod = super().evaluate(x)

        return Spacecraft_Trajectory_OptimizationC1(x_mod)


class Spacecraft_trajectory_optimizationC2(Practical_Problem):
   

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Spacecraft_Trajectory_OptimizationC2"
        super().__init__(n_variables,1127)
        
    
    def _set_bounds(self,nn:int)->None:
        if nn ==22:
            self._actual_lower_bound =  np.array([-1000,3,0,0,100,100,
                                                  30,400,800,0.01,0.01,
                                                  0.01,0.01,0.01,1.05,1.05,
                                                  1.15,1.7,-np.pi,-np.pi,-np.pi,-np.pi]).ravel()
            
            self._actual_upper_bound =np.array([0,5,1,1,400,500,300,1600,2200,0.9,
                                                0.9,0.9,0.9,0.9,
                                                6,6,6.5,291,np.pi,np.pi,np.pi,np.pi]).ravel()

        else:
            raise AttributeError("This problem only allows 22 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Spacecraft_Trajectory_OptimizationC2 problem
        """
        x_mod = super().evaluate(x)

        return Spacecraft_Trajectory_OptimizationC2(x_mod)

class Spread_spectrum_radar(Practical_Problem):

    

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Spread_Spectrum_Radar"
        super().__init__(n_variables,1128)
    
    def _set_bounds(self,nn:int)->None:
        if nn ==20:
            self._actual_lower_bound =  np.zeros((nn, ))
            self._actual_upper_bound = np.ones((nn, ))*2*np.pi
        else:
            raise AttributeError("This problem only allows 20 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Spread Spectrum Radar problem
        """
        x_mod = super().evaluate(x)

        return Spread_Spectrum_Radar(x_mod)

class Tersoff_potentialC1(Practical_Problem):

   

    # Call the superclass
    def __init__(self, n_variables):
        self._name = "Tersoff_PotentialC1"
        super().__init__(n_variables,1129)
    
    def _set_bounds(self,nn:int)->None:
        if nn in [6,12,24,48]:

            self._actual_lower_bound =  np.zeros((nn, ))
            self._actual_upper_bound = np.zeros_like(self._actual_lower_bound)

            self._actual_upper_bound[0:3] = np.asarray([4,4,3])
            for ii in range(3, nn):
                self._actual_lower_bound[ii] = -4 - (0.25)*((ii-3)/3)
                self._actual_upper_bound[ii] = 4 + (0.25)*((ii-3)/3)
       
        else:
            raise AttributeError("This problem only allows 6, 12, 24 and 48 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Tersoff Potential problem
        """
        x_mod = super().evaluate(x)

        return Tersoff_PotentialC1(x_mod)


class Windwake(Practical_Problem):

    # Call the superclass
    def __init__(self, n_variables, 
                 wind_seed:int = 0,
                 n_samples:int = 5
                 ):
        self._name = "WindWake"
        super().__init__(n_variables,1130)

        # Set hyperparameters of the problem
        self._wind_seed = wind_seed
        self._n_samples = n_samples

        self._layout_object = WindWakeLayout(n_turbines=n_variables//2,wind_seed=self._wind_seed,
                             n_samples=self._n_samples)
    
    @property
    def wind_seed(self)->int:
        return self._wind_seed
    
    @property
    def n_samples(self)->int:
        return self._n_samples
    
    def _set_bounds(self,nn:int)->None:
        if nn in [6,12,24,48]:

            self._actual_lower_bound =  np.zeros((nn, ))
            self._actual_upper_bound = np.ones_like(self._actual_lower_bound)
       
        else:
            raise AttributeError("This problem only allows 6, 12, 24 and 48 dimensions")
    
    def evaluate(self, x:np.ndarray)->float:
        r"""
        Returns the function evaluation of the Wind Wake Layout problem
        """
        x_mod = super().evaluate(x)

    

        return self._layout_object.evaluate(x_mod)



def get_practical_problem(idx:int,
                          dim:int,
                          **kwargs)->Practical_Problem:
    
    r"""
    Returns a problem instance given an identifier and a dimensionality
    """

    # The kwargs are just meant as add ones for the Wind Layout Optimization
    wind_seed = int(kwargs.pop('wind_seed',0))
    n_samples = int(kwargs.pop('n_samples',5))

    

    if idx == 1121:
        return Bench_Fun_Eps(dim)
    elif idx == 1122:
        return Bench_Fun_Pitz(dim)
    elif idx == 1123:
        return Circular_antenna_array(dim)
    elif idx ==1124:
        return Frequency_modulated_sound_waves(dim)
    elif idx ==1125:
        return Lennard_Jones_Potential(dim)
    elif idx ==1126:
        return Spacecraft_trajectory_optimizationC1(dim)
    elif idx==1127:
        return Spacecraft_trajectory_optimizationC2(dim)
    elif idx ==1128:
        return Spread_spectrum_radar(dim)
    elif idx ==1129:
        return Tersoff_potentialC1(dim)
    elif idx ==1130:
        return Windwake(dim,wind_seed=wind_seed,
                        n_samples=n_samples)
    
    