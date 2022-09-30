import numpy as np
from sims import main
import GPy
import os
    
if __name__ == "__main__":
    """__init__ function to run the main function. 
    It has two simulation program to run - 
        1. Equation of state learning
        2. Stress vs shear rate
    """
    # EOS simulation
    # path = os.path.dirname(os.path.realpath(__file__))
    # path = os.path.join(path,'data')    
    # kernel = GPy.kern.RBF(1) * GPy.kern.Linear(1) * GPy.kern.Poly(1)
    # main.eos_active_learning(kernel,path,8,1e-1,25,5,100, 1000)
    
    # Stress vs rate simulation
    main.stress_viscosity_learning(0,1.2e1,5,100,100, 1e-5, 1)


       
    