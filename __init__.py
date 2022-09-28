import numpy as np
from sims import main
import GPy
import os

    
if __name__ == "__main__":
    """__init__ function to run the main function
    """
    # EOS simulation
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'sim_ml/data'))    
    kernel = GPy.kern.RBF(1) * GPy.kern.Linear(1) * GPy.kern.Poly(1)
    main.eos_active_learning(8,1e-1,25,5,path,100, kernel, 1000)
    
    # main.stress_viscosity_learning()

       
    