# Skeleton
This project is developed during a research assistant position at [IMTEK Simulation Department](https://www.imtek.uni-freiburg.de/professuren/simulation/simulation) under the supervision of Hannes Holey (PhD. student at [Karlsruhe Institute of Technology](https://www.kit.edu/kit/english/index.php)). The project is mainly aimed to implement ML predictions in Tribology. We use [Numpy](https://numpy.org/), [Matplotlib](https://matplotlib.org/) and [GPy](https://gpy.readthedocs.io/en/deploy/) for the computation. 

## Folder Structure
    .
    ├── _init_                            # contains two simulation entry point
    │   ├── data                          # contains all the data files and results   
    │   ├── sims                          # contains all simulation files and plot files
    │   ├── utils                         # contails all the supporting files for the project like plots      
    │   |   └── ...                                
    │   └── ...
    └── ...     

## Simulations:
1. Stress vs shear rate analysis:<br/>
We are predicting the Newtonian fluid stress to shear rate function with some training points. 
<p align="center">  
  <img src="/data/results/al_rbf_lin_linear.png" width="300"/>
</p>
2. Equation of state:<br/>
We are predicting the equation of state from the training data generated using continuum solver and MD. This also predicts the posituions where MD data is relevant. 
<p align="center">  
  <img src="/data/results/eos_al.png" width="300"/>
</p>
