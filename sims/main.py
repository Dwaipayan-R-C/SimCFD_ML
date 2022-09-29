import os
import numpy as np
import matplotlib.pyplot as plt
from sims import eos
from sims import viscLearn
import GPy

def eos_active_learning(save_every,threshold,time_steps,N_init,path,N, kernel, iter_num):
    """Main function for Active learning of equation of state

    Args:
        save_every (int): Intervals to save plots
        threshold (double): variance allowance
        time_steps (int): last temperature profile count
        N_init (int): initial no. of training data points
        path (string): path of the data folder
        N (int): resolution
        kernel (GPy.Kern): Kernel that decides the Cov and mean
        iter_num (int): to save space for Y_Var
    """
    data_path_max = f"{path}\\densitiy-vs-x\\density_t{time_steps:02d}.dat"
    data_path_min = f"{path}\\densitiy-vs-x\\density_t{0:02d}.dat"
    x, dens = np.loadtxt(data_path_max, unpack=True)
    density_max = np.max(dens)
    x, dens = np.loadtxt(data_path_min, unpack=True)
    density_min = np.min(dens)
    xhi, xlo = density_min,np.max(dens)
    exp_resolution = 5   
    fig = eos.eos_al_func(save_every, threshold,time_steps,N_init,path,N,kernel,iter_num,density_min,density_max,xlo,xhi, exp_resolution)
    path = os.path.join(path,'results\\eos_al.png')
    os.makedirs = True
    fig.savefig(path)
    
    
def stress_viscosity_learning(x_low,x_high,N_init,N,iter_num,threshold):
    """Shear stress vs shear rate main function to create initial setup
    for active learning.

    Args:
        x_low (double): Lower limit of x scale
        x_high (double): Higher limit of x scale
        N_init (int): number of initial sample space
        N (int): Grid resolution
        iter_num (int): Iteration number to allocate space for Y estimates
        threshold (double): Tolerance in max. variance
    """
    
    X_grid = np.linspace(x_low, x_high, N)
    kernel = GPy.kern.RBF(1) * GPy.kern.RatQuad(1)
    scale = 'linear'
    x_label = r"shear stress"
    y_label = 'shear rate'
    
    index_list = list(np.concatenate((np.array([x_low]), np.random.randint(x_low+1, N//2, N_init))))
    viscLearn.active_learning_visc(X_grid,index_list,threshold,iter_num,kernel, scale, x_label, y_label)