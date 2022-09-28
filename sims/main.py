import os
import numpy as np
import matplotlib.pyplot as plt
from sims import eos

def eos_active_learning(save_every,threshold,time_steps,N_init,path,N, kernel, iter_num):
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
    