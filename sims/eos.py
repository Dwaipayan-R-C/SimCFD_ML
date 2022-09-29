import os
import numpy as np
import matplotlib.pyplot as plt
from utils import TFuncs
from utils import plots
from utils import GPLearn

def eos_al_func(save_every,threshold,time_count,N_init,path,N, kernel, iter_num, density_min,density_max,xlo,xhi, exp_resolution,ar=1.61,zoom=3.5 ):
    var_list = []        
    fig, axes = plt.subplots(time_count//save_every+1, 3, figsize=(11,6))
    dirname = os.path.join(path, 'densitiy-vs-x')  
   
    X_grid = np.linspace(xlo,xhi, N)    
    index_list = list( np.random.randint(0, 100, N_init))    
    X_samples = X_grid[index_list]
    Y_samples = TFuncs.target_function(X_samples, path)

    # Sample
    X_samples = np.linspace(xlo,xhi,N_init)
    Y_samples = np.array(TFuncs.modified_BWR(X_samples, 2, path))[:, None]
    X_samples = X_samples[:, None]
    X_grid = np.array(X_grid)[:, None]    

    # Number of GP iterations
    iter_num = 1000    
    # We will use this matrix to store the GP mean at every iteration.
    
    Y_estimates, varY_estimates = [],[]
    # GP regression
    mean, Cov, variance, m = GPLearn.GP_analysis(X_samples, Y_samples, X_grid,kernel)
    low_y = np.min(mean)
    # Store the results to plot later    
    i = 0
    start_extrapolation_index = 0
    x_grid_last = 0
    
    for time_step in range(time_count):
        ax2 = axes[time_step//save_every][1]          
        if(time_step!=0):
            mean, Cov, variance, m = GPLearn.GP_analysis(X_samples, Y_samples, X_grid,kernel) 
        print(np.max(variance))
        while(np.max(variance)>threshold):   
            var_list.append(np.max(variance))                 
            # find the next sample
            next_sample_index = np.argmax(variance)
            index_list.append(next_sample_index)
            next_sample_loc = X_grid[next_sample_index]
            next_sample_value = TFuncs.target_function(next_sample_loc, path)
            
            # add the desired sample to our data
            X_samples = np.vstack((X_samples, X_grid[next_sample_index, :]))
            Y_samples = np.vstack((Y_samples, np.array([next_sample_value])[0, None]))

            # GP regression
            mean, Cov, variance, m = GPLearn.GP_analysis(X_samples, Y_samples, X_grid, kernel)
            i = i+1        
        iter_num = i
        Y_estimates = mean.ravel()
        varY_estimates = Cov  
        index_list.append(0)

        # Subplots  
        if(time_step%save_every==0):            
            ax = axes[time_step//save_every][2]
            for plot_loop in range(0,time_step+1): 
                x, dens = np.loadtxt(f"{dirname}\\density_t{plot_loop:02d}.dat", unpack=True)
                ax.plot(x, dens)
            data_time = '{:.2e}'.format((time_step+1)*.0002)            
            ax.text(.2, 0.8,f'Timestep = {time_step+1}', ha='center', va='center', transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='red', alpha=0.5))
            ax.set_xlabel('Distance')
            ax.set_ylabel(r'Density $\rho$')
            ax2.grid()  
            ax.grid()
            ax.set_ylim([density_min, density_max]) 
            ax1 = axes[time_step//save_every][0]  
            ax1.grid()
            ax1.set_xlim([density_min, density_max])
            x_sample_new = []
            y_sample_new = []
            no_last_data=True
            for j in range(start_extrapolation_index,np.shape(X_samples)[0]):
                if(X_samples[j]>=X_grid[x_grid_last,0]):                    
                    x_sample_new.append(X_samples[j])
                    y_sample_new.append(Y_samples[j])    
            if (len(x_sample_new)!=0):            
                x_sample_new = np.array(x_sample_new)
                y_sample_new = np.array(y_sample_new)      
            else:
                no_last_data = False
                    
            plots.plot_summary(path,N_init,iter_num ,
                        X_grid[x_grid_last:],
                        x_sample_new,
                        y_sample_new,
                        index_list,
                        Y_estimates[x_grid_last:],
                        varY_estimates[x_grid_last:],fig, ax2,start_extrapolation_index,False, no_last_data )
            plots.plot_summary(path,N_init,iter_num,
                        X_grid,
                        X_samples,
                        Y_samples,
                        index_list,
                        Y_estimates,
                        varY_estimates,fig, ax1)
            
            x_grid_last = np.shape(X_grid)[0]-1
            start_extrapolation_index = np.shape(X_samples)[0]-1
        x, dens = np.loadtxt(f"{dirname}\\density_t{time_step+1:02d}.dat", unpack=True) 
        exp_region = np.linspace(X_grid[-1][0],np.max(dens),10)
        X_grid = np.concatenate((X_grid,exp_region[:,None]))
    print(len(var_list))        
    axes[0][0].set_ylim([low_y-5000,np.max(Y_estimates)]) 
    fig.tight_layout()
    return fig
    
    