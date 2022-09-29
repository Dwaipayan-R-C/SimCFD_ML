import numpy as np
from utils import TFuncs
from utils import GPLearn
from utils import plots


def active_learning_visc(X_grid,index_list,threshold,Number_of_iterations,kernel, scale, x_label, y_label ):
    """Active learning code to execute the Active learning

    Args:
        X_grid (array): X data for prediction
        index_list (array): random integers stored for sampling
        threshold (double): cut-off variance
        Number_of_iterations (int): Iteration number to allocate space for Y estimates
        kernel (GPy.kernel): Kernel that decides the Cov matrix
        scale (bool): linear/logarithmic
        x_label (string): x label
        y_label (string): y label
    """
    N_init = len(index_list)
    
    X_samples = X_grid[index_list]
    print(X_samples.shape)

    Y_samples = TFuncs.target_stress(X_samples,0.5,2,.5,.2,4)
    Y_grid = TFuncs.target_stress(X_grid, 0.5,2,.5,.2,4)   
        
    # pseudo 2D shape for GPs
    X_samples = np.array(X_samples)[:, None]
    Y_samples = np.array(Y_samples)[:, None]
    X_grid = np.array(X_grid)[:, None]
    
    # We will use this matrix to store the GP mean at every iteration.
    Y_estimates = np.full((len(X_grid), Number_of_iterations + 1), np.nan)
    varY_estimates = np.full((len(X_grid), len(X_grid), Number_of_iterations + 1), np.nan)

    # GP regression
    mean, Cov, variance, m = GPLearn.GP_analysis_viscosity(X_samples, Y_samples, X_grid,kernel)
    
    # Store the results to plot later
    Y_estimates[:, 0] = mean.ravel()
    varY_estimates[:, :, 0] = Cov
    num_list=[]
    i = 0
    while(np.max(variance)>threshold):
        
        # find the next sample
        next_sample_index = np.argmax(variance)
        print(np.max(variance))

        index_list.append(next_sample_index)
        next_sample_loc = X_grid[next_sample_index]
        next_sample_value = TFuncs.target_stress(next_sample_loc, 0.5,2,.5,.2,4)        

        # add the desired sample to our data
        X_samples = np.vstack((X_samples, X_grid[next_sample_index, :]))
        Y_samples = np.vstack((Y_samples, np.array([next_sample_value])[0, None]))

        # GP regression
        mean, Cov, variance, m = GPLearn.GP_analysis_viscosity(X_samples, Y_samples, X_grid, kernel)  

        Y_estimates[:, i+1] = mean.ravel()
        varY_estimates[:, :, i+1] = Cov
        if(i%2==0):
            num_list.append(i)
        i = i+1        
    Number_of_iterations = i
    index_list.append(0)
    
    # plots
    plots.plot_summary_visc(Number_of_iterations ,
                 X_grid,
                 X_samples,
                 Y_samples,
                 index_list,
                 Y_estimates,
                 varY_estimates, Y_grid, N_init, num_list,
                 scale, x_label, y_label)