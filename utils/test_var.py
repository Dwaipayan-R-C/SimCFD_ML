import numpy as np
import matplotlib.pyplot as plt


var_list = [1.00E-01,1.00E-02,1.00E-03]
run_time_list = [11,17,22]
data_points = [16,22,27]

fig, axes = plt.subplots(1, 2,figsize=(11,4))    
axes[0].plot(var_list,run_time_list,'b-o')
axes[0].set_xlabel('Threshold value')
axes[0].set_ylabel('No. of runs')
axes[0].set_xscale('log')
axes[0].grid(True,which="both")

axes[1].plot(var_list,data_points, 'r-o')
axes[1].set_xlabel('Threshold value')
axes[1].set_ylabel('Total data points')
axes[1].set_xscale('log')
axes[1].grid(True,which="both")

plt.tight_layout()
fig.show()
print()