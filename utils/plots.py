import matplotlib.pyplot as plt
import numpy as np
from utils import TFuncs
import os


def plot_gp(ax, X, m, C, no_last_data, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval"""
    # Plot 95% confidence interval
    ax.fill_between(X[:, 0],
                    m - 1.96*np.sqrt(np.diag(C)),
                    m + 1.96*np.sqrt(np.diag(C)),
                    alpha=0.5)
    # Plot GP mean and initial training points
    ax.plot(X, m, "-", label="Predicted GP mean")
    # plt.show()
    # Plot training points if included
    if no_last_data:
        if training_points is not None:
            X_, Y_, varY_ = training_points
            l, _, _ = ax.errorbar(X_[:, 0], Y_[:, 0], yerr=np.sqrt(varY_[:, 0]),
                                  ls="",
                                  marker="o",
                                  markersize=5,
                                  color="red")
            l.set_label("Training points")
    return ax.get_lines()


def plot_AL_iteration(ax, X_grid, mean, Cov, alpha_full, X_samples, Y_samples, Y_var, next_sample, last, sample=True, no_last_data=True):

    l1 = plot_gp(ax, X_grid, mean, Cov, no_last_data, training_points=(X_samples, Y_samples, Y_var))
    # ax2 = ax.twinx()
    ax.set_ylabel(r"pressure $p$")
    ax.set_xlabel(r"density $\rho$")
    total_data = np.shape(X_samples)[0]
    # ax2.set_ylabel(r"var($p$)", color='r')
    # ax2.tick_params(axis='y', labelcolor='r')
    edited_alpha = '{:.2e}'.format(np.max(alpha_full))
    if sample:
        ax.text(0.33, 0.8, f'Maximum variance = {edited_alpha} \n Total data points = {total_data}',
                ha='center', va='center', transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='red', alpha=0.5))
    # l2 = ax2.plot(X_grid, alpha_full, 'r', label="Aquisition function")
    if not last:
        l3 = ax.plot([X_grid[next_sample], X_grid[next_sample]], ax.get_ylim(), 'g--', label="Next sample")
    else:
        l3 = ax.plot([], [], 'g--', label="Next sample")

    lns = l1 + l3
    # lns = l1 + l2 + l3
    return lns


def plot_summary(path, N_init, N, X_grid, X_samples, Y_samples,
                 index_list, Mean, Cov, fig1, ax1, start_data=0, x_yes=True, no_last_data=True):

    last = True
    global_error = []
    mean = Mean
    cov = Cov
    # axis = ax1.flat[N-1]
    alpha_full = np.diag(cov)
    global_error.append(np.linalg.norm(alpha_full))

    if not(x_yes):
        if not(no_last_data):
            Y_var = np.zeros_like(Y_samples)
            lns = plot_AL_iteration(ax1, X_grid, mean, cov, alpha_full,
                                    X_samples, Y_samples, Y_var, index_list, last=last, sample=False, no_last_data=False)
        else:
            Y_var = np.zeros_like(Y_samples)
            lns = plot_AL_iteration(ax1, X_grid, mean, cov, alpha_full,
                                    X_samples, Y_samples, Y_var, index_list, last=last, sample=False, no_last_data=True)
    else:
        Y_var = np.zeros_like(Y_samples[start_data:N_init+N])
        lns = plot_AL_iteration(ax1, X_grid, mean, cov, alpha_full,
                                X_samples[start_data:N_init+N], Y_samples[start_data:N_init+N], Y_var, index_list[N_init+N], last=last)

    ax1.plot(X_grid, TFuncs.target_function(X_grid, path), '--', color='0.0')


def plot_gp_visc(ax, X, m, C, scale, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    if(scale == 'log'):
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.fill_between(X[:, 0],
                    m - 1.96*np.sqrt(np.diag(C)),
                    m + 1.96*np.sqrt(np.diag(C)),
                    alpha=0.5)
    # Plot GP mean and initial training points
    ax.plot(X, m, "-", label="Predicted GP mean")

    # Plot training points if included
    if training_points is not None:
        X_, Y_, varY_ = training_points

        l, _, _ = ax.errorbar(X_[:, 0], Y_[:, 0], yerr=np.sqrt(varY_[:, 0]),
                              ls="",
                              marker="x",
                              markersize=6,
                              color="0.0")
        l.set_label("Training points")

    return ax.get_lines()


def plot_AL_iteration_visc(ax, fig1, X_grid, mean, Cov, alpha_full,
                           X_samples, Y_samples, Y_var, next_sample, last, scale, x_label, y_label):

    l1 = plot_gp_visc(ax, X_grid, mean, Cov, training_points=(X_samples, Y_samples, Y_var), scale=scale)

    ax2 = ax.twinx()
    if(scale == 'log'):
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_ylabel(y_label)
    # ax.set_ylabel(r"Shear viscosity η ")
    ax.set_xlabel(x_label)
    ax2.set_ylabel(r"variance ", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    l2 = ax2.plot(X_grid, alpha_full, 'r', label="Aquisition function")

    if not last:
        l3 = ax.plot([X_grid[next_sample], X_grid[next_sample]], ax.get_ylim(), 'g--', label="Next sample")
    else:
        l3 = ax.plot([], [], 'g--', label="Next sample")

    lns = l1 + l2 + l3

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'results'))
    save_path = os.path.join(path, f'al_rbf_lin_{scale}.png')
    fig1.savefig(save_path, pad_inches=1)
    return lns


def plot_summary_visc(N, X_grid, X_samples, Y_samples, index_list,
                      Mean, Cov, target_function, N_init, num_list, scale, x_label, y_label, ar=1.61, zoom=3.5):

    # create subplot grid
    Nx, Ny = (len(num_list) // 2 + len(num_list) % 2, 2)
    fig1, ax1 = plt.subplots(Nx, Ny, figsize=(Ny*ar*zoom, Nx*zoom), sharex=True)
    fig2, ax2 = plt.subplots(1, figsize=(ar*zoom, zoom))
    last = False

    global_error = []
    for i in range(len(num_list)):
        if i == N - 1:
            last = True

        axis = ax1.flat[i]
        mean = Mean[:, num_list[i]]
        cov = Cov[:, :, num_list[i]]

        alpha_full = np.diag(cov)

        global_error.append(np.linalg.norm(alpha_full))

        Y_var = np.zeros_like(Y_samples[:N_init+num_list[i]])

        lns = plot_AL_iteration_visc(axis, fig1, X_grid, mean, cov, alpha_full,
                                     X_samples[:N_init+num_list[i]], Y_samples[:N_init+num_list[i]], Y_var, index_list[N_init+num_list[i]],
                                     last=last, scale=scale, x_label=x_label, y_label=y_label)

        axis.plot(X_grid, target_function, '--', color='0.0')

    labs = [line.get_label() for line in lns]
    ax1.flat[0].legend(lns, labs, loc="lower center", bbox_to_anchor=(1., 1.), ncol=5, frameon=False)

    # ax2.plot(np.arange(N), global_error, '-')
    ax2.set_xlabel("iteration")
    ax2.set_ylabel(r"$||\mathsf{var}(f)||$")
