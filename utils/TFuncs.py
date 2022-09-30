import numpy as np
import os


def modified_BWR(rho, T, path, gamma=3., param="MP"):
    """
    Modified Benedict-Webb-Rubin equation of state.
    Parameters
    ----------
    rho : float or array
        Non-dimensional density.
    T : float or array
        Non-dimensional temperature.
    gamma : float
        Nonlinear fitting parameter (the default is 3.).
    param : str
        Choice of fitting parameters.
        Two different textfiles for the 32 fitting parameters can be used:
        "MP": Johnson et al., Mol. Phys. 78 (1993)
        "PRE": May and Mausbach, PRE 85 (2012) (+ Erratum).
        (The default is "MP")
    Returns
    -------
    float or array
        Non-dimensional pressure.
    """

    if param == "MP":
        x = np.loadtxt(os.path.join(path, "mbwr_MolPhys78.dat"))
    elif param == "PRE":
        x = np.loadtxt(os.path.join(path, "mbwr_PRE85.dat"))
    # x = np.multiply(x,2.5)
    p = rho * T +\
        rho**2 * (x[0] * T + x[1] * np.sqrt(T) + x[2] + x[3] / T + x[4] / T**2) +\
        rho**3 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2) +\
        rho**4 * (x[9] * T + x[10] + x[11] / T) +\
        rho**5 * x[12] +\
        rho**6 * (x[13] / T + x[14] / T**2) +\
        rho**7 * (x[15] / T) +\
        rho**8 * (x[16] / T + x[17] / T**2) +\
        rho**9 * (x[18] / T**2) +\
        np.exp(-gamma * rho**2) * (rho**3 * (x[19] / T**2 + x[20] / T**3) +
                                   rho**5 * (x[21] / T**2 + x[22] / T**4) +
                                   rho**7 * (x[23] / T**2 + x[24] / T**3) +
                                   rho**9 * (x[25] / T**2 + x[26] / T**4) +
                                   rho**11 * (x[27] / T**2 + x[28] / T**3) +
                                   rho**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4))
    return p


def Calculate_RMSE(mean, x_val):
    y_real = np.sin(x_val[:, 0])
    diff_val = np.subtract(mean[:, 0], y_real)
    sq_diff_val = np.square(diff_val)
    mean_sq_diff_val = np.mean(sq_diff_val)
    rmse = np.sqrt(mean_sq_diff_val)
    return rmse


def target_stress(gamma, n, eta_0, eta_inf, lamda, a):
    """
    x: shear rate
    eta_N: Newtonian viscosity
    lamda: relaxation time
    """
    # return
    stress = (eta_inf + (eta_0 - eta_inf)*(1+(lamda*gamma)**a)**((n-1)/1))*gamma
    return stress


def target_function(x, path):
    """_summary_

    Args:
        x (array): input data
        path (string): path of the MD data

    Returns:
        list: output from the MD simulation
    """
    pressure = modified_BWR(x, 2, path)
    return pressure
