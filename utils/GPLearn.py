import numpy as np
import GPy


def GP_analysis(X, Y, X_grid, k):
    """Runs the GPR model and returns mean, Cov, Var,m"""
    k = GPy.kern.RBF(1) * GPy.kern.Linear(1) * GPy.kern.Poly(1)
    m = GPy.models.GPRegression(X, Y, k)
    m.optimize('bfgs', max_iters=100)
    # Predict the mean and covariance of the GP fit over the grid
    mean, Cov = m.predict(X_grid, full_cov=True)
    variance = np.diag(Cov)
    return mean, Cov, variance, m


def GP_analysis_viscosity(X, Y, X_grid, k):
    """Runs the GPR model and returns mean, Cov, Var,m"""
    k = GPy.kern.RBF(1) * GPy.kern.RatQuad(1)
    m = GPy.models.GPRegression(X, Y, k)
    m.optimize('bfgs', max_iters=100)
    # Predict the mean and covariance of the GP fit over the grid
    mean, Cov = m.predict(X_grid, full_cov=True)
    variance = np.diag(Cov)
    return mean, Cov, variance, m
