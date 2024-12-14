import numpy as np
from sparse_multiplication import compute_Xb, compute_MXt

def get_objective(X, Y, s):
    """
    Get objective function value from data and model parameters
    
    Parameters:
    X: ndarray, covariate matrix
    Y: ndarray, response vector
    s: dict, dictionary containing model parameters including KL, sigma2, alpha, mu, mu2, Xr
    """
    return Eloglik(X, Y, s) - np.sum(s.KL)

def Eloglik(X, Y, s):
    """
    Expected log-likelihood for the fit
    """
    n = X.shape[0]
    return -(n/2) * np.log(2*np.pi*s.sigma2) - (1/(2*s.sigma2)) * get_ER2(X, Y, s)

def get_ER2(X, Y, s):
    """
    Expected squared residuals
    """
    # Compute L x N matrix
    Xr_L = compute_MXt(s.alpha * s.mu, X)
    # Posterior second moment
    postb2 = s.alpha * s.mu2
    
    return np.sum((Y - s.Xr)**2) - np.sum(Xr_L**2) + np.sum(X.d[:, np.newaxis] * postb2.T)

def SER_posterior_e_loglik(X, Y, s2, Eb, Eb2):
    """
    Posterior expected log-likelihood for single effect regression
    
    Parameters:
    X: ndarray, n x p matrix of covariates
    Y: ndarray, n-dimensional response vector
    s2: float, residual variance
    Eb: ndarray, posterior mean of b (p-dimensional vector)
    Eb2: ndarray, posterior second moment of b (p-dimensional vector)
    """
    n = X.shape[0]
    return -0.5*n*np.log(2*np.pi*s2) - 0.5/s2*(np.sum(Y*Y) 
                                               - 2*np.sum(Y*compute_Xb(X, Eb))
                                               + np.sum(X.d * Eb2))


def estimate_residual_variance(X, y, s):
    """
    Estimate residual variance
    
    Parameters:
    X: ndarray, n by p matrix of covariates
    y: ndarray, n-dimensional vector of data
    s: dict, susie fit object containing model parameters
    
    Returns:
    float: estimated residual variance
    """
    n = X.shape[0]
    return (1/n) * get_ER2(X, y, s)