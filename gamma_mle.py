# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------------------
# -------- Estimator for shape & dispersion parameters of Gamma distribution ------------
# -------- Python implementation is based on MASS R package MLE estimator ---------------
# ---------------------------------------------------------------------------------------

import numpy as np
import pandas
import statsmodels
from   scipy.special import digamma, polygamma

def gamma_shape_estimator(data, glm_gamma_fit, verbose=False, iter_limit=10, eps_max=1e-4):
    """
    
    A glm fit (from statsmodels) for the Gamma family correctly calculates 
    the maximum likelihood estimate of the mean parameters but provides 
    only a crude estimate of the dispersion parameter. This function takes 
    the results of the glm fit and solves the maximum likelihood equation 
    for the reciprocal of the dispersion parameter, which is usually called 
    the shape (or exponent) parameter.
    
    Parameters
    ----------
    data: array-like          
        NumPy or Pandas vector 
    glm_gamma_fit: statsmodels class instance
        Fitted model instance of glm.fit(scale='dev') of statsmodels module
    
    Local variables
    ---------------
    y: array-like
        Vector of observations
    A: array-like
        Array of weights
    u: array-like
        Mean of GLM predicted values
    Dbar: float
        Scale / dispersion - MUST be similar to scale value from glm.fit(scale='dev')
    iter_limit: integer
        Upper limit on the number of iterations.
    eps_max: float
        Maximum discrepancy between approximations for the iteration process to continue
    verbose: bool
        If TRUE, causes successive iterations to be printed out. The initial estimate 
        is taken from the deviance.
        
    Returns
    -------
    alpha: float
        Shape parameter of gamma distribution (the maximum likelihood estimate)  
    dispersion: float
        Dispersion parameter of gamma distribution
    SE: float
        The approximate standard error, the square-root of the reciprocal of the observed 
        information.
    
    References
    ----------
    Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with S. 
    Fourth edition. Springer.
    
    """
    
    if isinstance(data, (np.ndarray, pandas.core.series.Series)): 
       y = data    # glm_gamma_fit.endog --- TODO as soon as statsmodels is ready for that
    else:
       raise TypeError('data must be np.ndarray or pandas.core.series.Series, not {}'.\
                       format(type(data)))
    
    A       = np.ones(len(data)) # !!! glm_gamma_fit.data_weights --- TODO as soon as statsmodels is ready for that
    
    if isinstance(glm_gamma_fit, statsmodels.genmod.generalized_linear_model.GLMResultsWrapper):
       u    = glm_gamma_fit.mu
       Dbar = glm_gamma_fit.scale
    else:
       raise TypeError('glm_gamma_fit must be statsmodels...GLMResultsWrapper, not {}'.\
                       format(type(glm_gamma_fit)))
    
    alpha   = (6. + 2.*Dbar) / (Dbar*(6. + Dbar))
    
    y[y==0] = 1.
    fixed   = -y/u - np.log(u) + np.log(A) + np.ones(len(A)) + np.log(y)
    eps     = 1.
    itr     = 0
    
    if verbose: print("Initial estimate: {}".format(alpha))
    
    while np.abs(eps) > eps_max and itr <= iter_limit:
        
        itr   = itr + 1
        sc    = np.sum(A * (fixed + np.log(alpha) - digamma(A * alpha)))
        inf   = np.sum(A * (A * polygamma(1., A * alpha) - 1./alpha))
        eps   = sc/inf
        alpha = alpha + eps
        
        if verbose: print("Iter. {} Alpha: {}".format(itr, alpha))
    
    if itr > iter_limit: print("Iteration limit reached")
    
    dispersion = 1./alpha
    
    return alpha, dispersion, np.sqrt(1./inf)

#if __name__ == '__main__':
#   a, d, s = gamma_shape_estimator(data=, glm_gamma_fit=)
