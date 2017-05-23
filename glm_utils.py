# -*- coding: utf-8 -*-

def enso_glm_model(glm_ds, leadtime=False, verbose=True):
    """
    
    Perform GLM (with Gamma link function) fit to SST data and 
    compute parameters of corresponding gamma distribution 
    
    Parameters
    ----------
    glm_ds   : array-like
               pandas dataframe with a set of covariates 
               (principal components of SST-sea surface temperature and 
               TCD-thermocline depth)
    leadtime : bool, optional
               used to include extra 4 lagged covariances to
               assess prediction performance 
    verbose  : bool, optional
               output extra information related to fitting and 
               Gamma distribuitn parameters
     
    Return
    ------
    glm_model : statsmodels instance the fit to SST data
    
    gamma_scalevec, gamma_shapevec : array-like
                scale and shape vectors of Gamma distribution
    
    Example
    -------
    
    """
    # -------- Build GLM model for NINO regions ----------------------------------------------
    import numpy as np
    import statsmodels.api as sm
    from gamma_mle import gamma_shape_estimator

    endog = glm_ds['sst']
    
    if leadtime:
        exog = glm_ds[['pc1_sst', 'pc2_sst', 'pc3_sst', 'pc1_tcd', 'pc2_tcd', 'pc3_tcd',\
                       'sc_sin_jn_lag1', 'sc_sin_jn_lag2', 'sc_sin_jn_lag3', 'sc_sin_jn_lag4',\
                       'sc_cos_jn_lag1', 'sc_cos_jn_lag2', 'sc_cos_jn_lag3', 'sc_cos_jn_lag4']]        
    else:
        exog = glm_ds[['pc1_sst', 'pc2_sst', 'pc3_sst', 'pc1_tcd', 'pc2_tcd', 'pc3_tcd',\
                       'sc_sin_jn_lag1', 'sc_cos_jn_lag1']]
    
    exog = sm.add_constant(exog, prepend=True)
    
    glm = sm.GLM(endog, exog, family=sm.families.Gamma(sm.families.links.log))
    
    # Dispersion parameter (Psi) = 1 / shape parameter (alpha)
    # Psi is estimated from the sum of squares of the Pearson residuals
    # Residual deviance is very sensitive to small values y_i
    
    glm_model = glm.fit(scale='dev') # scale='dev' !!!!!
    
    if verbose:
        print(glm_model.summary())
        print('Akaike Information Criterion: {}'.format(glm_model.aic))
        print('Bayes Information Criterion: {}\n'.format(glm_model.bic))
        print('Confidence interval:')
        print(glm_model.conf_int())
        print('Odds ratios:')
        print(np.exp(glm_model.params))
    
    # -------- MLE estimator for gamma distribution -----------------------------------------
    if verbose: print('\nMLE estimation for Gamma distribution:\n')
    
    fit_alfa, fit_dispersion, se = gamma_shape_estimator(endog, glm_model)
    
    # Model fit must use the following option: scale='dev' - glm.fit(scale='dev') 
    # in order to get correct MLE estimate
    
    print('Alfa: {}, Dispersion: {}, SE: {}'.format(fit_alfa, fit_dispersion, se))
    
    gamma_scalevec = np.asarray(glm_model.mu) / fit_alfa
    gamma_shapevec = np.ones(len(endog)) * fit_alfa
    
    return glm_model, gamma_scalevec, gamma_shapevec

def generate_enso_intensity(gamma_scalevec, gamma_shapevec):
    """
    Generate SST anomaly from fit and derived parameters (from enso_glm_model routine)

    Parameters
    ----------
    gamma_scalevec, gamma_shapevec : array-like
                scale and shape vectors of Gamma distribution

    Return
    ------
    array-like with SST 

    Example
    -------
    
    """
    import numpy as np
    
    n = len(gamma_scalevec)
    return np.asarray([np.random.gamma(shape = gamma_shapevec[i], scale = gamma_scalevec[i]) for i in range(n)])
