import numpy as np
import numpy.ma as ma
import rootsolve as rs


def osc_prob(theta, dmsq, energy, length):
    ''' Returns the probability that a muon will have not oscillated out of 
    the muon eigenstate given its energy and length traversed.
    '''
    
    return(1-(np.sin(2*theta)**2)*(np.sin(1.267*dmsq*1e-3*length/energy)**2))

    
def nll(theta, dmsq, alpha, k, flux, energy, length):
    ''' Returns the negative log likelihood for the given parameters
    theta, dmsq and alpha matching the given measured values of lambda, k.
    
    In cases where we do not wish to include the effects of cross section
    changing withe energy, we can simply set alpha to be the reciprocal
    of the energy array such that the alpha*energy factor becomes an array
    of ones. 
    '''
    
    lmda_p = lmda(theta, dmsq, alpha, flux, energy, length)
    return(np.sum(lmda_p - k + k*ma.log(k/lmda_p)))


def nll_4d(theta, dmsq, alpha, beta, k, flux, energy, length):
    ''' NLL function which includes an additional parameter beta which 
    controls the offset of the linear cross section energy dependence.
    '''
    
    lmda_p = lmda_4d(theta, dmsq, alpha, beta, flux, energy, length)
    return(np.sum(lmda_p - k + k*ma.log(k/lmda_p)))
    
    
def lmda_4d(theta, dmsq, alpha, beta, flux, energy, length):
    ''' Does the same as the lmda() function except allowes for an aditional
    parameter beta which controlls the offset of the linear cross section
    energy depenence. It is to be used in conjunction with nll_4d only. 
    '''
    
    return(osc_prob(theta, dmsq, energy, length)*flux*(alpha*energy + beta))
    
    
def lmda(theta, dmsq, alpha, flux, energy, length):
    ''' Returns the predicted mean value of events which will be observed
    in an energy bin centred at 'energy'.
    '''
    
    return(osc_prob(theta, dmsq, energy, length)*flux*alpha*energy)


def nll_theta(theta, dmsq, alpha, k, flux, energy, length):
    ''' Helper function for when theta is being varied. '''
    
    return(nll(theta, dmsq, alpha, k, flux, energy, length))

    
def nll_dmsq(dmsq, theta, alpha, k, flux, energy, length):
    ''' Helper function for when dmsq is being varied. '''
    
    return(nll(theta, dmsq, alpha, k, flux, energy, length))

    
def nll_alpha(alpha, theta, dmsq, k, flux, energy, length):
    '' 'Helper function for when alpha is being varied. '''
    
    return(nll(theta, dmsq, alpha, k, flux, energy, length))

    
def nll_theta_shifted(theta, dmsq, alpha, k, flux, energy, length, nll_min):
    ''' '''
    return(np.abs(nll_theta(theta, dmsq, alpha, k, flux, energy, length) - nll_min) - 0.5)

    
def nll_dmsq_shifted(dmsq, theta, alpha, k, flux, energy, length, nll_min):
    return(np.abs(nll_dmsq(dmsq, theta, alpha, k, flux, energy, length) - nll_min) - 0.5)

    
def nll_alpha_shifted(alpha, theta, dmsq, k, flux, energy, length, nll_min):
    return(np.abs(nll_alpha(alpha, theta, dmsq, k, flux, energy, length) - nll_min) - 0.5)

    
def nll_theta_err(theta_min, params):
    nll_min = nll_theta(theta_min, *params)
    params = *params, nll_min
    theta_min_pos = rs.root_find_secant(nll_theta_shifted, params, 
                                        theta_min + 0.01)
    theta_min_neg = rs.root_find_secant(nll_theta_shifted, params, 
                                        theta_min - 0.01)
    return(theta_min_neg, theta_min_pos)

    
def nll_err_2d(theta_min, dmsq_min, params):
    nll_min = nll(theta_min, dmsq_min, *params)
    theta_params = dmsq_min, *params, nll_min
    dmsq_params = theta_min, *params, nll_min
    theta_min_pos = rs.root_find_secant(nll_theta_shifted, theta_params, 
                                        theta_min + 0.01)
    theta_min_neg = rs.root_find_secant(nll_theta_shifted, theta_params, 
                                        theta_min - 0.01)
    dmsq_min_pos = rs.root_find_secant(nll_dmsq_shifted, dmsq_params, 
                                        dmsq_min + 0.01)
    dmsq_min_neg = rs.root_find_secant(nll_dmsq_shifted, dmsq_params, 
                                        dmsq_min - 0.01)
    return(theta_min_neg, theta_min_pos, dmsq_min_neg, dmsq_min_pos)
