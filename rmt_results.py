# This file contains all the implementations of our theoretical results found using RMT
import numpy as np
from utils import *
import scipy.integrate as integrate
import utils
from scipy.optimize import minimize

# Need to test these functions before usage
def Delta(eta, gamma):
    return (eta - gamma - 1 + np.sqrt((eta - gamma - 1)**2 + 4*eta*gamma)) / (2 * gamma)

def Q_bar(vmu, delta, gamma):
    p = len(vmu)
    mu = np.sqrt(np.sum(vmu**2))
    r = (1 + delta) / (1 + gamma * (1 + delta))
    M = np.eye(p) - np.outer(vmu, vmu) / (mu**2 + 1 + gamma * (1 + delta))
    return r * M

def test_expectation_naive(n, p, pi, epsp, epsm, mu, gamma):
    eta = p/n
    delta = Delta(eta, gamma)
    pi_1 = pi
    pi_2 = 1 - pi
    return (1 - 2*pi_1*epsm - 2*pi_2*epsp)*(mu**2) / (mu**2 + 1 + gamma * (1 + delta))

def test_expectation_imp(n, p, pi, epsp, epsm, rhop, rhom, mu, gamma):
    eta = p/n
    delta = Delta(eta, gamma)
    pi_1 = pi
    pi_2 = 1 - pi_1
    beta = 1 / (1 - rhop - rhom)
    lambda_m = (1 - rhop + rhom) * beta
    lambda_p = (1 - rhom + rhop) * beta
    return (mu**2)*(pi_1 * (lambda_m - 2 * beta * epsm) + pi_2 * (lambda_p - 2 * beta * epsp)) / (mu**2 + 1 + gamma * (1 + delta))

# E[g_\rho(x)]
def test_expectation(classifier, n, p, pi, epsp, epsm, rhop, rhom, mu, gamma):
    if 'naive' in classifier :
        return test_expectation_naive(n, p, pi, epsp, epsm, mu, gamma)
    elif 'improved' in classifier :
        return test_expectation_imp(n, p, pi, epsp, epsm, rhop, rhom, mu, gamma)
    else: # oracle
        return test_expectation_naive(n, p, pi, 0, 0, mu, gamma)

# h and Tr((\Sigma \bar \rmQ )^2)
def trace_sigma_q(p, gamma, delta):
    return p*(1 + delta)**2 / (1 + gamma * (1 + delta))**2

def denom(gamma, p, n):
    delta = Delta(p/n, gamma)
    return 1 -  trace_sigma_q(p, gamma, delta) / (n * (1 + delta)**2)

def test_expectation_2_naive(pi, n, p, epsp, epsm, mu, gamma):
    # Constants
    pi_1 = pi
    pi_2 = 1 - pi_1
    eta = p/n
    delta = Delta(eta, gamma)
    d = denom(gamma, p, n)
    alpha = mu**2 + 1 + gamma * (1 + delta)
    

    # coefficient 1
    r_1 = (mu * (1 - 2 *pi_1*epsm - 2*pi_2*epsp))**2
    r_1 = r_1 / (d * alpha)
    
    # sum
    s_1 = (mu**2 + 1) / alpha - 2*(1 - d)

    return r_1 * s_1  + (1 - d) / d

def test_expectation_2_imp(pi, n, p, epsp, epsm, rhop, rhom, mu, gamma):
    # Constants
    pi_1 = pi
    pi_2 = 1 - pi_1
    eta = p/n
    delta = Delta(eta, gamma)
    d = denom(gamma, p, n)
    alpha = mu**2 + 1 + gamma * (1 + delta)
    
    # Problem related
    beta = 1 / (1 - rhop - rhom)
    lambda_m = (1 - rhop + rhom) *  beta
    lambda_p = (1 - rhom + rhop) * beta

    # First term
    r_1 =(mu * (pi_1*(2*beta*epsm - lambda_m) + pi_2*(2*beta*epsp - lambda_p)) )**2 
    r_1 = r_1 / (d * alpha)

    s_1 = (mu**2 + 1) / alpha - 2*(1 - d)

    # Second term
    r_2 = pi_1 * (4* beta**2 * epsm *(rhop - rhom) + lambda_m**2)
    r_3 = pi_2 * (4* beta**2 * epsp *(rhom - rhop) + lambda_p**2)

    return r_1 * s_1 + (r_2 + r_3)* (1 - d) / d

# E[g^2s]
def test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma):
    if 'naive' in classifier :
        return test_expectation_2_naive(pi, n, p, epsp, epsm, mu, gamma)
    elif 'improved' in classifier:
        return test_expectation_2_imp(pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    else: # oracle
        return test_expectation_2_naive(pi, n, p, 0, 0, mu, gamma)

# Computing Test accuracy
def test_accuracy(classifier, n, p, epsp, epsm, rhop, rhom, mu, gamma, pi):

    # E[g] and E[g^2]
    mean = test_expectation(classifier, n, p, pi, epsp, epsm, rhop, rhom, mu, gamma)
    expec_2 = test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    std = np.sqrt(expec_2 - mean**2)
    return 1 - integrate.quad(lambda x: utils.gaussian(x, 0, 1), abs(mean)/std, np.inf)[0]

# Computing Test Risk
def test_risk(classifier, n, p, epsp, epsm, rhop, rhom, mu, gamma, pi):

    # E[g] and E(g^2)
    mean = test_expectation(classifier, n, p, pi, epsp, epsm, rhop, rhom, mu, gamma)
    expec_2 = test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    return expec_2 + 1 - 2 * mean

# Optimal rhop
def optimal_rhop(pi, epsp, epsm, rhom = 0):
    pi_1 = pi
    pi_2 = 1 - pi
    num = pi_1**2 * epsm * (epsm - 1) + pi_2**2 * epsp * (1 - epsp)
    den = pi_1 * pi_2 * (1 - epsp - epsm)
    return num / den + rhom

# Optimal rhom
def optimal_rhom(pi, epsp, epsm, rhop = 0):
    pi_1 = pi
    pi_2 = 1 - pi
    num = pi_1**2 * epsm * (1 - epsm) + pi_2**2 * epsp * (epsp - 1)
    den = pi_1 * pi_2 * (1 - epsp - epsm)
    return num / den + rhop

def optimal_rhos(pi, epsp, epsm):
    # Returns rhop, rhom
    rho = optimal_rhop(pi, epsp, epsm)
    if rho >= 0:
        rhop = 0
        rhom = optimal_rhom(pi, epsp, epsm)
        return rhop, rhom
    else:
        return rho, 0

# Worst rhop: that verifies m_\rho = 0
def worst_rhop(pi, epsp, epsm):
    assert pi != 0.5
    return -(1 - 2 * pi * epsm - 2 * (1 - pi) * epsp) / (1 - 2 * pi)

def ratio(classifier, n, p, epsp, epsm, rhop, rhom, mu, gamma, pi):

    # E[g] and E(g^2)
    mean = test_expectation(classifier, n, p, pi, epsp, epsm, rhop, rhom, mu, gamma)
    expec_2 = test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    std = np.sqrt(expec_2 - mean**2)
    return abs(mean)/std

# finding optimal parameters for LPC-optimized with scipy.minimize
def find_optimal_rho(n, p, mu, pi, gamma, epsp, epsm, metric):
    if 'accuracy' in metric:
        #func = lambda x : 1 - test_accuracy('improved', n, p, epsp, epsm, x[0], x[1], mu, gamma, pi)
        func = lambda x : - ratio('improved', n, p, epsp, epsm, x[0], x[1], mu, gamma, pi)
    else: # risk
        func = lambda x : test_risk('improved', n, p, epsp, epsm, x[0], x[1], mu, gamma, pi)
    x0 = [0, 0]
    bounds = ((0, 1), (0, 1))
    #constraints = ({'type' : 'ineq', 'fun': lambda x: x[0] + x[1] - 1})
    res = minimize(func, x0, bounds = bounds , tol = 1e-8)#, constraints = constraints)
    return res.x

# Finding optimal gamma for Naive and Unbiased
def find_optimal_gamma(classifier, n, p, mu, pi, epsp, epsm, rhop, rhom, metric):
    if 'accuracy' in metric:
        func = lambda x : 1 - test_accuracy(classifier, n, p, epsp, epsm, rhop, rhom, mu, x, pi)
    else:
        func = lambda x : test_risk(classifier, n, p, epsp, epsm, rhop,rhom, mu, x, pi)
    
    x0 = 1e-4
    bounds = [(1e-4, 1e2)]
    res = minimize(func, x0, bounds = bounds)
    return res.x[0]

# Finding only rhop optimal for a given rhom with scipy.minimize
def find_optimal_rhop(n, p, mu, pi, gamma, epsp, epsm, rhom, metric):
    if 'accuracy' in metric:
        func = lambda x : - ratio('improved', n, p, epsp, epsm, x,rhom, mu, gamma, pi)
    else:
        func = lambda x : test_risk('improved', n, p, epsp, epsm, x, rhom, mu, gamma, pi )
    
    x0 = 0
    bounds = [(-10, 10)]
    res = minimize(func, x0, bounds = bounds)
    return res.x[0]

