import numpy as np
from utils import *
from rmt_results import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

study_plot_directory = "./study-plot"

# pi and mu estimation
def estimate_mu_pi(X_train):
    # X_train of shape (p, n)
    eigenvalues, eigenvectors = np.linalg.eig(X_train.T @ X_train / X_train.shape[1])
    argm = np.argmax(eigenvalues)
    u = np.real(eigenvectors[:, argm])
    # Estimation of pi
    pi_est = np.mean((u < 0))

    # Estimation of mu
    lam_max = np.real(eigenvalues[argm])
    eta = X_train.shape[0] / X_train.shape[1]
    mu_est = np.sqrt((lam_max - eta - 1 + np.sqrt((lam_max - eta - 1)**2 - 4*eta)) / 2)
    
    # Plotting result of eigenvector
    t = np.arange(0, len(u))
    fig, ax = plt.subplots()
    ax.scatter(t , u)
    ax.set_title('Coordinates of the dominant eigenvector')
    path = study_plot_directory + f"/libsvm-eigenvector-n-{X_train.shape[1]}-p-{X_train.shape[0]}.pdf"
    fig.savefig(path)
    
    return mu_est, pi_est

# Finding (epsp, epsm) given pi_est
def solve_epsilons(n, p, pi, pi_est, mu, gamma, epsp, epsm, rhops, rhoms, batch, data_type = 'synthetic'):
    #l1 = test_expectation_2_imp(pi, n, p, epsp, epsm, rhops[0], rhoms[0], mu, gamma)
    l1 = empirical_mean_2('improved', batch, n, p, mu, epsp, epsm, rhops[0], rhoms[0], gamma, pi, data_type)
    #l2 = test_expectation_2_imp(pi, n, p, epsp, epsm, rhops[1], rhoms[1], mu, gamma)
    l2 = empirical_mean_2('improved', batch, n, p, mu, epsp, epsm, rhops[1], rhoms[1], gamma, pi, data_type)
    # functions: x[0] = epsp, x[1] = epsm
    f_1 = lambda x: test_expectation_2_imp(pi_est, n, p, x[0], x[1], rhops[0], rhoms[0], mu, gamma)
    f_2 = lambda x: test_expectation_2_imp(pi_est, n, p, x[0], x[1], rhops[1], rhoms[1], mu, gamma)
    func = lambda x : [f_1(x) - l1, f_2(x) - l2]
    res = fsolve(func, [0., 0.])
    return res

# Pipeline that works to find the parameters (pi, epsp, epsm)
def find_params(n, p, pi, vmu, gamma, X_train, epsp, epsm, rhops, rhoms, batch, W, data_type = 'synthetic'):
    # The parameters pi, epsm and epsp are only used for Monte Carlo estimation of the variance !
    # mu is obtained with estimation
    # Estimating mu and pi
    mu_est, pi_est = estimate_mu_pi(X_train)
    print("mu estimation", mu_est)
    print("pi estimation", pi_est)

    # Estimating epsm and epsp with pi_est
    couples = []
    epsp_1, epsm_1 = solve_epsilons(n, p, pi, pi_est, vmu, gamma, epsp, epsm, rhops, rhoms, batch, W, data_type)

    # Checking correctness of the first couple
    if check_coherence(epsp_1, epsm_1):
        couples.append([pi_est, epsp_1, epsm_1])

    # Estimating epsm and epsp with 1 - pi_est
    epsp_2, epsm_2 = solve_epsilons(n, p, pi, 1 - pi_est, vmu, gamma, epsp, epsm, rhops, rhoms, batch, W, data_type)

    # Checking correctness of the second couple
    if check_coherence(epsp_2, epsm_2):
        couples.append([1 - pi_est, epsp_2, epsm_2])

    return couples
