# This file is used to generate distribution plots for real datasets
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from dataset import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

fix_seed(123)
# Plot directory
study_plot_directory = "./study-plot"

fontsize = 40
labelsize = 35

# Amazon
p = 400
n = 1600
pi = 0.3
gamma = 100
epsm = 0.3
epsp = 0.4
type = 'book'
data_type = 'amazon_' + type
data = Amazon(epsp, epsm, type, pi, n)
mu = data.mu

metric = 'accuracy'
batch = 100
eta = p/n
classifiers = ['naive', 'improved', 'improved']
classifier_name = {0: 'Naive', 1: 'Unbiased', 2: 'LPC-optimized'}
X_train, y_train, y_train_noisy = data.X_train.T, data.y_train, data.y_train_noisy
X_test, y_test = data.X_test.T, data.y_test

fig, ax = plt.subplots(1, 3, figsize = (30, 6))

# Classifier
for i, classifier in enumerate(classifiers):
    if i == 2:
        rhop, rhom = optimal_rhos(pi, epsp, epsm)
    else:
        rhop, rhom = epsp, epsm

    w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)

    delta = Delta(eta, gamma)

    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation(classifier, n, p, pi, epsp, epsm, rhop, rhom, mu, gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    t1 = np.linspace(mean_c1 - 4*std, mean_c1 + 4*std, 100)
    t2 = np.linspace(mean_c2 - 4*std, mean_c2 + 4*std, 100)

    
    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= 3)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= 3)
    ax[i].set_xlabel('$\\mathbf{w}^\\top \\mathbf{x}$', fontsize = fontsize)
    acc_emp = empirical_accuracy(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type)
    ax[i].set_title(f'{classifier_name[i]}, Acc = {round(acc_emp*100, 2)} \%', fontsize= fontsize)

    # Plotting histogram
    ax[i].hist(X_test[:, (y_test < 0)].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, (y_test > 0)].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    # Label: label = '$\mathcal{C}_2$'
ax[0].set_ylabel(r'Density', fontsize = fontsize)
path = './results-plot' + f'/distribution_real-type-{data_type}-n-{n}-p-{p}-pi-{pi}-epsp-{epsp}-epsm-{epsm}-mu-{mu}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')
