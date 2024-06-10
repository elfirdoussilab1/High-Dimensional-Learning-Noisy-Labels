# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Plot directory
directory = "./results-plot"

fontsize = 40
labelsize = 35

# Parameters
p = 1000
#p = 50
n = 5000
gamma = 1e-1
epsm = 0.3
epsp = 0.4
rhom = epsm
rhop = epsp
mu = 2
pi = 1/3
eta = p/n
(X_train, y_train, y_train_noisy), (X_test, y_test) = generate_data(p, n, epsp, epsm, mu, pi)
classifiers = ['naive', 'improved', 'oracle']

fig, ax = plt.subplots(1, 3, figsize = (30, 5))

# Classifier
for i, classifier in enumerate(classifiers):
    if 'oracle' in classifier:
        w = fw(X_train, y_train, gamma)
    else:
        w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)

    delta = Delta(eta, gamma)

    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation(classifier, n, p , pi, epsp, epsm, rhop, rhom, mu, gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2(classifier, pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    t1 = np.linspace(mean_c1 - 4*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 4*std, mean_c2 + 5*std, 100)

    
    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= 3)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= 3)
    ax[i].set_xlabel('$\\mathbf{w}^\\top \\mathbf{x}$', fontsize = fontsize)
    ax[i].set_title(rf'{classifier.capitalize()}', fontsize= fontsize)

    # Plotting histogram
    ax[i].hist(X_test[:, :n].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, n:].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    if n/p < 100: # High-dimension
        if 'improved' in classifier:
            ax[i].axis([-5, 5, 0, 0.5])
        else:
            ax[i].axis([-2, 2, 0, 1.25])
    else: # low-dim
        if 'naive' in classifier:
            ax[i].axis([-1, 1, 0, 3.1])
        else:
            ax[i].axis([-2, 2, 0, 1.25])
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    # Label: label = '$\mathcal{C}_2$'
ax[1].set_title('Unbiased', fontsize= fontsize)
ylabel = 'Low-dimension' if n/p >= 100 else 'High-dimension'
ax[0].set_ylabel(ylabel, fontsize = fontsize)
path = directory + f'/distribution-all-n-{n}-p-{p}-pi-{pi}-epsp-{epsp}-epsm-{epsm}-mu-{mu}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')
