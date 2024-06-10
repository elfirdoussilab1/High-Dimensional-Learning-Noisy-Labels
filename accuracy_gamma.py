# In this file, we will verify compare the accuracies of all our algorithms in function of gamma.
import numpy as np
from utils import *
from rmt_results import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

results_plot_directory = "./results-plot"

# Parameters
p = 20
# p = 200
# n = 200
n = 2000
pi = 0.3
epsp = 0.4
epsm = 0.3
rhop, rhom = optimal_rhos(pi, epsp, epsm)
eps = [epsp, epsm]
metric = 'accuracy'
batch = 50

metric_to_func = {'error': test_risk, 'accuracy': test_accuracy}
metric_to_emp = {'error': empirical_risk, 'accuracy': empirical_accuracy}
# Validate the result by simulation
k = 20 # number of gamma points
gammas = np.logspace(-6, 3, k)
mus = [0.5, 1, 2]

fig, ax = plt.subplots(1, 3, figsize = (30, 6))

linewidth = 5
fontsize = 40
labelsize = 35
s = 100
alpha = .7

for i, mu in enumerate(mus):
    perf_unb_th = []
    perf_unb = []
    perf_naive_th = []
    perf_naive = []
    perf_lpa_th = []
    perf_lpa = []
    perf_oracle_th = []
    perf_oracle = []
    for gamma_ in tqdm(gammas):
        perf_unb_th.append( test_accuracy('improved', n, p, epsp, epsm, epsp, epsm, mu, gamma_, pi))
        perf_unb.append( empirical_accuracy('improved', batch, n, p, mu, epsp, epsm, epsp, epsm, gamma_, pi, 'synthetic'))
        perf_naive_th.append( test_accuracy('naive', n, p, epsp, epsm, 0, 0, mu, gamma_, pi))
        perf_naive.append(empirical_accuracy('naive', batch, n, p, mu, epsp, epsm, 0, 0, gamma_, pi, 'synthetic'))
        perf_oracle_th.append( test_accuracy('oracle', n, p, 0, 0, 0, 0, mu, gamma_, pi))
        perf_oracle.append(empirical_accuracy('naive', batch, n, p, mu, 0, 0, 0, 0, gamma_, pi, 'synthetic'))
        perf_lpa_th.append( test_accuracy('improved', n, p, epsp, epsm, rhop, rhom, mu, gamma_, pi))
        perf_lpa.append(empirical_accuracy('improved', batch, n, p, mu, epsp, epsm, rhop, rhom, gamma_, pi, 'synthetic'))

    # Plot 
    # Improved
    ax[i].plot(gammas, perf_unb_th, linewidth = linewidth, color = 'tab:green')
    ax[i].scatter(gammas, perf_unb, linewidth = linewidth, color = 'tab:green', marker = 'o', s = s, alpha = alpha )
    # Naive
    ax[i].plot(gammas, perf_naive_th, linewidth = linewidth, color = 'tab:orange')
    ax[i].scatter(gammas, perf_naive, linewidth = linewidth, color = 'tab:orange', marker = '^', s= s, alpha = alpha)
    # LPOA
    ax[i].plot(gammas, perf_lpa_th, linewidth = linewidth, color = 'tab:blue')
    ax[i].scatter(gammas, perf_lpa, linewidth = linewidth, color = 'tab:blue', marker = 'o', s= s, alpha = alpha)
    # Oracle
    ax[i].plot(gammas, perf_oracle_th, linewidth = linewidth, color = 'tab:purple')
    ax[i].scatter(gammas, perf_oracle, linewidth = linewidth, color = 'tab:purple', marker = 'D', s = s, alpha = alpha)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].set_xlabel('$ \gamma $', fontsize = fontsize)
    ax[i].set_title(f'$ \| \mu \| = {mu}$', fontsize = fontsize)
    ax[i].grid(True)
    ax[i].set_xscale('log')
ax[0].set_ylabel(f'Test {metric.capitalize()}', fontsize = fontsize)
title = 'High-dimension' if n/p < 100 else 'Low-dimension'
fig.suptitle(title, fontsize=fontsize, ha='center', va = 'top', y = 1.1)

path = results_plot_directory + f'/{metric}-VS-gamma-n-{n}-p-{p}-pi-{pi}-epsp-{epsp}-epsm-{epsm}.pdf'
fig.savefig(path, bbox_inches='tight')



