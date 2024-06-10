# In this file, we test the functions implemented in param_estimation.py file
import numpy as np
import matplotlib.pyplot as plt
from param_estimation import *
from dataset import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Synthetic data
# Parameters
p = 100
n = 1000
pi = 1/3
epsm = 0.2
eta = p/n
gamma = 1e-1
batch = 100
data_type = 'synthetic'

# Unknown (a priori) parameters
mus = np.logspace(-1, 1, 3)

epsps = np.linspace(0, 1 - epsm - 0.1, 25)
seeds = [1, 123, 303, 404]

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
linewidth = 5
fontsize = 40
labelsize = 35

rhops = [0., 0.]
rhoms = [0.1, 0.4]
for i, mu in enumerate(tqdm(mus)):
    epsp_estim_global = []
    for seed in seeds:
        fix_seed(seed)
        epsp_estim = []

        for epsp in epsps:
            couples = solve_epsilons(n, p, pi, pi, mu, gamma, epsp, epsm, rhops, rhoms, batch, data_type)
            epsp_estim.append(couples[0])
        epsp_estim_global.append(np.array(epsp_estim))
    
    ax[i].plot(epsps, epsps, ':', label = '$ y = x $', linewidth = linewidth, color = 'black' )
    ax[i].plot(epsps, np.mean(epsp_estim_global, axis = 0), color = 'tab:green', linewidth = linewidth, alpha = .8)
    ax[i].fill_between(epsps, np.mean(epsp_estim_global, axis = 0) - np.std(epsp_estim_global, axis = 0), np.mean(epsp_estim_global, axis = 0) + np.std(epsp_estim_global, axis = 0), 
                       alpha = 0.2, linestyle = '-.', color = 'tab:red')
    ax[i].set_title('$ \| \mathbf{\mu} \| = $' + f'{mu}', fontsize = fontsize)
    ax[i].set_xlabel('True $\\varepsilon_+$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid(True)
    #ax[i].axis([0, 0.8, 0, 0.8])
ax[0].legend(fontsize = fontsize) 
ax[0].set_ylabel('Estimated $\\varepsilon_+$', fontsize = fontsize)
path = './results-plot/' + f'synthetic-param_estimation-n-{n}-p-{p}-pi-{pi}-epsm-{epsm}-gamma-{gamma}-batch-{batch}.pdf'
fig.savefig(path, bbox_inches='tight')