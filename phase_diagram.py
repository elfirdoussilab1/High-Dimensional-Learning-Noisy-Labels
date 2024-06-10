from utils import *
import numpy as np
from rmt_results import *
from plot_utils import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset import *
from scipy.optimize import minimize
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

results_plot_directory = "./phase_diagrams"

def phase_transition(criteria, config, approach='optimized'):
    if 'risk' in criteria:
        func = test_risk
    else: # use accuracy
        func = test_accuracy
    
    def fun(x):
        return test_risk(config['classifier'], 
                         config['n'], 
                         config['p'], 
                         config['epsp'], 
                         config['epsm'], 
                         x[0], 
                         x[1], 
                         config['mu'], 
                         config['gamma'], 
                         1/config['split'])
    
    if 'LPC-Optimized' in approach:
        res = minimize(fun, [config['epsp'], config['epsm']])
        rhop = res.x[0]
        rhom = res.x[1]
    elif 'Unbiased' in approach:
        rhop = config['epsp']
        rhom = config['epsm']
    elif 'Naive' in approach:
        rhop = 0
        rhom = 0
        
        
    r = func(config['classifier'], 
             config['n'], 
             config['p'], 
             config['epsp'], 
             config['epsm'], 
             rhop, 
             rhom, 
             config['mu'], 
             config['gamma'], 
             1/config['split'])
    
    return r



# Parameters
n = 10000
gamma = 1e-1
fontsize = 30
labelsize = 20


approaches = ['Naive', 'Unbiased', 'LPC-Optimized']

etas = np.linspace(1e-5, 5, 120)
mus = np.linspace(0, 9, 150)
rs = np.zeros((len(etas), len(mus)))

for approach in approaches:
    for i, eta in enumerate(tqdm(etas[::-1])):
        for j, mu in enumerate(mus):
            config = {
                'p': int(eta*n),
                'n': n,
                'epsm': 0.2,
                'epsp': 0.3,
                'mu': mu,
                'split': 3,
                'eta': eta,
                'classifier': 'improved',
                'gamma': gamma
            }
            rs[i, j] = phase_transition('risk', config, approach)

    plt.figure()
    plt.imshow(rs, interpolation='nearest', cmap='RdBu', vmin=0, vmax=np.max(rs));
    plt.xticks(mus);
    plt.xticks(np.arange(len(mus))[::25], np.floor(10*mus[::25])/10)
    plt.yticks(np.arange(len(etas))[::20], np.floor(10*etas[::-1][::20])/10)
    plt.xlabel('$\Vert \mathbf{\mu}\Vert$', fontsize=fontsize);
    plt.ylabel('$\eta$', fontsize=fontsize);
    if 'naive' in approach:
        approach = approach.capitalize()
    plt.title(f'{approach}', fontsize=fontsize)
    plt.text(70, 50, f'$\gamma={gamma}$', fontsize=fontsize)
    plt.tick_params(axis='x', which = 'both', labelsize=labelsize)
    plt.tick_params(axis='y', which = 'both', labelsize=labelsize)
    cbar = plt.colorbar();
    cbar.ax.tick_params(labelsize=labelsize)
    epsm = config['epsm']
    epsp = config['epsp']
    path = results_plot_directory + f'/{approach}-{gamma}-{epsp}-{epsm}.pdf'
    plt.savefig(path, bbox_inches='tight')
    