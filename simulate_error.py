# This file can be used to generate plots of risks that prove that our theory matches empirical quantities
import numpy as np
import matplotlib.pyplot as plt
from rmt_results import *
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
classifier = 'naive' # or classifier = 'imporved' for Unbiased and LPC
n = 2000
p = 200
pi = 0.3
epsilons = [(0.1, 0.2), (0.1, 0.4), (0.3, 0.4)] # couples (epsp, epsm)
rhop = 0
rhom = 0
mu = 2
batch = 50

gammas = np.logspace(-6, 2, 20)

linewidth = 5
fontsize = 40
labelsize = 35
s = 200
alpha = .9

# Expectation
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
for i, epss in enumerate(epsilons):
    expec_th = []
    expec = []
    epsp, epsm = epss
    for gamma in tqdm(gammas):
        # Theory
        expec_th.append(test_risk(classifier, n, p, epsp, epsm, rhop, rhom, mu, gamma, pi))

        # Practice
        expec.append(empirical_risk(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, 'synthetic'))
    
    ax[i].semilogx(gammas, expec_th, linewidth = linewidth, label = 'Theory', color = 'tab:red')
    ax[i].scatter(gammas, expec, s = s , color = 'tab:green', alpha = alpha, marker = 'D', label = 'Simulation')
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].set_xlabel('$\gamma $', fontsize = fontsize)
    ax[i].set_title(f'$ \\varepsilon_- = {epsm}$, $ \\varepsilon_+ = {epsp} $', fontsize = fontsize)
    ax[i].grid(True)

ax[0].set_ylabel('Test Risk', fontsize = fontsize)
ax[0].legend(fontsize = labelsize)

path = './study-plot/' + f'theory-vs-practice-risk-{classifier}-n-{n}-p-{p}-pi-{pi}-mu-{mu}-rhop-{rhop}-rhom-{rhom}.pdf'
fig.savefig(path, bbox_inches='tight')
