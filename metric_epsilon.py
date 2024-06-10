# In this file, we will generate plots of Test Accuracy | Test Risk with epsilon
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from rmt_results import *
from tqdm.auto import tqdm
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 100
ps = [10, 100, 1000]
pi = 1/3
epsm = 0.2
rhom = 0
mu = 2
gamma = 10
batch = 100
#metric = 'risk'
metric = 'accuracy'

metric_to_func_th = {'accuracy': test_accuracy, 'risk': test_risk}
metric_to_func = {'accuracy': empirical_accuracy, 'risk': empirical_risk}

fix_seed(123)
rhop = 0.2

epsps = np.linspace(0, 0.65, 25)

# figure
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 4
s = 100

for i, p in enumerate(ps):
    acc_naive = []
    acc_naive_th = []
    acc_unb = []
    acc_unb_th = []
    acc_oracle = []
    acc_oracle_th = []
    acc_lpc = []
    acc_lpc_th = []
    for epsp in tqdm(epsps):
        acc_naive_th.append(metric_to_func_th[metric]('naive', n, p, epsp, epsm, 0, 0, mu, gamma, pi))
        acc_naive.append(metric_to_func[metric]('naive', batch, n, p, mu, epsp, epsm, 0, 0, gamma, pi, 'synthetic'))
        acc_unb_th.append(metric_to_func_th[metric]('improved', n, p, epsp, epsm, epsp, epsm, mu, gamma, pi))
        acc_unb.append(metric_to_func[metric]('improved', batch, n, p, mu, epsp, epsm, epsp, epsm, gamma, pi, 'synthetic'))
        acc_lpc_th.append(metric_to_func_th[metric]('improved', n, p, epsp, epsm, rhop, rhom, mu, gamma, pi))
        acc_lpc.append(metric_to_func[metric]('improved', batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, 'synthetic'))
        acc_oracle_th.append(metric_to_func_th[metric]('naive', n, p, 0, 0, 0, 0, mu, gamma, pi))
        acc_oracle.append(metric_to_func[metric]('naive', batch, n, p, mu, 0, 0, 0, 0, gamma, pi, 'synthetic'))

    # Plotting results
    # Naive
    ax[i].plot(epsps, acc_naive_th, linewidth = linewidth, color = 'tab:orange')
    ax[i].scatter(epsps, acc_naive, color = 'tab:orange', s = s, marker = '^')
    # Unbiased
    ax[i].plot(epsps, acc_unb_th, linewidth = linewidth, color = 'tab:green')
    ax[i].scatter(epsps, acc_unb, color = 'tab:green', s = s, marker = 'o')
    # Oracle
    ax[i].plot(epsps, acc_oracle_th, linewidth = linewidth, color = 'tab:purple')
    ax[i].scatter(epsps, acc_oracle, color = 'tab:purple', s = s, marker = 'D')
    # LPC
    ax[i].plot(epsps, acc_lpc_th, linewidth = linewidth, color = 'tab:blue')
    ax[i].scatter(epsps, acc_lpc, color = 'tab:blue', s = s, marker = 'o')

    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].set_xlabel('$ \\varepsilon_+ $', fontsize = fontsize)
    ax[i].set_title(f'$ \eta = {p/n} $', fontsize = fontsize)
    ax[i].grid(True)

ax[0].set_ylabel(f'Test {metric.capitalize()}', fontsize = fontsize)
path = './results-plot/' + f'{metric}_epsilon-n-{n}-pi-{pi}-epsm-{epsm}-mu-{mu}-gamma-{gamma}-rhom-{rhom}-rhop-{rhop}.pdf'
fig.savefig(path, bbox_inches='tight')




