# In this file, we verify where the maximum of Test Accuracy is reached in function with rho_+
import numpy as np
import matplotlib.pyplot as plt
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})
# Parameters
n = 1000
pi = 0.3
epsm = 0.3
epsp = 0.4
rhom = 0
mu = 2
gamma = 10

ps = [100, 1000, 10000]
N = 1000
rhops = np.linspace(-10, 10, N)
linewidth = 4
fontsize = 40
labelsize = 35
s = 250
metric = 'accuracy'
fig, ax = plt.subplots(1, 3, figsize = (30, 6))

for i, p in enumerate(ps):
    accs = []
    for rhop in rhops:
        # Test accuracy
        accs.append(test_accuracy('improved', n, p, epsp, epsm, rhop, rhom, mu, gamma, pi))
    ax[i].plot(rhops, accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].scatter(rhops[np.argmax(accs)], np.max(accs), color = 'tab:green', s = s, marker = 'D')
    ax[i].scatter(rhops[np.argmin(accs)], np.min(accs), color = 'tab:red', s = s, marker = 'D')
    # Naive
    acc_naive = test_accuracy('naive', n , p, epsp, epsm, 0, 0, mu, gamma, pi)
    ax[i].plot([-10, 10], [acc_naive, acc_naive], '-.', color = 'tab:orange', linewidth = linewidth)

    # Unbiased
    acc_unb = test_accuracy('improved', n , p, epsp, epsm, epsp, epsm, mu, gamma, pi)
    ax[i].plot([-10, 10], [acc_unb, acc_unb], '-.', color = 'tab:green', linewidth = linewidth)
    
    # Oracle
    acc_oracle = test_accuracy('naive', n , p, 0, 0, 0, 0, mu, gamma, pi)
    ax[i].plot([-10, 10], [acc_oracle, acc_oracle], '-.', color = 'tab:purple', linewidth = linewidth)

    # Optimal rhop
    #x_max, y_max = rhops[np.argmax(accs)], np.max(accs)
    #x_min, y_min = rhops[np.argmax(accs)], np.max(accs)
    rhop_max = optimal_rhop(pi, epsp, epsm)
    acc_max = test_accuracy('improved', n, p, epsp, epsm, rhop_max, rhom, mu, gamma, pi)

    rhop_min = worst_rhop(pi, epsp, epsm)
    acc_min = test_accuracy('improved', n, p, epsp, epsm, rhop_min, rhom, mu, gamma, pi)

    #sentence = '$\\rho_+^*=%.2f$, $Acc^*=%.2f$' % (x, y)
    sentence_max = f'$\\rho_+^*= {round(rhop_max, 2)}$'
    sentence_min = f'$\\bar \\rho_+= {round(rhop_min, 2)}$'
    hx = 1e-2
    if i == 0:
        hy = -8e-2
    else:
        hy = 3e-2

    ax[i].text(rhop_max+hx, acc_max+hy, sentence_max, fontsize = fontsize - 10)
    ax[i].text(rhop_min+ 1, acc_min , sentence_min, fontsize = fontsize - 10)
    ax[i].set_title(f'$ \eta = {p/n}$', fontsize = fontsize)
    
    ax[i].set_xlabel('$\\rho_+$', fontsize = fontsize)
    ax[i].set_xticks([-10, -5, 0, 5, 10])
    ax[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid()

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
path = './results-plot/' + f'accuracy_rho-n-{n}-pi-{pi}-mu-{mu}-epsp-{epsp}-epsm-{epsm}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')



