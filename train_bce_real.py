# In this file, we will evaluate the performance of LPC trained with BCE Loss on real data
import numpy as np
from utils import *
from dataset import *
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Amazon
p = 400
n = 1600
pi = 0.3
epsm = 0.2
epsp = 0.3
type = 'dvd'
data_type = 'amazon_' + type
data = Amazon(epsp, epsm, type, pi, n)
mu = data.mu

rhom = 0
rhops = np.linspace(-2, 2, 50)

lr = 0.1
gamma = 0
N = 100
seeds = [1, 123, 404]
accs_lpc = []
accs_naive = []
accs_unb = []
accs_oracle = []

for seed in seeds:
    fix_seed(seed)

    # Dataset
    data_seed = Amazon(epsp, epsm, type, pi, n)
    X_train = data_seed.X_train
    y_train_noisy = data_seed.y_train_noisy
    X_test = data_seed.X_test
    y_test = data_seed.y_test
    accs = []
    # LPC
    for rhop in tqdm(rhops):
        w, losses = grad_descent(X_train, y_train_noisy, lr, gamma, rhop, rhom, N)
        accs.append(accuracy(decision(w, X_test.T), y_test))

    accs_lpc.append(np.array(accs))

    # Unbiased
    w_unb, l = grad_descent(X_train, y_train_noisy, lr, gamma, epsp, epsm, N)
    accs_unb.append(accuracy(decision(w_unb, X_test.T), y_test))

    # Naive
    w_naive, l = grad_descent(X_train, y_train_noisy, lr, gamma, 0, 0, N)
    accs_naive.append(accuracy(decision(w_naive, X_test.T), y_test))

# Oracle
for seed in seeds:
    fix_seed(seed)
    data_temp = Amazon(0, 0, type, pi, n)
    X_train = data_temp.X_train
    y_train = data_temp.y_train
    X_test = data_temp.X_test
    y_test = data_temp.y_test
    w_oracle, l = grad_descent(X_train, y_train, lr, gamma, 0, 0, N)
    accs_oracle.append(accuracy(decision(w_oracle, X_test.T), y_test))

"""
# Save arrays in a npy file
np.save('accs_lpc_real.npy', np.array(accs_lpc))
np.save('accs_unb_real.npy', np.array(accs_unb))
np.save('accs_naive_real.npy', np.array(accs_naive))
np.save('accs_oracle_real.npy', np.array(accs_oracle))

accs_lpc = np.load('accs_lpc_real.npy')
accs_unb = np.load('accs_unb_real.npy')
accs_naive = np.load('accs_naive_real.npy')
accs_oracle = np.load('accs_oracle_real.npy')
"""
# Plotting results
linewidth = 2
fontsize = 13
labelsize = 10
s = 20

plt.figure(figsize=(7, 4))

# LPC
plt.plot(rhops, np.mean(accs_lpc, axis = 0), color = 'tab:blue', linewidth = linewidth, label = 'LPC')
plt.fill_between(rhops, np.mean(accs_lpc, axis = 0) - np.std(accs_lpc, axis = 0), np.mean(accs_lpc, axis = 0) + np.std(accs_lpc, axis = 0), 
         alpha = 0.2, linestyle = '-.', color = 'tab:blue')

# Max and Min points
x_max, y_max = rhops[np.argmax(np.mean(accs_lpc, axis = 0))], np.max(np.mean(accs_lpc, axis = 0))
x_min, y_min = rhops[np.argmin(np.mean(accs_lpc, axis = 0))], np.min(np.mean(accs_lpc, axis = 0))
plt.scatter(x_max, y_max, color = 'tab:green', marker = 'D', s = s)
plt.scatter(x_min, y_min, color = 'tab:red', marker = 'D', s = s)
sentence_max = f'$\\rho_+^*= {round(x_max, 2)}$'
sentence_min = f'$\\bar \\rho_+= {round(x_min, 2)}$'
plt.text(x_max - 5e-2, y_max-  5e-2, sentence_max, fontsize = labelsize)
plt.text(x_min+ 3e-1, y_min , sentence_min, fontsize = labelsize)

# Unbiased
plt.plot([rhops[0], rhops[-1]], [np.mean(accs_unb), np.mean(accs_unb)], '-.', color = 'tab:green', linewidth = linewidth, label = 'Unbiased')
plt.fill_between([rhops[0], rhops[-1]], [np.mean(accs_unb) - np.std(accs_unb), np.mean(accs_unb) - np.std(accs_unb)], 
         [np.mean(accs_unb) + np.std(accs_unb), np.mean(accs_unb) + np.std(accs_unb)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:green')

# Naive
plt.plot([rhops[0], rhops[-1]], [np.mean(accs_naive), np.mean(accs_naive)], '-.', color = 'tab:orange', linewidth = linewidth, label = 'Naive')
plt.fill_between([rhops[0], rhops[-1]], [np.mean(accs_naive) - np.std(accs_naive), np.mean(accs_naive) - np.std(accs_naive)], 
         [np.mean(accs_naive) + np.std(accs_naive), np.mean(accs_naive) + np.std(accs_naive)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:orange')

# Oracle
plt.plot([rhops[0], rhops[-1]], [np.mean(accs_oracle), np.mean(accs_oracle)], '-.', color = 'tab:purple', linewidth = linewidth, label = 'Oracle')
plt.fill_between([rhops[0], rhops[-1]], [np.mean(accs_oracle) - np.std(accs_oracle), np.mean(accs_oracle) - np.std(accs_oracle)], 
         [np.mean(accs_oracle) + np.std(accs_oracle), np.mean(accs_oracle) + np.std(accs_oracle)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:purple')

plt.tick_params(axis='x', which = 'both', labelsize=labelsize)
plt.tick_params(axis='y', which = 'both', labelsize=labelsize)
plt.xlabel('$\\rho_+ $', fontsize = fontsize)
plt.ylabel('Test Accuracy', fontsize = fontsize)
plt.grid()
plt.legend(fontsize = labelsize)
path = './study-plot/' + f'bce-{type}-n-{n}-p-{p}-pi-{pi}-epsp-{epsp}-epsm-{epsm}-mu-{mu}-lr-{lr}-gamma-{gamma}.pdf'
plt.savefig(path, bbox_inches='tight')
