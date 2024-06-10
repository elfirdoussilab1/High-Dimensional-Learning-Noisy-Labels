# In this file, we will test multi-class extension that was described in the appendix
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

def find_best_param(params_grid, X_train, Y_noisy, X_test, Y_test, gamma):
    G = params_grid.shape[0]
    accs = []
    # Find the best parameter 
    for i in tqdm(range(G)):
        alphas = params_grid[i][:k]
        betas = params_grid[i][k:]
        W = multi_classifier(X_train, Y_noisy, alphas, betas, gamma)
        accs.append(accuracy_multi(X_test @ W, Y_test))

    # Best accuracy parameters
    best_params = params_grid[np.argmax(accs)]
    worst_params = params_grid[np.argmin(accs)]
    return best_params, worst_params

# Paramaters
n = 2000
p = 200
G = 5000
ks = [3, 4]
pis_all = [[0.3, 0.3, 0.4], [0.3, 0.2, 0.3, 0.2]]
mus_all = [[-2, 0, 2], [-6, -2, 2, 6]]
epsilons_all = [np.array([[0, 0.3, 0], [0, 0, 0.4], [0.5, 0, 0]]),
                np.array([[0, 0, 0.5, 0], [0, 0, 0, 0.3], [0, 0.4, 0, 0], [0.3, 0, 0, 0]])]

seeds = [1, 123, 404]

# plotting params
fontsize = 25
labelsize = 20
linewidth = 3
fig, ax = plt.subplots(1, 2, figsize = (15, 5))

for i, k in enumerate(ks):
    accs_naive = []
    accs_oracle = []
    accs_lpc = []

    for seed in seeds:
        fix_seed(seed)

        params_grid = (np.random.rand(G-1, 2*k) - 0.5)*4 # shape (G, 2k)
        naive_params = np.zeros(2*k)
        naive_params[:k] = 1
        params_grid = np.vstack((params_grid, naive_params))

        pis = pis_all[i]
        mus = mus_all[i]
        epsilons = epsilons_all[i]

        # Dataset
        X_train, Y_train, Y_noisy = generate_data_multi(n, p, pis, mus, epsilons, train = True)
        X_test, Y_test = generate_data_multi(n, p, pis, mus, epsilons, train = False)
        A = np.linspace(0, 1, 50)

        # Naive
        gamma_naive = find_optimal_gamma_multi(X_train, Y_noisy, X_test, Y_test, naive_params)
        W_naive = multi_classifier(X_train, Y_noisy, naive_params[:k], naive_params[k:], gamma_naive)
        acc_naive = accuracy_multi(X_test @ W_naive, Y_test)
        accs_naive.append(acc_naive)

        # LPC
        best_params, worst_params = find_best_param(params_grid, X_train, Y_noisy, X_test, Y_test, 1)
    
        accs = []
        for a in A:
            params = a * best_params + (1 - a) * worst_params
            W = multi_classifier(X_train, Y_noisy, params[:k], params[k:], gamma_naive)
            accs.append(accuracy_multi(X_test @ W, Y_test))

        accs_lpc.append(np.array(accs))

        # Oracle
        X_train, Y_train, Y_noisy = generate_data_multi(n, p, pis, mus, np.zeros((k, k)), train = True)
        gamma_oracle = find_optimal_gamma_multi(X_train, Y_noisy, X_test, Y_test, naive_params)
        W_oracle = multi_classifier(X_train, Y_noisy, naive_params[:k], naive_params[k:], gamma_oracle) 
        acc_oracle = accuracy_multi(X_test @ W_oracle, Y_test)
        accs_oracle.append(acc_oracle)
    
    # Plotting results
    # Naive
    ax[i].plot([A[0], A[-1]], [np.mean(accs_naive), np.mean(accs_naive)], '-.', label = 'Naive', color = 'tab:orange', linewidth = linewidth)
    ax[i].fill_between([A[0], A[-1]], [np.mean(accs_naive) - np.std(accs_naive), np.mean(accs_naive) - np.std(accs_naive)], 
         [np.mean(accs_naive) + np.std(accs_naive), np.mean(accs_naive) + np.std(accs_naive)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:orange')
    
    # Oracle
    ax[i].plot([A[0], A[-1]], [np.mean(accs_oracle), np.mean(accs_oracle)], '-.', label ='Oracle', color = 'tab:purple', linewidth = linewidth)
    ax[i].fill_between([A[0], A[-1]], [np.mean(accs_oracle) - np.std(accs_oracle), np.mean(accs_oracle) - np.std(accs_oracle)], 
         [np.mean(accs_oracle) + np.std(accs_oracle), np.mean(accs_oracle) + np.std(accs_oracle)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:purple')
    
    # LPC
    ax[i].plot(A, np.mean(accs_lpc, axis = 0), label = 'Multi-LPC', color = 'tab:blue', linewidth = linewidth)
    ax[i].fill_between(A, np.mean(accs_lpc, axis = 0) - np.std(accs_lpc, axis = 0), np.mean(accs_lpc, axis = 0) + np.std(accs_lpc, axis = 0), 
         alpha = 0.2, linestyle = '-.', color = 'tab:blue')

    ax[i].set_xlabel('$\\tau $', fontsize = fontsize)
    ax[i].set_title(f'Number of classes k = {k}', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid(True)

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
ax[0].legend(fontsize = labelsize)

path = './results-plot/' + f'multi_class-n-{n}-p-{p}-gamma_naive-{gamma_naive}-gamma_oracle-{gamma_oracle}.pdf'
fig.savefig(path, bbox_inches='tight')
