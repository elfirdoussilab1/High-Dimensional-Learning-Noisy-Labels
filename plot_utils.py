# This file contains some util functions for plotting results
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from rmt_results import *
from utils import *
from mpl_toolkits import mplot3d

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

results_plot_directory = "./results-plot"
if not os.path.exists(results_plot_directory):
    os.makedirs(results_plot_directory)

study_plot_directory = "./study-plot"
if not os.path.exists(study_plot_directory):
    os.makedirs(study_plot_directory)

def plot_metric(p, n, mu, gammas, epsp, epsm, rhop, rhom, pi, batch, classifier, metric, loss_practice, loss_theory):
    fig, ax = plt.subplots()
    ax.semilogx(gammas, loss_theory, label = 'Theory', color = 'r')
    ax.scatter(gammas, loss_practice, label = 'Simulation', color = 'g', marker = '*')
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel(f'{metric.capitalize()}')
    ax.set_title(f'$\epsilon_-$ = {epsm} and $\epsilon_+$ = {epsp}')
    ax.legend()
    path = results_plot_directory + f"/{metric}-{classifier}-theory_vs_practice-n-{n}-p-{p}-epsm-{epsm}-epsp-{epsp}-rhop-{rhop}-rhom-{rhom}-mu-{mu}-pi-{int(pi*100)}-batch-{batch}.pdf"
    fig.savefig(path)

metric_to_func = {'error': test_risk, 'accuracy': test_accuracy}

def plot_metric_epsm(p, n, mu, gamma, pi, epsp, epsm_grid, classifier, metric):
    eta = p/n
    delta = Delta(eta, gamma)
    vmu = np.zeros(p)
    vmu[0]= mu
    q_bar = Q_bar(vmu, delta, gamma)
    accuracy_theory = []
    for epsm in tqdm(epsm_grid):
        accuracy_theory.append(metric_to_func[metric](n, p, epsp, epsm, delta, q_bar, vmu, gamma, pi, classifier))

    fig, ax = plt.subplots()
    ax.plot(epsm_grid, accuracy_theory)
    ax.set_xlabel('$\epsilon_-$')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'$\epsilon_+$ = {epsp}')
    path = study_plot_directory + f"/{metric}-vs-epsm-{classifier}-n-{n}-p-{p}-epsm-{epsm}-epsp-{epsp}-mu-{mu}.pdf"
    fig.savefig(path)

def plot_3D_metric(p, n, mu, gamma, pi, epsp_grid, epsm_grid, classifier, metric):
    eta = p/n
    delta = Delta(eta, gamma)
    vmu = np.zeros(p)
    vmu[0]= mu
    k = len(epsm_grid) # = len(epsp_grid)
    q_bar = Q_bar(vmu, delta, gamma)
    accuracy_theory = np.zeros((k, k))
    X, Y = np.meshgrid(epsp_grid, epsm_grid)
    for i, epsp in enumerate(epsp_grid):
        for j, epsm in enumerate(epsm_grid):
            accuracy_theory[i, j] = metric_to_func[metric](n, p, epsp, epsm, delta, q_bar, vmu, gamma, pi, classifier)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, accuracy_theory, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(f'{metric.capitalize()} VS $\epsilon_\pm$')
    ax.set_xlabel('$\epsilon_+$')
    ax.set_ylabel('$\epsilon_-$')
    ax.set_zlabel(metric.capitalize())
    ax.view_init(30, 35)
    path = study_plot_directory + f"/{metric}-3D-{classifier}-n-{n}-p-{p}-mu-{mu}.pdf"
    fig.savefig(path)


# Past code for performance
def plot_acc(accs, ns, epss, column='eval'):
    """
    accs is the dictionary containing: train | train-noisy | eval
    Plots the accuracy acc in function of epss for all the ns
    """
    fig, ax = plt.subplots()
    ax.plot(epss, np.array(accs[column]).T, '-.', label=['n=%d' % n for n in ns]);
    ax.set_xlabel('epsilon')
    ax.set_ylabel('accuracy')
    ax.set_title(column)
    ax.legend();
    path = results_plot_directory + "/plot_acc-{column}.pdf"
    fig.savefig(path)

def plot_comp_acc(acc_naive, acc_improved, ns, epss, column = 'eval'):
    fig, ax = plt.subplots()
    ax.plot(epss, acc_naive, ':', label=['(naive) n=%d' % n for n in ns]);
    plt.gca().set_prop_cycle(None)
    ax.plot(epss, acc_improved, label=['(improved) n=%d' % n for n in ns]);
    ax.set_xlabel('epsilon')
    ax.set_ylabel('accuracy')
    ax.legend()
    path = results_plot_directory + f"/plot_acc_comp-{column}.pdf"
    fig.savefig(path)


def plot_comp_global(path_naive, path_imp, ns, epss):
    """
    Make three plots in the same row: train | train-noisy | eval
    """
    accs_naive = pd.read_csv(path_naive)
    accs_imp = pd.read_csv(path_imp)
    fig, axes = plt.subplots(1, 3, figsize = (25, 7))
    plt.gca().set_prop_cycle(None)
    for n in ns:
        # train
        axes[0].plot(epss, np.array(accs_naive['train'][(accs_naive['n'] == n)]), ':', label=f'(naive) n={n}' )
        # train-noisy
        axes[1].plot(epss, np.array(accs_naive['train-noisy'][(accs_naive['n'] == n)]).T, ':', label=f'(naive) n={n}')
        # eval
        axes[2].plot(epss, np.array(accs_naive['eval'][(accs_naive['n'] == n)]).T, ':', label=f'(naive) n={n}') 
    plt.gca().set_prop_cycle(None)
    for n in ns:
        axes[0].plot(epss, np.array(accs_imp['train'][(accs_naive['n'] == n)]), label=f'(improved) n={n}')
        axes[1].plot(epss, np.array(accs_imp['train-noisy'][(accs_naive['n'] == n)]).T, label=f'(improved) n={n}')
        axes[2].plot(epss, np.array(accs_imp['eval'][(accs_naive['n'] == n)]).T, label=f'(improved) n={n}')

    # legend and save
    axes[0].set_title('train')
    axes[1].set_title('train-noisy')
    axes[2].set_title('eval')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()    
    fig.text(0.5, 0.04, 'epsilon', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    path = results_plot_directory + "/plot_comp_global.pdf"
    fig.savefig(path)