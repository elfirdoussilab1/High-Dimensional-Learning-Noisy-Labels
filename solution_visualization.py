import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import *
from rmt_results import *

study_plot_directory = "./study-plot"

# Parameters
p = 300
n = 400
pi = 1/3
epsp = 0.1
epsm = 0.2
gamma = 1e-1
eta = p/n
delta = Delta(eta, gamma)

#mus = [1e-1, 1, 10]

#fig, ax = plt.subplots(1, 3, figsize = (30, 8), subplot_kw={'projection': '3d'})
fig = plt.figure()
ax = plt.axes(projection='3d')
mu = 4

# 3D Plot
#for m, mu in enumerate(mus):

# Grids
rhops = np.linspace(0, 0.4, 50)
rhoms = np.linspace(0, 0.4, 50)

k = len(rhops)
expec_2 = np.zeros((k, k))

X, Y = np.meshgrid(rhops, rhoms)
for i, rhop in enumerate(rhops):
    for j, rhom in enumerate(rhoms):
        expec_2[i, j] = test_expectation_2_imp(pi, n, p, epsp, epsm, rhop, rhom, mu, gamma)

ax.plot_surface(X, Y, expec_2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title(f"$ \| \mu \| = {mu} $ ")
#ax.view_init(elev=90, azim=0)
ax.set_xlabel("$ \\rho_+ $")
ax.set_ylabel("$ \\rho_- $")
path = study_plot_directory + f'/3D-param_estimation-n-{n}-p{p}-epsp-{epsp}-epsm-{epsm}-gamma-{gamma}.pdf'
fig.savefig(path)
