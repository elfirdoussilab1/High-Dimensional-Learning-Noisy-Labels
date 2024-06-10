from utils import *
from plot_utils import *
from rmt_results import *
from tqdm.auto import tqdm

# Parameters
p = 2
n = 20000
pi = 1/3
epsp = 0.3
epsm = 0.2
rhom = epsm
#rhop = worst_rho(pi, epsp, epsm)
rhop = epsp
print("Worst rhop", rhop)
mu = 2

eta = p/n
data_type = 'synthetic'

# Generating train and test data
fix_seed(123) # reproducibility

# Computing the L2 loss over test data by the naive separator (Practice)
classifier = 'naive'
metric = 'accuracy'
gammas = np.logspace(-5, 4, 20)
accuracy_practice = []
accuracy_theory = []
batch = 20

for gamma in tqdm(gammas):
    # Simulation
    accuracy_practice.append(empirical_accuracy(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type))

    # Theory
    accuracy_theory.append(test_accuracy(classifier, n, p, epsp, epsm, rhop, rhom, mu, gamma, pi))

plot_metric(p, n, mu, gammas, epsp, epsm, rhop, rhom, pi, batch, classifier, metric, accuracy_practice, accuracy_theory)