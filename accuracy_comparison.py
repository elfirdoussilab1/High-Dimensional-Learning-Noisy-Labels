# This file will be used to generate the values of accuracies in the table 1 in the paper.
from utils import *
from rmt_results import *
import pandas as pd
from tqdm.auto import tqdm
from dataset import *

# Parameters
n = 1600
p = 400
pi = 0.3
epsp = 0.5
epsm = 0.4
gamma = 100
batch = 50
metric = 'accuracy'

# Create results in a dataframe: rows = datasets (4), columns = algorithms (3)
data_types = ['book', 'dvd', 'elec', 'kitchen']
results = pd.DataFrame(columns=['Naive', 'std_naive', 'Unbiased', 'std_unbiased', 'LPC-optimized', 'std_optimized'])
seeds = [1, 123, 404]

for typ in tqdm(data_types):
    data_name = 'amazon_' + typ
    data = Amazon(epsp, epsm, typ, pi, n)
    mu = data.mu

    acc_naive = []
    acc_unb = []
    acc_opt = []
    acc_oracle = []
    rhop, rhom = optimal_rhos(pi, epsp, epsm)
    
    for seed in seeds:
        fix_seed(seed)
        # Naive
        acc_naive.append(empirical_accuracy('naive', batch, n, p, mu, epsp, epsm, 0, 0, gamma, pi, data_name))

        # Unbiased
        acc_unb.append(empirical_accuracy('improved', batch, n, p, mu, epsp, epsm, epsp, epsm, gamma, pi, data_name))

        # Optimized
        acc_opt.append(empirical_accuracy('improved', batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_name))

        # Oracle
        acc_oracle.append(empirical_accuracy('naive', batch, n, p, mu, 0, 0, 0, 0, gamma, pi, data_name))

    row = {'Naive': round(np.mean(acc_naive) * 100, 2),
           'std_naive' : round(np.std(acc_naive) * 100, 2),
           'Unbiased' : round(np.mean(acc_unb) * 100, 2),
           'std_unbiased' : round(np.std(acc_unb) * 100, 2),
           'LPC-optimized': round(np.mean(acc_opt) * 100, 2), 
           'std_optimized' : round(np.std(acc_opt) * 100, 2),
           'Oracle': round(np.mean(acc_oracle) * 100, 2),
            'std_oracle': round(np.std(acc_oracle) * 100, 2)
           }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.index = data_types
path = './results-data/' + f'accuracy_comp-n-{n}-p-{p}-pi-{pi}-epsp-{epsp}-epsm-{epsm}-gamma-{gamma}.csv'
results.to_csv(path)
