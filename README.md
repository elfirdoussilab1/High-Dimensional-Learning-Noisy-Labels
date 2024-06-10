# rmt-noisy-labels
This is the official code repository of the paper: High-dimensional Learning with Noisy Labels.

## Abstract
This paper provides theoretical insights into high-dimensional binary classification with class-conditional noisy labels. Specifically, we study the behavior of a linear classifier with a label noisiness aware loss function, when both the dimension of data $p$ and the sample size $n$ are large and comparable. Relying on random matrix theory by supposing a Gaussian mixture data model, the performance of the linear classifier when $p,n\to \infty$ is shown to converge towards a limit, involving scalar statistics of the data. Importantly, our findings show that the low-dimensional intuitions to handle label noise do not hold in high-dimension, in the sense that the optimal classifier in low-dimension dramatically fails in high-dimension. Based on our derivations, we design an optimized method that is shown to be provably more efficient in handling noisy labels in high dimensions.
Our theoretical conclusions are further confirmed by experiments on real datasets, where we show that our optimized approach outperforms the considered baselines.

## Paper figures:
All the figures in the paper and more can be found in the folders: 
* [results-plot](results-plot/): contains all the figures shown in the paper
* [study-plot](study-plot/) : for some additonal plots (that are not included in the paper)

## Reproducing figures:
* Run the file [simulate_distribution](simulate_distribution.py) or command `python3 simulate_distribution.py` to get distribution plots of $w^\top x$ in Low-dimension (by taking $p = 50$) or in High-dimension ($p = 1000$) in Figure 1. The corresponding legend is [legend_1](legend/legend_1.pdf).
* Run the file [metric_epsilon](metric_epsilon.py) to get the plots of Figure 2, and set the parameter `metric = 'accuracy'` to get the Test Accuracy plots, and `metric = 'risk'` to get Figures of Test Risk (second row). The corresponding legend is [legend_3](legend/legend_3.pdf). 
* Run the file [accuracy_rho](accuracy_rho.py) to get the Test Accuracy with $\rho_+$ (Figure 3). The corresponding legend can be found in [legend_6](legend/legend_6.pdf).
* Run the file [simulate_distribution_real](simulate_distribution_real.py) to get the plots of Figure 4. 
* Run the file [accuracy_gamma](accuracy_gamma.py) to get the plots of Figure 5: Empirical versus Theoretical Test Accuracy. Take the values $(n, p)= (2000, 20)$ for Low-dimensional setting, and $(n, p)= (200, 200)$ for High-dimension. The corresponding legend is [legend_3](legend/legend_3.pdf).
* Run the file [test_param_estimation](test_param_estimation.py) to get the plot of Figure 6: Estimation of the label noise rates.
* Run the file [train_bce](train_bce.py) to get the plot of Figure 7: Test Accuracy with BCE Loss on Synthetic data.
* Run the file [train_bce_real](train_bce_real.py) to get the plot of Figure 8: Test Accuracy with BCE Loss on real data (DvD dataset).
* Run the file [multi_class_extension](multi_class_extension.py) to get the plot of Figure 9: Multi-class classification.

## Reproducing the values in Table 1:
* Run the file [accuracy_comparison](accuracy_comparison.py) to get values of the table by setting the parameters to the desired values: to choose one of the Amazon dataset, set the parameter `data_type = 'amazon_' + name_data` where `name_data` is in `{'book', dvd, elec, kitchen}`. The results will be found in the folder [results-data](results-data/).

## Datasets:
The real datasets used in our analysis can be found in the folder: [Amazon_review](datasets/Amazon_review/).

## Requirements:
The reader can refer the file `requirements.txt` to use the exact same versions of libraries that we used in our experiments.