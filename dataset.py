# This file contains functions to load and preprocess our datasets
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

type_to_path = {
    'book': './datasets/Amazon_review/books.mat',
    'dvd': './datasets/Amazon_review/dvd.mat',
    'elec': './datasets/Amazon_review/elec.mat',
    'kitchen': './datasets/Amazon_review/kitchen.mat',
}
# Amazon review dataset
class Amazon:
    def __init__(self, epsp, epsm, type = 'book', pi = 0.4, n = 1600) -> None:
        self.epsp = epsp
        self.epsm = epsm
        # Load the dataset
        data = loadmat(type_to_path[type])
        self.X = data['fts'] # shape (n, p)

        # Transformation on labels
        self.y = data['labels'].reshape((len(self.X), )).astype(int) # shape (n, ), features are sorted by reversed order (ones then zeros)
        self.y = 1 - 2 * self.y
        
        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        vmu_1 = np.mean(self.X[self.y > 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y < 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1 , vmu_2)))

        # Assure that there is pi negative labels
        positive_labels_idx = np.where(self.y > 0)[0]
        negative_labels_idx = np.where(self.y < 0)[0]
        train_negative = np.random.choice(negative_labels_idx, int(pi * n), replace= True)
        train_positive = np.random.choice(positive_labels_idx, n - int(pi * n), replace= True)
        train_labels_idx = np.concatenate([train_negative, train_positive])
        test_labels_idx = np.setdiff1d(np.arange(0, len(self.y)), train_labels_idx)

        # Train and Test labels
        self.y_train = np.zeros(n)
        self.y_train[:int(pi * n)] = self.y[train_negative]
        self.y_train[int(pi * n):] = self.y[train_positive]
        self.y_test = self.y[test_labels_idx]

        # Train and Test data
        self.X_train = np.zeros((n, self.X.shape[1]))
        self.X_train[:int(pi * n)] = self.X[train_negative]
        self.X_train[int(pi * n):] = self.X[train_positive]
        self.X_test = self.X[test_labels_idx]
        

        # Apply noise
        self.y_train_noisy = apply_noise(self.y_train, epsp, epsm)


# Preprocessing techniques
def apply_noise(y, epsp, epsm):

    length = len(y)
    # either 1 or -1: -1 means flipped
    b1 = -(1 - 2 * np.random.binomial(size=length, n=1, p=epsm))
    b2 = 1 - 2 * np.random.binomial(size=length, n=1, p=epsp) 
    y_noisy = (y == -1) * b1 + (y== 1) * b2
    return y_noisy

def null_indices(l):
    indices = []
    for i in range(len(l)):
        if np.all(l[i] == 0):
            indices.append(i)
    return indices

def center(X, y):
    X_prep = X.copy()
    X_1 = X[(y == -1)]
    X_2 = X[(y == 1)]

    # -mu and +mu
    vmu_1 = np.mean(X_1, axis=0)
    vmu_2 = np.mean(X_2, axis=0)
    vmu = (vmu_2 - vmu_1)/2
    X_prep = X_prep - (vmu_1 + vmu_2) / 2
    return X_prep, vmu
