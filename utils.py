# This file contains all the util functions used in almost every experiment reported in the papaer.
import numpy as np
import random
from tqdm.auto import tqdm
import dataset

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Data generation
def gaussian_mixture(vmu, n, pi=0.5, cov = False):
    p = len(vmu)
    y = np.ones(n)
    y[:int(n * pi)] = -1
    Z = np.random.randn(p, n) # (z_1, ..., z_n)
    # Adding a covariance matrix
    if cov : 
        M_2 = np.random.rand(p, p)
        C_2 = M_2.T @ M_2 / p
        Z[:, int(n * pi):] = C_2 @ Z[:, int(n * pi):]
    X = np.outer(vmu, y) + Z # np.outer = vmu.T @ y
    return X, y

def generate_data(p, n, epsp, epsm, mu, pi):  
    """
    Function to generate synthetic data
    params:
        p (int): dimension of a single data vector
        n (int): total number of data vectors
        epsm (float): epsilon minus, in (0,1)
        epsp (float): epsilon plus
        mu (float): norm of the mean of vectors
        pi (int): n*pi is the proportion of negative labels
    """
    vmu = np.zeros(p) # vecteur ligne: shape = (p,)
    vmu[0] = mu # vmu is of norm mu
    
    X_train, y_train = gaussian_mixture(vmu, n, pi)
    X_test, y_test = gaussian_mixture(vmu, 2*n)
    
    y_train_noisy = np.zeros_like(y_train)
    
    b = 1 - 2 * np.random.binomial(size=int(n * pi), n=1, p=epsm)
    y_train_noisy[:int(n * pi)] = b * y_train[:int(n * pi)]
    
    b = 1 - 2 * np.random.binomial(size=n - int(n * pi), n=1, p=epsp)
    y_train_noisy[int(n * pi):] = b * y_train[int(n * pi):]
    
    return (X_train, y_train, y_train_noisy), (X_test, y_test)

# Binary accuracy function
def accuracy(y, y_pred):
    acc = np.mean(y == y_pred)
    return max(acc, 1 - acc)

# Decision functions:

# Naive w
fw = lambda X, y, gamma: np.linalg.inv(X @ X.T / X.shape[1] + gamma * np.eye(X.shape[0])) @ X @ y / X.shape[1]  

# g(x) = <w, x>
g = lambda w, X: X.T @ w

# Labelling
decision = lambda w, X: 2 * (g(w, X) >= 0) - 1

fepsm = lambda y, epsm, epsp: epsp * (-y==1) + epsm * (y==1)
fepsp = lambda y, epsm, epsp: epsm * (-y==1) + epsp * (y==1)

# D_+ and D_-
Dp = lambda y, rhom, rhop: np.diag((1 - fepsm(y, rhom, rhop) + fepsp(y, rhom, rhop)) / (1 - rhom - rhop))

# w improved
fw_imp = lambda X, y, gamma, rhom, rhop: np.linalg.inv(X @ X.T / X.shape[1] + gamma * np.eye(X.shape[0])) @ X @ Dp(y, rhom, rhop) @ y / X.shape[1]

def classifier_vector(classifier, X, y, gamma, rhop, rhom):
    if 'improved' in classifier:
        return fw_imp(X, y, gamma, rhom, rhop)
    else: # naive
        return fw(X, y, gamma)
    
def empirical(p, n, eps, classifier='naive', gamma = 1e-2):
    """
    Computes the train and test accuracies
    Params:
        p (int)
        n (int)
        eps (float): epsilon plus
        classifier (str): either 'naive' or 'improved'
        gamma (float): regularization parameter
    Returns:
        A dictionary {'train':.., 'train-noisy':..., 'eval':..}
    """
    epsm = .01
    (X, y, y_noisy), (Xt, yt) = generate_data(p, n, epsp=eps, epsm=epsm, mu=3, pi=0.3)
    
    w = classifier_vector(classifier, X, y_noisy, gamma, eps, epsm)

    acc = {}
    
    y_pred = decision(w, X)
    acc['train'] = accuracy(y, y_pred)
    acc['train-noisy'] = accuracy(y_noisy, y_pred)
    
    yt_pred = decision(w, Xt)
    acc['eval'] = accuracy(yt, yt_pred)
    
    return acc

# Losses
def L2_loss(w, X, y):
    # X of shape (p, n)
    return np.mean((X.T @ w - y)**2)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def bce_loss(w, x, y):
    # y in {-1, 1}
    # -y log(sigmoid(w^T x)) - (1 - y)log(1 - sigmoid(w^T x))
    # transform y to be in {0, 1}
    y = (y + 1) / 2
    return -y * np.log(sigmoid(np.sum(w * x))) - (1 - y) * np.log(1 - sigmoid(np.sum(w * x)))

def improved_bce_loss(w, x, y, rhop, rhom):
    # y in {-1, 1}
    beta = 1 - rhop - rhom
    if y > 0:
        return ((1 - rhom) * bce_loss(w, x, y) - rhop * bce_loss(w, x, -y)) * beta
    else: # y = -1
        return ((1 - rhop) * bce_loss(w, x, y) - rhom * bce_loss(w, x, -y)) * beta

# Gradients
def bce_grad(w, x, y):
    # x and w must be of the same shape
    # y in {-1, 1}
    y = (y + 1) / 2
    return (sigmoid(np.sum(w * x)) - y) * x

def improved_bce_grad(w, x, y, rhop, rhom):
    # y in {-1, 1}
    beta = 1 - rhop - rhom
    if y > 0:
        return ((1 - rhom) * bce_grad(w, x, y) - rhop * bce_grad(w, x, -y)) * beta
    else: # y = -1
        return ((1 - rhop) * bce_grad(w, x, y) - rhom * bce_grad(w, x, -y)) * beta

# BCE gradient descent
def grad_descent(X, y, lr, gamma, rhop, rhom, N):
    # X of shape (n, p)
    n, p = X.shape
    w = np.zeros(p)
    losses = []
    for k in range(N):
        loss = 0
        grad_step = 0
        for i in range(n):
            grad_step += improved_bce_grad(w, X[i], y[i], rhop, rhom) / n
            loss += improved_bce_loss(w, X[i], y[i], rhop, rhom) / n 
        loss += gamma * np.sum(w**2)
        losses.append(loss)
        # step
        w = (1 - 2 * gamma) * w - lr * grad_step 
    return w, losses

def empirical_accuracy(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_train, y_train, y_train_noisy), (X_test, y_test) = generate_data(p, n, epsp, epsm, mu, pi)
        elif 'amazon' in data_type:
            type = data_type.split('_')[1]
            data = dataset.Amazon(epsp, epsm, type, pi, n)
            X_train, y_train_noisy = data.X_train.T, data.y_train_noisy
            X_test, y_test = data.X_test.T, data.y_test

        w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)

        res += accuracy(y_test, decision(w, X_test))
    return res / batch

def empirical_mean(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_train, y_train, y_train_noisy), (X_test, y_test) = generate_data(p, n, epsp, epsm, mu, pi)
        elif 'amazon' in data_type:
            type = data_type.split('_')[1]
            data = dataset.Amazon(epsp, epsm, type, pi, n)
            X_train, y_train_noisy = data.X_train.T, data.y_train_noisy
            X_test, y_test = data.X_test.T, data.y_test

        w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)

        res += np.mean(y_test * (X_test.T @ w))
    return res / batch

def empirical_mean_2(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_train, y_train, y_train_noisy), (X_test, y_test) = generate_data(p, n, epsp, epsm, mu, pi)
        elif 'amazon' in data_type:
            type = data_type.split('_')[1]
            data = dataset.Amazon(epsp, epsm, type, pi, n)
            X_train, y_train_noisy = data.X_train.T, data.y_train_noisy
            X_test, y_test = data.X_test.T, data.y_test

        w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)
        res += np.mean((X_test.T @ w)**2)
    return res / batch

def empirical_risk(classifier, batch, n, p, mu, epsp, epsm, rhop, rhom, gamma, pi, data_type):
    res = 0
    for i in range(batch):
        # generate new data
        if 'synthetic' in data_type:
            (X_train, y_train, y_train_noisy), (X_test, y_test) = generate_data(p, n, epsp, epsm, mu, pi)
        elif 'amazon' in data_type:
            type = data_type.split('_')[1]
            data = dataset.Amazon(epsp, epsm, type, pi, n)
            X_train, y_train_noisy = data.X_train.T, data.y_train_noisy
            X_test, y_test = data.X_test.T, data.y_test

        w = classifier_vector(classifier, X_train, y_train_noisy, gamma, rhop, rhom)
        res += L2_loss(w, X_test, y_test)
    return res / batch

# Checking coherence of couples (epsp, epsm)
def check_coherence(epsp, epsm):
    if epsp < 0 or epsm < 0:
        return False
    elif epsp + epsm > 1:
        return False
    else:
        return True
    
# Gaussian density function
def gaussian(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

######## Multi-class extension #########

def gaussian_mixture_multi(n, p, pis, mus):
    assert np.sum(pis) == 1
    assert len(pis) == len(mus)
    k = len(pis) # the number of classes
    
    # means
    vmus = np.zeros((k, p))
    vmus[:, 0] = mus

    # data and labels: here its Y^T
    y = np.zeros(n) # labels 0, 1, ..., k-1
    Y = np.zeros((n, k))
    X = np.zeros((n, p))
    count = 0
    for l in range(k):
        label = np.zeros(k)
        label[l] = 1
        if l == k-1:
            Y[count:] = label
            y[count:] = l
            X[count:] = vmus[l]
        else:
            Y[count: count + int(n * pis[l])] = label
            y[count: count + int(n * pis[l])] = l
            X[count: count + int(n * pis[l])] = vmus[l]
            count += int(n * pis[l])

    # data
    Z = np.random.randn(n, p)
    X = X + Z
    return X, y, Y # return of shape: X (n, p)  y of shape (n,) and Y (n, k)

def generate_data_multi(n, p, pis, mus, epsilons, train = True):
    assert epsilons.shape[0] == epsilons.shape[1] # shape (k, k)
    # epsilons contain one non null element in each row/column
    k = len(pis)

    if not train:
        # Test data
        X_test, y_test, Y_test = gaussian_mixture_multi(n, p, pis, mus)
        return X_test, Y_test
        

    else:
        # Train data
        X_train, y_train, Y_train = gaussian_mixture_multi(n, p, pis, mus)
    
        # Noise the labels
        y_train_noisy = y_train.copy()
        for i in range(k):
            j = np.argmax(epsilons[i])
            # j is the new class for class i
            b = np.random.binomial(size= int(n * pis[i]), n=1, p=epsilons[i, j])
            y_train_noisy[(y_train == i)] = i * (1 - b) + j * b

        # Sort values in labels and data
        indices = np.argsort(y_train_noisy)
        X_train = X_train[indices]
        y_train_noisy = sorted(y_train_noisy)
        y_train_noisy = np.array(y_train_noisy).astype(int)

        # Create the One-Hot Encoding
        Y_train_noisy = np.zeros((n, k))

        for l in range(k):
            label = np.zeros(k)
            label[l] = 1
            Y_train_noisy[(y_train_noisy == l)] = label

        return X_train, Y_train, Y_train_noisy

def accuracy_multi(Y_pred, Y_true):
    # Ys are of shape (n, k)
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(Y_pred, axis=1)
    return np.mean(y_pred == y_true)

def multi_classifier(X, Y, alphas, betas, gamma):
    # X of shape (n, p)
    # Y of shape (n, k)
    # alphas of shape (k,)
    assert X.shape[0] == Y.shape[0]
    assert len(alphas) == Y.shape[1]
    assert len(alphas) == len(betas)
    n, p = X.shape
    k = Y.shape[1]

    Q = np.linalg.solve(X.T @ X / n + gamma * np.eye(p), np.eye(p))
    M = np.ones_like(Y)
    d_alpha = np.diag(alphas)
    d_beta = np.diag(betas)
    W = Q @ X.T @ (Y @ d_alpha + (M - Y) @ d_beta) / n
    return W 

def find_optimal_gamma_multi(X_train, Y_noisy, X_test, Y_test, params):
    k = Y_noisy.shape[1]
    gammas = np.logspace(-6, 2, 9)
    accs = []
    for gamma in gammas:
        W = multi_classifier(X_train, Y_noisy, params[:k], params[k:], gamma)
        accs.append(accuracy_multi(X_test @ W, Y_test))
        
    return gammas[np.argmax(accs)]
