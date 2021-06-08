import numpy as np
from sklearn.metrics import pairwise_distances


def gaussian(Pseq, sigma=1):
    return np.exp(-Pseq / sigma ** 2)

def polynomial(Pdot, c=0, d=1):
    return (Pdot + c) ** d

def cosine(Pdot):
    return Pdot / np.outer(np.diag(Pdot), np.diag(Pdot)) ** 0.5


def compute_kernels(X):
    Pseq = pairwise_distances(X, X, metric='sqeuclidean', n_jobs=-1)
    Pdot = X.dot(X.T)
    K = np.zeros((12, X.shape[0], X.shape[0]))
    list_sigmas = [1e-2, 5e-2, 1e-1, 1, 10, 50, 100]
    for i in range(len(list_sigmas)):
        K[i] = gaussian(Pseq, sigma=list_sigmas[i])
    list_c = [0, 1]
    list_d = [2, 4]
    for i in range(len(list_c)):
        for j in range(len(list_d)):
            K[7+i*2+j] = polynomial(Pdot, c=list_c[i], d=list_d[i])
    K[11] = cosine(Pdot)
    return K
