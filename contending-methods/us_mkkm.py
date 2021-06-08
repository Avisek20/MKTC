'''
###################
Unsupervised Multiple Kernel k-Means
###################
'''


import time
import numpy as np
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed


def gaussian(sqdist, sigma=1):
    K = np.exp(-sqdist / (2 * (sigma * sqdist.max()) ** 2))
    tmp = K.max()
    if tmp == 0:
        tmp = 1
    return K / tmp


def us_mkkm(X, n_clusters, list_sigmas, max_iter=100, n_init=10, n_jobs=1):
    n = X.shape[0]
    m = len(list_sigmas)

    min_cost = +np.inf
    for _ in range(n_init):
        centers = X[np.random.choice(X.shape[0], size=n_clusters, replace=True)]
        #
        H = np.zeros((n_clusters, n), dtype=int)
        H[np.random.randint(0, n_clusters, n), np.arange(n)] = 1
        #
        w = np.ones((m), dtype=np.float32) / m
        #
        sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
        K = np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
        dist = ((w ** 2)[:,None,None] * (2 - K)).sum(axis=0)
        cost = np.zeros((max_iter)) - 1

        for v_iter in range(max_iter):
            # Update H
            prevH = np.array(H)
            mem = dist.argmin(axis=0)
            H = np.zeros(dist.shape, dtype=int)
            H[mem, np.arange(n)] = 1

            # Update centers
            dist2 = (((w ** 2) / ((list_sigmas * sqdist.max()) ** 2))[:,None,None] * K).sum(axis=0)
            for j in range(n_clusters):
                if H[j].sum() > 0:
                    tmp = H[j] * dist2[j]
                    centers[j] = (tmp[:,None] * X).sum(axis=0) / tmp.sum()

            # Update K
            sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
            K = np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))

            # Update w
            a = 1 / np.fmax((H[None, :, :] * (2 - K)).sum(axis=-1).sum(axis=-1), 1e-6)
            w = a / a.sum()

            dist = ((w ** 2)[:,None,None] * (2 - K)).sum(axis=0)
            cost[v_iter] = (H * dist).sum()

            if not (prevH ^ H).sum():
                print('break at', v_iter)
                break

        if cost[min(v_iter, max_iter)] < min_cost:
            min_cost = cost[min(v_iter, max_iter)]
            mincost_centers = np.array(centers)
            mincost_H = np.array(H)
            mincost_w = np.array(w)
            saved_costs = np.array(cost)
            mincost_v_iter = min(v_iter, max_iter)

    return mincost_centers, mincost_H, mincost_w, saved_costs, mincost_v_iter


if __name__ == '__main__':
    list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])
    m = list_sigmas.shape[0]
    n_jobs = -1

    dataset = 'digits'
    from sklearn.datasets import load_digits
    X = load_digits().data
    y = load_digits().target
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp
    n_clusters = len(np.unique(y))

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    start = time.time()
    centers, H, w, costs, v_iter = us_mkkm(
        X, n_clusters, list_sigmas=list_sigmas, n_init=10, n_jobs=n_jobs
    )
    exec_time = time.time() - start

    mem = H.argmax(axis=0)

    ari = ARI(y, mem)
    nmi = NMI(y, mem)
    print('ARI:', ari, 'NMI:', nmi)
