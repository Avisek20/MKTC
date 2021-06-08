'''
###################
Multiple Kernel Transfer Clustering
###################
The function interface: mktc(X, bagX, bagY, bag_sidx, bag_len, list_sigmas, max_iter=100, n_init=10, n_jobs=1)
X : The dataset
bagX : Multi-Instance dataset
bagY : Weak Supervised labels provided to the Multi-Instance dataset
bag_sidx : Starting idx of each bag
bag_len : Length of each bag
list_sigmas : List of Gaussian kernel parameters
max_iter : Maximum number of iterations
n_init : Number of runs of MKTC
n_jobs : Number of parallel processor threads to compute distances and kernel similarities
###################
Returns:
data_mem : The cluster memberships of X
multi_instance_mem : The cluster memberships of the Multi-Instance dataset bagX
w : Multiple kernel parameter learnt on the Multi-Instance dataset
centers : Cluster centers learnt on the Multi-Instance dataset
costs : Objective Function values per iteration for MKMIKM
v_iter : Iteration number on which MKMIKM convergences
###################
'''

import time
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


def gaussian(sqdist, sigma=1):
    K = np.exp(-sqdist / (2 * (sigma * sqdist.max()) ** 2))
    tmp = K.max()
    if tmp == 0:
        tmp = 1
    return K / tmp


def get_bag_mem(dist, Y, bag_sidx, bag_len):
    H = np.zeros(dist.T.shape, dtype=int)
    for j in range(len(bag_len)):
        nzY = Y[j,:]!=0
        if nzY.sum() == 1:
            H[
                nzY,
                bag_sidx[j]+dist[
                    bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY
                ].argmin()
            ] = 1
        elif bag_len[j] == 1:
            H[
                nzY[dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY].argmin()],
                bag_sidx[j]:bag_sidx[j]+bag_len[j]
            ] = 1
        else:
            tmp = np.zeros((bag_len[j], nzY.sum()), dtype=int)
            optimal_assign_idx = linear_sum_assignment(
                dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY]
            )
            pi = min(bag_len[j], nzY.sum())
            idx = dist[optimal_assign_idx].argsort()[::-1][0:pi]
            tmp[optimal_assign_idx[0][idx], optimal_assign_idx[1][idx]] = 1
            H[nzY, bag_sidx[j]:bag_sidx[j]+bag_len[j]] = tmp.T
    return H


def MKMIKM(X, Y, bag_sidx, bag_len, list_sigmas, max_iter=300, n_init=1, n_jobs=1):
    n = X.shape[0]
    k = Y.shape[1]
    m = len(list_sigmas)

    min_cost = +np.inf
    for _ in range(n_init):
        centers = X[np.random.choice(X.shape[0], size=k, replace=True)]
        #
        H = np.zeros((k, n), dtype=int)
        H[np.random.randint(0, k, n), np.arange(n)] = 1
        #
        w = np.ones((m), dtype=np.float32) / m
        #
        sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
        K = np.array(Parallel(n_jobs=7)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
        dist = ((w ** 2)[:,None,None] * (2 - K)).sum(axis=0)
        cost = np.zeros((max_iter)) - 1

        for v_iter in range(max_iter):
            # Update H
            prevH = np.array(H)
            H = get_bag_mem(dist.T, Y, bag_sidx, bag_len)

            # Update centers
            dist2 = (((w ** 2) / ((list_sigmas * sqdist.max()) ** 2))[:,None,None] * K).sum(axis=0)
            for j in range(k):
                tmp = H[j] * dist2[j]
                centers[j] = (tmp[:,None] * X).sum(axis=0) / tmp.sum()

            # Update K
            sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
            K = np.array(Parallel(n_jobs=7)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))

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


def target_clustering(X, bagX, w, H, list_sigmas, n_jobs=1):
    m = w.shape[0]
    n_clusters = H.shape[0]
    sqdist = pairwise_distances(bagX, Y=bagX, n_jobs=n_jobs)
    Kcomb = (
        np.array(Parallel(n_jobs=7)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
        * (w[:,None,None] ** 2)
    ).sum(axis=0)
    term3 = np.zeros((n_clusters))
    for j in range(n_clusters):
        term3[j] = ((Kcomb * H[j]).sum(axis=1) * H[j]).sum() / ((H[j].sum() * (w ** 1).sum()) ** 2)
    sqdist = pairwise_distances(X, Y=bagX, n_jobs=n_jobs)
    K = np.array(Parallel(n_jobs=7)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
    dist = np.zeros((n_clusters, X.shape[0]))
    for j in range(n_clusters):
        dist[j] = (1 - 2 * (
            (H[j] * (K * w[:,None,None] ** 2).sum(axis=0)).sum(axis=1) / (H[j].sum() * (w ** 1).sum())
        ) + term3[j])
    data_mem = dist.argmin(axis=0)
    return data_mem


def mktc(X, bagX, bagY, bag_sidx, bag_len, list_sigmas, max_iter=100, n_init=10, n_jobs=1):
    centers, H, w, costs, v_iter = MKMIKM(bagX, bagY, bag_sidx, bag_len, list_sigmas, max_iter=max_iter, n_init=n_init, n_jobs=n_jobs)
    multi_instance_mem = H.argmax(axis=0)
    data_mem = target_clustering(X, bagX, w, H, list_sigmas, n_jobs=n_jobs)
    return data_mem, multi_instance_mem, w, centers, costs, v_iter


if __name__ == '__main__':
    list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])
    m = list_sigmas.shape[0]
    n_jobs = -1

    #'''
    #dataset = 'mnist'
    #tmp = np.load('datasets-npz/'+dataset+'.npz')
    #X = tmp['X']
    #y = tmp['y']
    #'''
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

    n_sets = 10
    mean_multi_instance_ari = 0
    mean_data_ari = 0

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    for i2 in range(n_sets):
        print(i2+1, '/', n_sets)
        tmp = np.load('data_bags_npz/'+dataset+'/'+dataset+'_set'+str(i2)+'.npz')
        bagX = X[tmp['bagX_idxs']]
        n = bagX.shape[0]
        bagY = tmp['bagy']
        n_clusters = bagY.shape[1]
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], n - bag_sidx[-1]))
        true_y = tmp['all_instances_y']

        start = time.time()
        data_mem, multi_instance_mem, w, centers, costs, v_iter = mktc(
            X, bagX, bagY, bag_sidx, bag_len, list_sigmas, max_iter=100, n_init=2, n_jobs=n_jobs
        )
        exec_time = time.time() - start

        multi_instance_ari = ARI(true_y, multi_instance_mem)
        multi_instance_nmi = NMI(true_y, multi_instance_mem)
        data_ari = ARI(y, data_mem)
        data_nmi = NMI(y, data_mem)
        print('Data ARI:', data_ari, 'NMI:', data_nmi,
            'Multi-Instance ARI:', multi_instance_ari, 'NMI:', multi_instance_nmi, 'exec:', exec_time)
        mean_multi_instance_ari += multi_instance_ari
        mean_data_ari += data_ari

    print('Mean Data ARI:', mean_data_ari/n_sets, ', Mean Multi-Instance ARI:', mean_multi_instance_ari/n_sets)
