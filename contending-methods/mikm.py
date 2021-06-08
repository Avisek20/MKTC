'''
###################
Multi-Instance k-Means
Paper Source: M. T. Law, Y. Yu, R. Urtasun, R. S. Zemel, and E. P. Xing,
“Efficient Multiple Instance Metric Learning using Weakly Supervised Data,”
in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
2017, pp. 576–584.
###################
'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


def get_bag_mem(dist, Y, bag_sidx, bag_len):
    H = np.zeros(dist.T.shape, dtype=bool)
    for j in range(len(bag_len)):
        nzY = Y[j,:]!=0
        if nzY.sum() == 1:
            H[
                nzY,
                bag_sidx[j]+dist[
                    bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY
                ].argmin() # attempt to get argmin of an empty sequence
            ] = 1
        elif bag_len[j] == 1:
            H[
                nzY[dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY].argmin()],
                bag_sidx[j]:bag_sidx[j]+bag_len[j]
            ] = 1
        else:
            tmp = np.zeros((bag_len[j], nzY.sum()), dtype=bool)
            optimal_assign_idx = linear_sum_assignment(
                dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY]
            )
            pi = min(bag_len[j], nzY.sum())
            idx = dist[optimal_assign_idx].argsort()[::-1][0:pi]
            tmp[optimal_assign_idx[0][idx], optimal_assign_idx[1][idx]] = 1
            H[nzY, bag_sidx[j]:bag_sidx[j]+bag_len[j]] = tmp.T
    return H


def MIKM(X, Y, bag_sidx, bag_len, max_iter=100, n_init=1, n_jobs=1):
    n = X.shape[0]
    k = Y.shape[1]

    min_cost = +np.inf
    for _ in range(n_init):
        H = np.zeros((k, n), dtype=bool)
        H[np.random.randint(0, k, X.shape[0]), np.arange(n)] = 1
        for _ in range(max_iter):
            Z = np.zeros((k, X.shape[1]), dtype=np.float32)
            for j in range(k):
                if H[j].sum() > 0:
                    Z[j] = X[H[j]].mean(axis=0)

            dist = pairwise_distances(X, Y=Z, n_jobs=n_jobs)

            prevH = np.array(H, dtype=bool)
            H = get_bag_mem(dist, Y, bag_sidx, bag_len)

            if not (prevH ^ H).sum():
                break

        cost = (H * dist.T).sum()
        if cost < min_cost:
            min_cost = cost
            mincost_H = np.array(H)

    return mincost_H


if __name__ == '__main__':
    import time
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

    n_sets = 10
    mean_multi_instance_ari = 0
    mean_multi_instance_nmi = 0

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    for i2 in range(n_sets):
        print(i2+1, '/', n_sets)

        tmp = np.load('../data_bags_npz/'+dataset+'/'+dataset+'_set'+str(i2)+'.npz')
        bagX = X[tmp['bagX_idxs']]
        n = bagX.shape[0]
        bagY = tmp['bagy']
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], n - bag_sidx[-1]))
        true_y = tmp['all_instances_y']

        start = time.time()
        H = MIKM(
            bagX, bagY, bag_sidx, bag_len, max_iter=100, n_init=2, n_jobs=n_jobs
        )
        exec_time = time.time() - start

        mem = H.argmax(axis=0)

        multi_instance_ari = ARI(true_y, mem)
        multi_instance_nmi = NMI(true_y, mem)

        print('ARI:', multi_instance_ari, 'NMI:', multi_instance_nmi, 'exec:', exec_time)
        mean_multi_instance_ari += multi_instance_ari
        mean_multi_instance_nmi += multi_instance_nmi

    print('Mean ARI:', mean_multi_instance_ari/n_sets)
    print('Mean NMI:', mean_multi_instance_nmi/n_sets)
