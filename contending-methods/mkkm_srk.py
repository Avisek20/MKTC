'''
###################
Multiple Kernel k-Means Clustering by Selecting Representative Kernels
Paper Source: Y. Yao, Y. Li, B. Jiang, and H. Chen,
“Multiple Kernel k-Means Clustering by Selecting Representative Kernels,”
IEEE Transactions on Neural Networks and Learning Systems, pp. 1–14, 2020.
###################
'''


import numpy as np
import scipy.optimize as sopt


def f_obj(Y, D, v_lambda, C):
    Y = Y.reshape(D.shape[0], D.shape[0]).T
    return (np.diag(D).dot(Y.sum(axis=1)).dot(Y.sum(axis=1)) / (D.shape[0] ** 2)
        + v_lambda * np.trace(C.T.dot(Y)))

def df_obj(Y, D, v_lambda, C):
    Y = Y.reshape(D.shape[0], D.shape[0]).T
    return (2 * D * Y.sum(axis=1)) / (D.shape[0] ** 2) + v_lambda * C


def mkkm_srk(K, n_clusters=2, v_lambda=0.01, max_iter=100, n_init=1, tol=1e-5):
    m, n = K.shape[0], K.shape[1]
    min_cost = +np.inf

    C = np.zeros((m, m))
    for t1 in range(m):
        for t2 in range(m):
            C[t1, t2] = np.trace(K[t1].T.dot(K[t2]))

    A1 = np.zeros((m, m ** 2))
    for t in range(m):
        A1[t,t*m:t*m+m] = 1

    cons = []
    for t in range(m):
        cons.append(sopt.LinearConstraint(A=A1[t], lb=1, ub=1))

    for _ in range(n_init):
        Y = np.random.rand(m, m)
        Y = Y / Y.sum(axis=0)
        w = Y.sum(axis=1)

        cost = +np.inf
        for v_iter in range(max_iter):
            Kcomb = ((w ** 2)[:,None,None] * K).sum(axis=0)

            # Update H
            u, _, _ = np.linalg.svd(Kcomb)
            H = u[:,0:n_clusters]

            # Update Y
            InHHt = np.eye(n) - H.dot(H.T)
            D = np.zeros((m))
            for t in range(m):
                D[t] = np.trace(K[t].dot(InHHt))
            res = sopt.minimize(fun=f_obj, method='SLSQP', x0=Y.T.flatten(),
                jac=df_obj, args=(D, v_lambda, C),
                constraints=cons,
                bounds=sopt.Bounds(lb=np.zeros((m**2)), ub=np.ones((m**2))))
            #print(res.message)
            Y = res.x.reshape(m,m).T

            # Update w
            w = Y.mean(axis=1)

            # Check for convergence
            prev_cost = cost
            cost = np.trace(((w ** 2)[:,None,None] * K).sum(axis=0).dot(InHHt))
            if np.abs(prev_cost - cost) < tol:
                print('v_iter', v_iter)
                break

        if min_cost > cost:
            min_cost = cost
            mincost_w = np.array(w)
            mincost_H = np.array(H)
            mincost_v_iter = v_iter
    return mincost_w, mincost_H, mincost_v_iter


if __name__ == '__main__':
    import time
    from compute_kernels import compute_kernels

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

    K = compute_kernels(X)
    for i in range(K.shape[0]):
        K[i] = K[i] / K[i].max()

    start = time.time()
    w, H, v_iter = mkkm_srk(K, n_clusters=n_clusters)
    exec_time = time.time() - start

    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters).fit(H)
    mem = km1.labels_

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    print('ARI:', ARI(y, mem), 'NMI:', NMI(y, mem))
