'''
###################
Co-regularized Spectral Clustering with Improved Kernels
Paper Source: Wang, Y., Liu, X., Dou, Y. and Li, R.,
“Multiple kernel clustering framework with improved kernels”,
in IJCAI 2017, pp. 2999-3005, 2017.
###################
'''


import numpy as np
import scipy.optimize as sopt


def f_obj(w, Z, M, v_lambda):
    return 0.5 * (2*Z + v_lambda*M).dot(w).dot(w)

def df_obj(w, Z, M, v_lambda):
    return (2*Z + v_lambda*M).dot(w)


def kernel_kmeans(Kt, n_clusters=2, max_iter=100, n_init=5, tol=1e-5):
    U, _, _ = np.linalg.svd(Kt)
    H = U[:,0:n_clusters]
    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, tol=tol).fit(H)
    mem = km1.labels_
    dist = np.zeros((Kt.shape[0], n_clusters))
    diagKt = np.diag(Kt)
    for j in range(n_clusters):
        if (mem==j).sum() > 1:
            dist[:,j] = (diagKt
                - (2 * Kt[:,mem==j].sum(axis=1) / (mem==j).sum())
                + (Kt[mem==j,:][:,mem==j].sum() / ((mem==j).sum()**2)))
    return mem, dist


def crsc_ik(K, n_clusters=2, r=0.05, v_lambda=0.5, max_iter=100, n_init=1, tol=1e-5):
    m, n = K.shape[0], K.shape[1]

    M = np.zeros((m, m))
    for t1 in range(m):
        for t2 in range(m):
            M[t1,t2] = np.trace(K[t1].T.dot(K[t2]))

    min_cost = +np.inf
    for _ in range(n_init):
        w = np.ones((m)) / m
        impK = np.array(K)

        cost = +np.inf
        for v_iter in range(max_iter):
            # Identify outliers and non-outliers
            non_outliers = np.zeros((m, n))
            for t in range(m):
                # Run kernel kmeans
                kkm_mem, kkm_dist = kernel_kmeans(impK[t], n_clusters=n_clusters,
                    max_iter=max_iter, tol=tol)
                # Get non-outliers
                for j in range(n_clusters):
                    non_outliers[t,
                        np.argsort(kkm_dist[kkm_mem==j,j])[0:int(n*(1-r))]] = 1

            # Optimize Kcomb
            Kcomb = ((w[:,None,None] ** 2) * impK).sum(axis=0)

            # Update H
            U, _, _ = np.linalg.svd(Kcomb)
            H = U[:,0:n_clusters]

            # Update w
            Z = np.zeros((m))
            Q = np.eye(n) - H.dot(H.T)
            for t in range(m):
                Z[t] = np.trace(K[t].dot(Q))
            Z = np.diag(Z)
            res = sopt.minimize(fun=f_obj, method='SLSQP', x0=w,
                jac=df_obj, args=(Z, M, v_lambda),
                constraints=sopt.LinearConstraint(A=np.ones((1,m)), lb=1, ub=1),
                bounds=sopt.Bounds(lb=np.zeros((m)), ub=np.ones((m))))
            w = res.x

            # Update impK
            impK = np.array(K)
            for t in range(m):
                Kcc = np.array(K[t,np.where(non_outliers[t]==1)[0][:,None],non_outliers[t]==1])
                Qcm = np.array(Q[np.where(non_outliers[t]==1)[0][:,None],non_outliers[t]==0])
                Qmm = np.array(Q[np.where(non_outliers[t]==0)[0][:,None],non_outliers[t]==0])
                impK[t,np.where(non_outliers[t]==1)[0][:,None],non_outliers[t]==0] = -(Kcc.dot(Qcm).dot(Qmm))
                impK[t,np.where(non_outliers[t]==0)[0][:,None],non_outliers[t]==1] = \
                    -Qmm.dot(Qcm.T).dot(Kcc)
                impK[t,np.where(non_outliers[t]==0)[0][:,None],non_outliers[t]==0] = \
                    -(impK[t,np.where(non_outliers[t]==0)[0][:,None],non_outliers[t]==1].dot(Qcm).dot(Qmm))

            rho = 0
            for t1 in range(m):
                rho = rho + np.trace(K[t1].dot(impK[t1].T)) / ((np.trace(K[t1].dot(K[t1].T)) * np.trace(impK[t1].dot(impK[t1].T))) ** 0.5)
            rho = rho / m

            # Check for convergence
            prev_cost = cost
            cost = (np.trace(((w[:,None,None] ** 2) * impK).sum(axis=0).dot(Q))
                + (0.5 * v_lambda * M.dot(w).dot(w)))
            if rho < 0.7 or np.abs(prev_cost - cost) < 1e-9:
                print(v_iter)
                break

        if min_cost > cost:
            min_cost = cost
            mincost_H = np.array(H)
            mincost_w = np.array(w)
            mincost_v_iter = v_iter
    return mincost_H, mincost_w, mincost_v_iter


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
    H, w, v_iter = crsc_ik(K, n_clusters=n_clusters)
    exec_time = time.time() - start

    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters).fit(H)
    mem = km1.labels_

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    print('ARI:', ARI(y, mem), 'NMI:', NMI(y, mem))
