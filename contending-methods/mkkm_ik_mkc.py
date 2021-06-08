'''
###################
Multiple Kernel k-Means with Incomplete Kernels
Paper Source: X. Liu, X. Zhu, M. Li, L. Wang, E. Zhu, T. Liu, M. Kloft, D. Shen, J. Yin, and W. Gao,
“Multiple Kernel k-Means with Incomplete Kernels,”
IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 42, no. 5, pp. 1191–1204, 2020.
###################
'''


import numpy as np
import scipy.optimize as sopt


def f_obj(beta, Z, f):
    return (0.5 * beta.dot(Z.dot(beta)) - f.dot(beta))

def df_obj(beta, Z, f):
    return (Z.dot(beta) - f)


def mkkm_ik_mkc(K, n_clusters=2, s=None, v_lambda=1, max_iter=100, n_init=1, tol=1e-5):
    m, n = K.shape[0], K.shape[1]
    if s == None:
        s = np.ones((m, n))

    min_cost = +np.inf
    for v_init in range(n_init):
        cost = +np.inf
        for t in range(m):
            K[t,~(s[t].astype('bool'))] = 0

        beta = np.ones((m)) / m
        for v_iter in range(max_iter):
            Kcomb = ((beta ** 2)[:,None,None] * K).sum(axis=0)

            # Update H
            H = np.linalg.svd(Kcomb)[0][:,0:n_clusters]

            # Update K
            for t in range(m):
                T = np.zeros((n, n))
                for t2 in range(m):
                    if t2 != t:
                        T += K[t2] * ((beta[t] + beta[t2] - ((m-2)*beta[t]*beta[t2]))
                            / (1 + (m-1) * (beta[t] ** 2)))
                T = T - (((beta[t] ** 2) * (np.eye(n) - H.dot(H.T)))
                    / (v_lambda * (1 + (m-1) * (beta[t] ** 2))))
                K[t] = T
                K[t,~(s[t].astype('bool'))] = 0
                u, sigma, vt = np.linalg.svd(K[t])
                sigma[sigma < 0] = 0
                K[t] = u.dot(np.diag(sigma)).dot(vt)

            # Update beta
            A = np.zeros((m, m)) + (m - 2)
            A[np.diag(np.ones((m))).astype('bool')] = m - 1
            d = np.zeros((m))
            for t in range(m):
                d[t] = np.trace(K[t].dot(np.eye(n) - H.dot(H.T)))
            M = np.zeros((m, m))
            for t1 in range(m):
                for t2 in range(m):
                    M[t1,t2] = np.trace(K[t1].dot(K[t2]))
            Z = ((A*M) + ((2/v_lambda) * np.diag(d)))
            f = M.sum(axis=1) - np.diag(M)
            res = sopt.minimize(fun=f_obj, method='SLSQP', x0=beta,
                jac=df_obj, args=(Z, f),
                constraints=sopt.LinearConstraint(A=np.ones((1,m)), lb=1, ub=1),
                bounds=sopt.Bounds(lb=np.zeros((m)), ub=np.ones((m))))
            prev_beta = np.array(beta)
            beta = res.x

            # Check for convergence
            if max(np.abs(prev_beta - beta)) < tol:
                print(v_iter)
                break

        prev_cost = cost
        cost = np.trace(((beta ** 2)[:,None,None] * K).sum(axis=0).dot(np.eye(n) - H.dot(H.T)))
        for t in range(m):
            cost += ((0.5 * v_lambda) * (
                np.linalg.norm(
                    K[t] - (np.delete(beta, t)[:,None,None] * (np.delete(K, t, axis=0))).sum(axis=0) , ord='fro'
                ) ** 2
            ))
        if min_cost > cost:
            mincost_H = np.array(H)
            mincost_beta = np.array(beta)
            mincost_v_iter = v_iter

    return mincost_H, mincost_beta, mincost_v_iter


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
    H, beta, v_iter = mkkm_ik_mkc(K, n_clusters=n_clusters)
    exec_time = time.time() - start

    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters).fit(H)
    mem = km1.labels_

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    print('ARI:', ARI(y, mem), 'NMI:', NMI(y, mem))
