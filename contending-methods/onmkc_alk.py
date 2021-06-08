'''
###################
Optimal Neighborhood Multiple Kernel Clustering with Adaptive Local Kernels
Paper Source: J. Liu, X. Liu, J. Xiong, Q. Liao, S. Zhou, S. Wang, and Y. Yang,
“Optimal Neighborhood Multiple Kernel Clustering with Adaptive Local Kernels,”
IEEE Transactions on Knowledge and Data Engineering, pp. 1–14, 2020.
###################
'''


import numpy as np
import scipy.optimize as sopt


def f_obj(beta, Z, a):
    return Z.dot(beta).dot(beta) + a.dot(beta)

def df_obj(beta, Z, a):
    return (2 * Z).dot(beta) - a


def onmkc_alk(K, n_clusters=2, rho=2**(-1), gamma=0, max_iter=100, n_init=1, tol=1e-5):
    m, n = K.shape[0], K.shape[1]

    # Calculate M
    M = np.zeros((m, m))
    for t1 in range(m):
        for t2 in range(m):
            M[t1,t2] = np.trace(K[t1].dot(K[t2]))

    # Calculate Mi
    Mi = np.zeros((m, m))
    for t1 in range(m):
        for t2 in range(m):
            Mi[t1,t2] = (K[t1] * K[t2]).sum()
    Mi = Mi / n

    min_cost = +np.inf
    for _ in range(n_init):
        beta = np.ones((m)) / m
        J = (beta[:,None,None] * K).sum(axis=0)

        cost = +np.inf
        for v_iter in range(max_iter):
            Kcomb = (beta[:,None,None] * K).sum(axis=0)

            # Update H
            H = np.linalg.svd(J)[0][:,0:n_clusters]

            # Update J
            InHHt = np.eye(n) - H.dot(H.T)
            B = Kcomb - (InHHt / rho)
            U, Sigma, Vt = np.linalg.svd(B)
            Sigma[Sigma < 0] = 0
            J = U.dot(np.diag(Sigma)).dot(Vt)

            # Update beta
            a = np.zeros((m))
            for t in range(m):
                a[t] = -rho * np.trace(J.dot(K[t]))
            Z = (0.5 * rho) * M + Mi
            res = sopt.minimize(fun=f_obj, method='SLSQP', x0=beta,
                jac=df_obj, args=(Z, a),
                constraints=sopt.LinearConstraint(A=np.ones((1,m)), lb=1, ub=1),
                bounds=sopt.Bounds(lb=np.zeros((m)), ub=np.ones((m))))
            beta = res.x

            # Check for convergence
            prev_cost = cost
            cost = (np.trace(J.dot(InHHt)) + ((0.5 * rho) * (np.linalg.norm(J - Kcomb, ord='fro') ** 2))
                + (M.dot(beta).dot(beta)))
            if ((prev_cost - cost) / cost) < tol:
                print(v_iter)
                break

        if min_cost > cost:
            min_cost = cost
            mincost_H = np.array(H)
            mincost_J = np.array(J)
            mincost_beta = np.array(beta)
            micost_v_iter = v_iter
    return mincost_H, mincost_J, mincost_beta, micost_v_iter


if __name__ == '__main__':
    import time
    from compute_kernels import compute_kernels

    '''
    tmp = np.load('../datasets-npz-2/olivetti.npz')
    X = tmp['X']
    y = tmp['y']
    tmp = None
    '''
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
    H, J, beta, v_iter = onmkc_alk(K, n_clusters=n_clusters)
    exec_time = time.time() - start

    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters).fit(H)
    mem = km1.labels_

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    print('ARI:', ARI(y, mem), 'NMI:', NMI(y, mem))
