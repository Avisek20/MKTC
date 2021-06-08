'''
###################
Optimal Neighborhood Kernel Clustering with Multiple Kernels
Paper Source: Liu X., Zhou S., Wang Y., Li M., Dou Y., Zhu E., Yin J.,
“Optimal Neighborhood Kernel Clustering with Multiple Kernels”,
in AAAI 2017.
###################
'''

import numpy as np
import scipy.optimize as sopt


def f_obj(gamma, M, a, rho, v_lambda):
    return ((((rho + v_lambda) * 0.5) * M.dot(gamma).dot(gamma))
        - a.dot(gamma))

def df_obj(gamma, M, a, rho, v_lambda):
    return (((rho + v_lambda) * M.dot(gamma)) - a)


def onkcmk(K, n_clusters=2, rho=2**(-4), v_lambda=2**(-7), max_iter=50, n_init=1, tol=1e-6):
    m, n = K.shape[0], K.shape[1]

    # Calculate M
    M = np.zeros((m, m))
    for t1 in range(m):
        for t2 in range(m):
            M[t1,t2] = np.trace(K[t1].dot(K[t2]))

    min_cost = +np.inf
    for _ in range(n_init):
        gamma = np.ones((m)) / m
        G = (gamma[:,None,None] * K).sum(axis=0)

        cost = +np.inf
        for v_iter in range(max_iter):
            Kcomb = (gamma[:,None,None] * K).sum(axis=0)

            # Update H
            U, _, _ = np.linalg.svd(G)
            H = U[:,0:n_clusters]

            # Update G
            InHHt = np.eye(n) - H.dot(H.T)
            B = Kcomb - (InHHt / rho)
            U, Sigma, Vt = np.linalg.svd(B)
            Sigma[Sigma < 0] = 0
            G = U.dot(np.diag(Sigma)).dot(Vt)

            # Update gamma
            a = np.zeros((m))
            for t in range(m):
                a[t] = rho * np.trace(G.dot(K[t]))
            res = sopt.minimize(fun=f_obj, method='SLSQP', x0=gamma,
                jac=df_obj, args=(M, a, rho, v_lambda),
                constraints=sopt.LinearConstraint(A=np.ones((1,m)), lb=1, ub=1),
                bounds=sopt.Bounds(lb=np.zeros((m)), ub=np.ones((m))))
            gamma = res.x

            # Check for convergence
            prev_cost = cost
            cost = (np.trace(G.dot(InHHt))
                + ((rho * 0.5) * (np.linalg.norm(G - Kcomb, ord='fro') ** 2))
                + ((v_lambda * 0.5) * M.dot(gamma).dot(gamma)))

            if (np.abs(prev_cost - cost) / cost) < tol:
                print(v_iter)
                break

        if min_cost > cost:
            min_cost = cost
            mincost_H = np.array(H)
            mincost_G = np.array(G)
            mincost_gamma = np.array(gamma)
            micost_v_iter = v_iter
    return mincost_H, mincost_G, mincost_gamma, micost_v_iter


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
    H, G, gamma, v_iter = onkcmk(K, n_clusters=n_clusters)
    exec_time = time.time() - start

    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters).fit(H)
    mem = km1.labels_

    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.metrics import normalized_mutual_info_score as NMI

    print('ARI:', ARI(y, mem), 'NMI:', NMI(y, mem))
