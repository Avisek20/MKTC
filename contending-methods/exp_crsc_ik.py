import os
import time
import numpy as np
from compute_kernels import compute_kernels
from crsc_ik import crsc_ik
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI


if __name__ == '__main__':
    datasets = [
        'digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32', 'yaleb_s32x32',
        'stl10', 'mnist', 'fashion_mnist', 'cifar10_gray', 'cifar100_gray'
    ]

    output_folder = 'res_crsc_ik/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for adataset in range(len(datasets)):
        print(datasets[adataset])
        tmp = np.load('../datasets-npz/'+datasets[adataset]+'.npz')
        X = tmp['X']
        y = tmp['y']
        tmp = None

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
        H, w, v_iter = crsc_ik(K, n_clusters=n_clusters, max_iter=100, n_init=10)
        exec_time = time.time() - start

        np.save(output_folder+datasets[adataset]+'_H.npy', H)
        np.save(output_folder+datasets[adataset]+'_w.npy', w)

        from sklearn.cluster import KMeans
        km1 = KMeans(n_clusters=n_clusters).fit(H)
        mem = km1.labels_

        with open(output_folder+datasets[adataset]+'_res.txt', 'w') as fw:
            fw.write(str(ARI(y, mem)) + ' ' + str(NMI(y, mem)) + ' ' + str(v_iter) + ' ' + str(exec_time) +  '\n')
