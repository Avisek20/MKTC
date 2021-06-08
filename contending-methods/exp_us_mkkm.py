import time
import os
import numpy as np
from us_mkkm import us_mkkm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI


if __name__ == '__main__':
    n_jobs = -1

    datasets = [
        'digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32', 'yaleb_s32x32',
        'stl10', 'mnist', 'fashion_mnist', 'cifar10_gray', 'cifar100_gray'
    ]

    output_folder = 'res_us_mkkm/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])

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

        start = time.time()
        centers, H, w, costs, v_iter = us_mkkm(X, n_clusters=n_clusters, list_sigmas=list_sigmas, n_jobs=n_jobs)
        exec_time = time.time() - start

        mem = H.argmax(axis=0)

        np.save(output_folder+datasets[adataset]+'_centers.npy', centers)
        np.save(output_folder+datasets[adataset]+'_mem.npy', mem)
        np.save(output_folder+datasets[adataset]+'_H.npy', H)
        np.save(output_folder+datasets[adataset]+'_w.npy', w)
        np.save(output_folder+datasets[adataset]+'_costs.npy', costs)

        with open(output_folder+datasets[adataset]+'_res.txt', 'w') as fw:
            fw.write(str(ARI(y, mem)) + ' ' + str(NMI(y, mem)) + ' ' + str(v_iter) + ' ' + str(exec_time) +  '\n')
