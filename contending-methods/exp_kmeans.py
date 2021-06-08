import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI


op_folder = 'res_exp_kmeans/'
if not os.path.exists(op_folder):
    os.makedirs(op_folder)
fw = open(op_folder+'res1.txt', 'w')
fw.close()

dataset_dir = '../datasets-npz/'
datasets = [
    'digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32', 'yaleb_s32x32',
    'stl10', 'mnist', 'fashion_mnist', 'cifar10_gray', 'cifar100_gray'
]


for i in range(len(datasets)):
    tmp = np.load(dataset_dir + datasets[i] + '.npz', allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    tmp = None
    k = len(np.unique(y))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp
    print(datasets[i], X.shape, k)

    if not os.path.exists(op_folder+datasets[i]):
        os.makedirs(op_folder+datasets[i])

    for j in range(10):
        km1 = KMeans(n_clusters=k, max_iter=100, n_init=10, tol=1e-5).fit(X)
        np.savez_compressed(
            op_folder+datasets[i]+'/'+datasets[i] +'_mem_set'+ str(j) +'.npz', km1.labels_.astype(int)
        )
        ari = ARI(y, km1.labels_)

        print(j, ari)

        with open(op_folder+'res1.txt', 'a') as fa:
            fa.write(datasets[i] + ' ' + str(j) + ' ' + str(ari) + '\n')

    #break
