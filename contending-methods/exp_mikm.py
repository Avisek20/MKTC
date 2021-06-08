import time
import os
import numpy as np
from mikm import MIKM
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import pairwise_distances


n_jobs = -1

dataset_dir = '../datasets-npz/'
datasets = [
    'digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32', 'yaleb_s32x32',
    'stl10', 'mnist', 'fashion_mnist', 'cifar10_gray', 'cifar100_gray'
]

op_folder = 'res_mikm/'
if not os.path.exists(op_folder):
    os.makedirs(op_folder)
fw = open(op_folder+'res1.txt', 'w')
fw.close()

for i in range(len(datasets)):
    tmp = np.load(dataset_dir + datasets[i] + '.npz', allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    del tmp
    k = len(np.unique(y))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp
    print(datasets[i], X.shape, k)

    n_set = 10
    multi_instance_ari = np.zeros((n_set))
    multi_instance_nmi = np.zeros((n_set))
    data_ari = np.zeros((n_set))
    data_nmi = np.zeros((n_set))


    for i_set in range(n_set):
        tmp = np.load('../data_bags_npz/'+datasets[i]+'/'+ datasets[i] +'_set'+ str(i_set) +'.npz')
        bagX = X[tmp['bagX_idxs']]
        bagY = tmp['bagy']
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], bagX.shape[0]-bag_sidx[-1]))
        all_instances_y = tmp['all_instances_y']
        tmp = None
        #print(i_set, bagX.shape, bagy.shape, bag_sidx.shape, bag_len.shape, all_instances_y.shape, len(np.unique(all_instances_y)))

        start = time.time()
        H = MIKM(bagX, bagY, bag_sidx, bag_len, max_iter=100, n_init=10, n_jobs=n_jobs)
        exec_time = time.time() - start

        bagmem = H.argmax(axis=0)

        if not os.path.exists(op_folder+datasets[i]):
            os.makedirs(op_folder+datasets[i])

        centers = np.zeros((k, X.shape[1]))
        for j in range(k):
            centers[j] = bagX[bagmem==j].mean(axis=0)

        datamem = pairwise_distances(centers, Y=X, n_jobs=n_jobs).argmin(axis=0)

        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_bagmem_set'+ str(i_set) +'.npy', bagmem)
        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_datamem_set'+ str(i_set) +'.npy', datamem)

        bag_ari = ARI(all_instances_y, bagmem)
        bag_nmi = NMI(all_instances_y, bagmem)
        data_ari = ARI(y, datamem)
        data_nmi = ARI(y, datamem)

        with open(op_folder+'res1.txt', 'a') as fa:
            fa.write(
                datasets[i] + ' ' + str(i_set) + ' ' + str(bag_ari) + ' ' + str(bag_nmi) + ' '
                + str(data_ari) + ' ' + str(data_nmi) + '\n'
            )
        print(datasets[i], i_set, bag_ari, bag_nmi, data_ari, data_nmi)
