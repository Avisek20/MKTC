import time
import os
import numpy as np
from mktc import mktc
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import pairwise_distances

list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])
m = list_sigmas.shape[0]
n_jobs = -1

op_folder = 'res_mktc/'
if not os.path.exists(op_folder):
    os.makedirs(op_folder)

dataset_dir = 'datasets-npz/'
datasets = [
    'digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32', 'yaleb_s32x32',
    'stl10', 'mnist', 'fashion_mnist', 'cifar10_gray', 'cifar100_gray'
]

for i in range(len(datasets)):
    tmp = np.load(dataset_dir + datasets[i] + '.npz', allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    tmp = None
    n_clusters = len(np.unique(y))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp
    print(datasets[i], X.shape, n_clusters)

    if not os.path.exists(op_folder+datasets[i]):
        os.makedirs(op_folder+datasets[i])

    n_set = 10
    multi_instance_ari = np.zeros((n_set))
    multi_instance_nmi = np.zeros((n_set))
    data_ari = np.zeros((n_set))
    data_nmi = np.zeros((n_set))


    for i_set in range(n_set):
        print(i_set+1,'/', n_set)
        tmp = np.load(
            'data_bags_npz/'+datasets[i]+'/'+ datasets[i] +'_set'+ str(i_set) +'.npz'
        )
        bagX = X[tmp['bagX_idxs']]
        bagY = tmp['bagy']
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], bagX.shape[0]-bag_sidx[-1]))
        all_instances_y = tmp['all_instances_y']
        tmp = None
        #print(i_set, bagX.shape, bagy.shape, bag_sidx.shape, bag_len.shape, all_instances_y.shape, len(np.unique(all_instances_y)))

        start = time.time()
        data_mem, multi_instance_mem, w, centers, costs, v_iter = mktc(
            X, bagX, bagY, bag_sidx, bag_len, list_sigmas, max_iter=100, n_init=10, n_jobs=n_jobs
        )
        exec_time = time.time() - start

        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_bagmem_set'+ str(i_set) +'.npy', multi_instance_mem)
        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_w_set'+ str(i_set) +'.npy', w)
        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_centers_set'+ str(i_set) +'.npy', centers)
        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_costs_set'+ str(i_set) +'.npy', costs)
        np.save(op_folder+datasets[i]+'/'+datasets[i] +'_datamem_set'+ str(i_set) +'.npy', data_mem)

        multi_instance_ari[i_set] = ARI(all_instances_y, multi_instance_mem)
        multi_instance_nmi[i_set] = NMI(all_instances_y, multi_instance_mem)
        data_ari[i_set] = ARI(y, data_mem)
        data_nmi[i_set] = NMI(y, data_mem)

        #print('BagARI, BagNMI, DataARI, DataNMI:', bag_ari[i_set], bag_nmi[i_set], data_ari[i_set], data_nmi[i_set])

        with open(op_folder+datasets[i]+'/'+datasets[i] +'_res.txt', 'a') as fa:
            fa.write(
                str(i_set) + ' '
                + str(multi_instance_ari[i_set]) + ' ' + str(multi_instance_nmi[i_set]) + ' ' + str(v_iter) + ' '
                + str(data_ari[i_set]) + ' ' + str(data_nmi[i_set]) + ' ' + str(exec_time) + '\n'
            )
    with open(op_folder+datasets[i]+'/'+datasets[i] +'_res.txt', 'a') as fa:
        fa.write(
            '\nAvg. ' + str(multi_instance_ari.mean()) + ' ' + str(multi_instance_nmi.mean())
            + ' ' + str(data_ari.mean()) + ' ' + str(data_nmi.mean()) + '\n'
        )
    #break
