import os
import numpy as np

def get_trainset_cv(cv, cv_name,task, n_clusters, N_confomation_per_cluster, task_affix):
    cv = np.array(cv)
    if task_affix:
        task_name = f'{task}_{task_affix}_n{n_clusters}'
    else:
        task_name = f'{task}_n{n_clusters}'
    ktrajs = np.load(f'{task}/kmeans_{task}_n{n_clusters}.npy')
    for c1 in range(n_clusters):
        c1_list = np.where(ktrajs == c1)
        c1_select = np.load(f'{task}/cluster_n{n_clusters}_{c1}_{N_confomation_per_cluster}.npy')
        with open(f'{task}/data/c{c1}_{task_name}_{cv_name}.out','w') as f:
            f.writelines(cv[c1_list[0]][c1_select])

if __name__ == '__main__':
    pass
