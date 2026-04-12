import os
from loss import bce_loss, mse_loss
import time
import numpy as np
from cvae_train import train_CVAE
from cvae_recon import recon_CVAE, label_rand_uniform
from reweight import relabel
from get_trainset_cv import get_trainset_cv

if __name__ == '__main__':
    task = '2RVD'
    beta = 1
    n_clusters = 4
    task_affix = None
    N_confomation_per_cluster = 10000
    batch_size = 40
    hidden_sizes = [256, 128, 64, 16]
    latent_size = 4
    num_epochs = 100
    loss = bce_loss
    num_samples = 50000
    pdbfile = '2RVD.pdb'
    recon_affix = 'randU'
    
    os.makedirs(task, exist_ok=True)
    os.makedirs(f'{task}/loss', exist_ok=True)
    os.makedirs(f'{task}/models', exist_ok=True)
    os.makedirs(f'{task}/recon', exist_ok=True)
    os.makedirs(f'{task}/data', exist_ok=True)
    
    print('Training Start!')
    time_start = time.time()
    input_size = train_CVAE(task, beta, n_clusters, N_confomation_per_cluster, task_affix, batch_size, hidden_sizes, latent_size, num_epochs, loss)
    time_spend = time.time() - time_start
    print(f'Training Done! Time consumed: {time_spend}s')
    
    print('Recon Start!')
    time_start = time.time()
    labels = label_rand_uniform(n_clusters, num_samples)
    recon_CVAE(task, num_samples, labels, task_affix, 0, n_clusters, input_size, hidden_sizes, latent_size, pdbfile, recon_affix)
    time_spend = time.time() - time_start
    print(f'Recon Done! Time consumed: {time_spend}s')
    
    print('Relabel Start!')
    time_start = time.time()
    relabel(task, task_affix, recon_affix, n_clusters, num_samples, pdbfile)
    time_spend = time.time() - time_start
    print(f'Relabel Done! Time consumed: {time_spend}s')
    
    with open(f'{task}/data/rmsd.out', 'r') as f:
        cv = f.readlines()
    get_trainset_cv(cv, 'rmsd',task, n_clusters, N_confomation_per_cluster, task_affix)
    
    with open(f'{task}/data/hb.out', 'r') as f:
        cv = f.readlines()
    get_trainset_cv(cv, 'hb',task, n_clusters, N_confomation_per_cluster, task_affix)
