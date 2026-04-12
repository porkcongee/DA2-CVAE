import mdtraj as md
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import joblib

def pre_xyz(dcd_list, top, ref, sele, xyz_file, stride=10, read=True):
    if read and os.path.exists(xyz_file):
        traj_tmp = md.load(dcd_list, top=top, stride=stride, atom_indices=sele)
        traj_tmp.superpose(ref,0)
        data = traj_tmp.xyz.reshape(traj_tmp.n_frames, traj_tmp.n_atoms * 3)
        np.save(xyz_file, data)
    else:
        data = np.load(xyz_file)
    return data

def get_pca(task, data=None, read=True, n_components=0.7, reduced_file=True):
    if read and os.path.exists(f'{task}/pca_{task}.pkl'):
        pca = joblib.load(f'{task}/pca_{task}.pkl')
        if reduced_file:
            reduced = np.load(f'{task}/pca_{task}.npy')
        else:
            reduced = pca.transform(data)
    else:
        pca = PCA(n_components)
        reduced = pca.fit_transform(data)
        joblib.dump(pca, f'{task}/pca_{task}.pkl')
        if reduced_file:
            np.save(f'{task}/pca_{task}.npy', reduced)
    return pca, reduced
        

def get_kmeans(task, n_clusters, reduced_data, read=True):
    model_name = f'{task}/kmeans_{task}_n{n_clusters}'
    if read and os.path.exists(f'{model_name}.pkl'):
        kmeans = joblib.load(f'{model_name}.pkl')
        if read and os.path.exists(f'{model_name}.npy'):
            ktrajs = np.load(f'{model_name}.npy')
        else:
            ktrajs = kmeans.fit_predict(reduced_data)
            np.save(f'{model_name}.npy', ktrajs)
    else:
        kmeans = MiniBatchKMeans(n_clusters)
        ktrajs = kmeans.fit_predict(reduced_data)
        joblib.dump(kmeans, f'{model_name}.pkl')
        np.save(f'{model_name}.npy', ktrajs)
    return kmeans, ktrajs

if __name__ == '__main__':
    task = '2RVD'
    
    ref_t = md.load('2RVD.pdb')
    sele = ref_t.topology.select('protein and backbone')
    ref = ref_t.atom_slice(sele)
    top = '2RVD.psf'
    dcd_list = ['2RVD.dcd']
    
    read = True
    PCA_n_components = 0.7
    
    os.makedirs(task, exist_ok=True)
    xyz_file = f'{task}/coor_xyz_{task}.npy'
    data = pre_xyz(dcd_list, top, ref, sele, xyz_file, stride=10)
    
    if read and os.path.exists(f'{task}/pca_{task}.pkl') and os.path.exists(f'{task}/pca_{task}.npy'):
        pca, reduced = get_pca(task, data=None, read=True, n_components=PCA_n_components, reduced_file=True)
    else:
        data = np.load(xyz_file)
        pca, reduced = get_pca(task, data=data, read=read, n_components=PCA_n_components, reduced_file=True)
    
    inertia = []
    for n in range(2,11):
        kmeans, ktrajs = get_kmeans(task, n, reduced, read)
        inertia.append(kmeans.inertia_)
    np.savetxt(f'{task}/kmeans_inertia.dat', inertia)
    