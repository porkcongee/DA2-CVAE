import mdtraj as md
import numpy as np
import os
import joblib

def pre_recon_xyz(datapath, sample_N, ref, sele, xyz_file, read=True):
    if read and os.path.exists(xyz_file):
        data = np.load(xyz_file)
    else:
        all_coords = []
        batch = 100
        for i in range(0,sample_N,batch):
            pdb_file = [f'{datapath}/protein_{i+x}.pdb' for x in range(batch)]
            traj_tmp = md.load(pdb_file, atom_indices=sele)
            traj_tmp.superpose(ref,0)
            coords = traj_tmp.xyz.reshape(batch, -1)
            all_coords.append(coords)
        data = np.concatenate(all_coords, axis=0)
        np.save(xyz_file, data)
    return data

def get_pca(name):
    assert os.path.exists(f'{name}.pkl'), f'{name}.pkl not exists'
    pca = joblib.load(f'{name}.pkl')
    reduced = np.load(f'{name}.npy')
    return pca, reduced
        

def get_kmeans(name):
    assert os.path.exists(f'{name}.pkl'), f'{name}.pkl not exists'
    kmeans = joblib.load(f'{name}.pkl')
    return kmeans

def relabel(task, task_affix, recon_affix, n_clusters, num_samples, pdbfile='example.pdb'):
    if task_affix:
        task_name = f'{task}_{task_affix}_n{n_clusters}'
    else:
        task_name = f'{task}_n{n_clusters}'

    recon_dir = f'{task}/recon/recon_{task_name}'
    if recon_affix:
        recon_dir = f'{recon_dir}_{recon_affix}'
    os.makedirs(recon_dir, exist_ok=True)
    
    ref_t = md.load(pdbfile)
    sele = ref_t.topology.select('protein and backbone')
    ref = ref_t.atom_slice(sele)
    xyz_file = f'{recon_dir}/coor_xyz_{task_name}_recon.npy'
    data = pre_recon_xyz(recon_dir, num_samples, ref, None, xyz_file, read=True)
    
    pca, reduced = get_pca(f'{task}/pca_{task}')
    kmeans = get_kmeans(f'{task}/kmeans_{task}_n{n_clusters}')
    
    kdist = kmeans.transform(reduced)
    np.save(f'{task}/kdist_{task}_n{n_clusters}.npy', kdist)
    
    recon_reduced = pca.transform(data)
    np.save(f'{task}/pca_{task_name}_recon.npy', recon_reduced)
    recon_ktrajs = kmeans.predict(recon_reduced)
    np.save(f'{task}/kmeans_{task_name}_recon.npy', recon_ktrajs)
    recon_kdist = kmeans.transform(recon_reduced)
    np.save(f'{task}/kdist_{task_name}_recon.npy', recon_kdist)
    
    return recon_reduced, recon_ktrajs


if __name__ == '__main__':
    pass
