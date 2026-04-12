import os
import numpy as np
import prody
import torch
from model import VAE
import joblib

def label_rand_uniform(n_cluster, num_samples):
    label = np.random.dirichlet(np.ones(n_cluster),size=num_samples).astype(np.float32) 
    print(label[:10])
    return label

def recon_CVAE(task, num_samples, labels, task_affix, start=0, n_clusters=4, input_size=120, hidden_sizes=[256, 128, 64, 16], latent_size=4, pdbfile='example.pdb', recon_affix='randU'):
    if task_affix:
        task_name = f'{task}_{task_affix}_n{n_clusters}'
    else:
        task_name = f'{task}_n{n_clusters}'
    scaler = joblib.load(f'{task}/{task_name}.pkl')

    recon_dir = f'{task}/recon/recon_{task_name}'
    if recon_affix:
        recon_dir = f'{recon_dir}_{recon_affix}'
    os.makedirs(recon_dir, exist_ok=True)

    z_samples = torch.randn(num_samples, latent_size)
    labels = torch.from_numpy(labels)

    model = VAE(input_size, hidden_sizes, latent_size, n_clusters)
    model.eval()

    encoder_state_dict = torch.load(f'{task}/models/encoder_{task_name}.pth')
    decoder_state_dict = torch.load(f'{task}/models/decoder_{task_name}.pth')
    model.encoder.load_state_dict(encoder_state_dict)
    model.decoder.load_state_dict(decoder_state_dict)

    with torch.no_grad():
        protein = model.decoder(z_samples, labels).detach().cpu().numpy()
        protein = scaler.inverse_transform(protein)
        print(protein.shape)

    pdb= prody.parsePDB(pdbfile)
    structure = pdb.select('protein and backbone or name OT1')

    for index, prot in enumerate(protein):
        coords = prot.reshape(-1,3)
        assert len(coords) == len(structure), 'Atom Number wrong!'
        structure.setCoords(coords * 10)
        prody.writePDB(f'{recon_dir}/protein_{start+index}.pdb', structure)

if __name__ == '__main__':
    pass
