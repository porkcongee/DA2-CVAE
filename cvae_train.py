from model import VAE
from dataset import ProteinDataset
from loss import bce_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np

def train_CVAE(task, beta=10, n_clusters=4, N_confomation_per_cluster=10000, task_affix=None, batch_size=40, hidden_sizes=[256, 128, 64, 16], latent_size=4, num_epochs=100, loss=bce_loss):
    data_file = f'{task}/coor_xyz_{task}.npy'
    dataset = ProteinDataset(data_file, task, n_clusters, N_confomation_per_cluster)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if task_affix:
        task_name = f'{task}_{task_affix}_n{n_clusters}'
    else:
        task_name = f'{task}_n{n_clusters}'
    dataset.save_scaler(f'{task}/{task_name}')

    input_size = dataset.atom_num * 3
    model = VAE(input_size, hidden_sizes, latent_size, n_clusters)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model.encoder = model.encoder.to(device)
    model.decoder = model.decoder.to(device)

    if next(model.parameters()).is_cuda:
        print("模型在 GPU 上")
    else:
        print("模型在 CPU 上")
    
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    recon_losses = []
    kl_losses = []
    total_losses = []
    for epoch in range(num_epochs):
        time_start = time.time()
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0

        for x in data_loader:
            label = x[:,-n_clusters:]
            x = x.to(device)
            label = label.to(device)
            recon, mu, log_var = model(x, label)
            recon_loss, kl_loss, total_loss = loss(recon, x[:, :-n_clusters], mu, log_var, beta=beta)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_total_loss += total_loss.item()

        recon_losses.append(epoch_recon_loss / len(dataset))
        kl_losses.append(epoch_kl_loss / len(dataset))
        total_losses.append(epoch_total_loss / len(dataset))

        print(f'Epoch {epoch + 1} - Recon Loss: {epoch_recon_loss / len(dataset):.4f} - KL Loss: {epoch_kl_loss / len(dataset):.4f} - Total Loss: {epoch_total_loss / len(dataset):.4f}')
        time_spend = time.time() - time_start
        print(f'Epoch {epoch + 1} - Time: {time_spend}s')
    
        if epoch % 10 == 9:
            torch.save(model.encoder.state_dict(), f'{task}/models/encoder_{task_name}_{epoch}.pth')
            torch.save(model.decoder.state_dict(), f'{task}/models/decoder_{task_name}_{epoch}.pth')

    np.savetxt(f'{task}/loss/recon_{task_name}.txt',np.array(recon_losses))
    np.savetxt(f'{task}/loss/kl_{task_name}.txt',np.array(kl_losses))
    np.savetxt(f'{task}/loss/total_{task_name}.txt',np.array(total_losses))

    torch.save(model.encoder.state_dict(), f'{task}/models/encoder_{task_name}.pth')
    torch.save(model.decoder.state_dict(), f'{task}/models/decoder_{task_name}.pth')
    
    return input_size

if __name__ == '__main__':
    pass
