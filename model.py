import torch
import torch.nn as nn

class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau : float in range (0,1)
        self.theta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.bn = nn.BatchNorm1d(dim_z, affine=True)
        self.bn.bias.requires_grad = False

        self.mu = mu

    def forward(self, x):  # x: (batch_size, dim_z)
        if self.mu:
            gamma = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))
        else:
            gamma = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))

        gamma = nn.Parameter(gamma.expand_as(self.bn.weight))

        self.bn.weight = gamma

        x = self.bn(x)
        return x

class VAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, label_size, tau=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size + label_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], latent_size * 2)

        self.bn_mu = BN_Layer(latent_size, tau, mu=True)
        self.bn_log_var = BN_Layer(latent_size, tau, mu=False)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        mu, log_var = x.split(x.size(1) // 2, dim=1)

        mu = self.bn_mu(mu)
        log_var = self.bn_log_var(log_var)

        return mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size, label_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size + label_size, hidden_sizes[3])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc2 = nn.Linear(hidden_sizes[3], hidden_sizes[2])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc4 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc5 = nn.Linear(hidden_sizes[0], output_size)

    def forward(self, x, label):
        x = torch.cat((x, label), 1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, label_size, tau=0.5):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_size, hidden_sizes, latent_size, label_size, tau)
        self.decoder = VAEDecoder(latent_size, hidden_sizes, input_size, label_size)

    def forward(self, x, label):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decoder(z, label)
        return recon, mu, log_var