import torch
import torch.nn as nn

def bce_loss(recon_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    total_loss = BCE + beta * KLD
    return BCE,KLD,total_loss

def mse_loss(recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    total_loss = MSE + beta * KLD
    return MSE,KLD,total_loss