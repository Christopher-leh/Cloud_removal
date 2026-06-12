import torch
from torch import nn

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    logvar = torch.clamp(logvar, min=-10, max=10)
    recon_loss = nn.L1Loss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss