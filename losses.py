import torch
from torch import nn


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    logvar = torch.clamp(logvar, min=-10, max=10)
    mu = torch.clamp(mu, min=-100, max=100)
    recon_loss = nn.L1Loss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, beta * kl_loss




def weighted_l1_loss(recon_x, x, mask, extra=4.0):
    """L1 über das ganze Bild, Wolkenregionen (mask=1) zusätzlich gewichtet.
    Kein Wolkengrößen-Artefakt, weil über alle Pixel gemittelt wird (alle Gewichte >=1)."""
    l1_map = torch.abs(recon_x - x)
    weight = 1.0 + extra * mask
    return (l1_map * weight).mean()