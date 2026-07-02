import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim_fn


# define the VAE loss function
def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # clamp logvar and mu to avoid numerical issues
    logvar = torch.clamp(logvar, min=-10, max=10)
    mu = torch.clamp(mu, min=-100, max=100)
    # compute the reconstruction loss and KL divergence
    recon_loss = nn.L1Loss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, beta * kl_loss


# define the weighted L1 loss function
def weighted_l1_loss(recon_x, x, mask, extra=4.0):
    """L1 loss with additional weight for masked regions."""
    l1_map = torch.abs(recon_x - x)
    weight = 1.0 + extra * mask
    return (l1_map * weight).mean()


# define  SSIM function to have the same parameters as in a paper i saw to compare results
def compute_ssim(img1_np, img2_np):
    """Gauß 11x11, sigma=1.5"""
    return ssim_fn(
        img1_np,
        img2_np,
        data_range=1.0,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
