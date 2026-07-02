from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import Rice2
import torch
import math

import numpy as np


# variable function for losses plotting can take any number of panels, each with different length of train/val lists
def plot_losses(
    panels, model_name="Model", save_path=None, ncols=None, clip_percentile=99
):
    """
    clip_percentile: y-Achse pro Panel auf dieses Perzentil der Daten begrenzen,
                     damit einzelne Spikes die Skala nicht stauchen. None = aus.
                     Die Kurven werden voll geplottet, nur der sichtbare
                     Ausschnitt (ylim) wird beschnitten.
    """

    titles = list(panels.keys())
    n = len(titles)

    if ncols is None:
        ncols = n
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, title in zip(axes, titles):
        train_list, val_list = panels[title]

        if train_list is not None and len(train_list) > 0:
            ax.plot(range(1, len(train_list) + 1), train_list, label="Train")
        if val_list is not None and len(val_list) > 0:
            ax.plot(range(1, len(val_list) + 1), val_list, label="Validation")

        # y-axis limits based on clip_percentile
        if clip_percentile is not None:
            vals = []
            for lst in (train_list, val_list):
                if lst is not None and len(lst) > 0:
                    vals.extend(lst)
            if vals:
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]  # NaN/Inf filter
                if vals.size > 0:
                    lo = vals.min()
                    hi = np.percentile(vals, clip_percentile)
                    if hi > lo:  # avoid zero range
                        margin = 0.05 * (hi - lo)
                        ax.set_ylim(lo - margin, hi + margin)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Losses over Epochs — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# convert [-1, 1] tensor to [0, 1] for visualization
def denorm(tensor):
    return tensor * 0.5 + 0.5


# different plotting functions for CVAE and GAN, since they have different architectures (could be merged into one function with a flag)


# takes the fixed batch so that i can choose which images to plot
def plot_samples_cvae(model, fixed_batch, device, use_masks=False, n=5, save_path=None):

    model.eval()
    cloudy, label, mask = fixed_batch
    cloudy, label, mask = cloudy.to(device), label.to(device), mask.to(device)
    n = min(n, cloudy.size(0))

    cond = torch.cat([cloudy, mask], dim=1) if use_masks else cloudy

    with torch.no_grad():
        recon = model.generate(cond)  # z ~ N(0,1)

    cloudy, label, recon = (
        denorm(cloudy.cpu()),
        denorm(label.cpu()),
        denorm(recon.cpu()),
    )
    rows = [("Cloudy", cloudy), ("Reconstruction", recon), ("Ground Truth", label)]

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 3, len(rows) * 3))
    if n == 1:
        axes = axes.reshape(-1, 1)
    for r, (title, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
            axes[r, i].axis("off")
        axes[r, 0].set_title(title, loc="left", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# same for gans and unet
def plot_samples_gan(
    generator, fixed_batch, device, epoch, n=5, save_path=None, use_masks=False
):
    generator.eval()
    cloudy, label, mask = fixed_batch
    cloudy, label, mask = cloudy.to(device), label.to(device), mask.to(device)
    n = min(n, cloudy.size(0))

    with torch.no_grad():
        if use_masks:
            output = generator(torch.cat((cloudy, mask), dim=1))
        else:
            output = generator(cloudy)

    # [-1, 1] -> [0, 1] for visualization
    cloudy = denorm(cloudy.cpu())
    label = denorm(label.cpu())
    output = denorm(output.cpu())

    rows = [("Cloudy", cloudy), ("Output", output), ("Ground Truth", label)]

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 3, len(rows) * 3))
    if n == 1:
        axes = axes.reshape(-1, 1)
    for r, (title, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
            axes[r, i].axis("off")
        axes[r, 0].set_title(title, loc="left", fontsize=10)
    fig.suptitle(f"Epoch {epoch + 1}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# for calculating PSNR and SSIM, we need to convert the [-1, 1] tensors to [0, 1] numpy arrays for skimage
def to_img(t):
    """[-1,1] Tensor (C,H,W) -> [0,1] numpy (H,W,C) für skimage."""
    t = (t.clamp(-1, 1) + 1) / 2
    return t.cpu().numpy().transpose(1, 2, 0)


# plot some test samples with the generator, for visual inspection of the results
def plot_test_samples(
    generator, dataloader, device, n_show=5, save_path=None, use_masks=False
):
    generator.eval()
    cloudy, target, mask = next(iter(dataloader))
    cloudy, target, mask = cloudy.to(device), target.to(device), mask.to(device)
    n_show = min(n_show, cloudy.size(0))

    with torch.no_grad():
        if use_masks:
            output = generator(torch.cat((cloudy, mask), dim=1))
        else:
            output = generator(cloudy)

    # convert to [0, 1] for visualization
    cloudy_d = denorm(cloudy.cpu())
    target_d = denorm(target.cpu())
    output_d = denorm(output.cpu())

    rows = [("Cloudy", cloudy_d), ("Output", output_d), ("Ground Truth", target_d)]

    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 3, len(rows) * 3))
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    for r, (title, imgs) in enumerate(rows):
        for i in range(n_show):
            axes[r, i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
            axes[r, i].axis("off")
        axes[r, 0].set_title(title, loc="left", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
