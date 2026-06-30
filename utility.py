from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import Rice2
import torch
import math


def plot_images(dataloader, n_img):
    cloudy, label, mask = next(iter(dataloader))
    for i in range(n_img):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cloudy[i].permute(1, 2, 0) * 0.5 + 0.5)
        axs[0].set_title("Cloudy Image")
        axs[0].axis("off")

        axs[1].imshow(label[i].permute(1, 2, 0) * 0.5 + 0.5)
        axs[1].set_title("Label Image")
        axs[1].axis("off")

        axs[2].imshow(mask[i][0], cmap="gray")
        axs[2].set_title("Mask Image")
        axs[2].axis("off")

        plt.show()


import numpy as np


def plot_losses(
    panels, model_name="Model", save_path=None, ncols=None, clip_percentile=99
):
    """
    Flexibel Train/Val-Losses plotten – beliebig viele Panels, je variabler Länge.

    Args:
        panels: dict, das Panel-Titel auf (train_list, val_list) abbildet.
                Eine Liste darf None sein. Train/Val dürfen unterschiedlich lang sein.
        model_name: Titel über der ganzen Figure.
        save_path:  wenn gesetzt, wird die Figure dorthin gespeichert.
        ncols:      Spaltenzahl im Subplot-Raster (default: alles in eine Reihe).
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

        # y-Achse robust begrenzen, damit Spikes die Skala nicht stauchen
        if clip_percentile is not None:
            vals = []
            for lst in (train_list, val_list):
                if lst is not None and len(lst) > 0:
                    vals.extend(lst)
            if vals:
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]  # NaN/Inf rausfiltern
                if vals.size > 0:
                    lo = vals.min()
                    hi = np.percentile(vals, clip_percentile)
                    if hi > lo:  # nur setzen, wenn sinnvoller Bereich
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


def denorm(tensor):
    return tensor * 0.5 + 0.5


def plot_samples(model, val_dataloader, device, epoch, n=4, use_masks=False):
    model.eval()
    cloudy, label, mask = next(iter(val_dataloader))
    cloudy, label, mask = cloudy.to(device), label.to(device), mask.to(device)

    with torch.no_grad():
        if use_masks:
            recon = model.generate(
                torch.cat((cloudy, mask), dim=1)
            )  # Testzeit: z aus N(0,1)
        else:
            recon = model.generate(cloudy)  # Testzeit: z aus N(0,1)

    cloudy = denorm(cloudy.cpu())
    label = denorm(label.cpu())
    recon = denorm(recon.cpu())

    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    for i in range(n):
        axes[0, i].imshow(cloudy[i].permute(1, 2, 0))
        axes[0, i].set_title("Cloudy")
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[1, i].set_title("Reconstruction")
        axes[2, i].imshow(label[i].permute(1, 2, 0))
        axes[2, i].set_title("Ground Truth")
        for row in range(3):
            axes[row, i].axis("off")
    fig.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()


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


def to_img(t):
    """[-1,1] Tensor (C,H,W) -> [0,1] numpy (H,W,C) für skimage."""
    t = (t.clamp(-1, 1) + 1) / 2
    return t.cpu().numpy().transpose(1, 2, 0)


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
