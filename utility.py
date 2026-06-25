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


def plot_losses(panels, model_name="Model", save_path=None, ncols=None):
    """
    Flexibel Train/Val-Losses plotten – beliebig viele Panels, je variabler Länge.

    Args:
        panels: dict, das Panel-Titel auf (train_list, val_list) abbildet, z.B.
                {
                    "Generator Loss":     (train_losses_g, val_losses_g),
                    "Discriminator Loss": (train_losses_d, val_losses_d),
                    "PSNR":               (None,           val_psnr),   # nur Val
                }
                Eine Liste darf None sein, wenn es zu der Kurve keine Daten gibt.
                Train- und Val-Liste dürfen unterschiedlich lang sein.
        model_name: Titel über der ganzen Figure.
        save_path:  wenn gesetzt, wird die Figure dorthin gespeichert.
        ncols:      Spaltenzahl im Subplot-Raster (default: alles in eine Reihe).
    """
    titles = list(panels.keys())
    n = len(titles)

    # Raster bestimmen
    if ncols is None:
        ncols = n
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # axes immer als flache Liste handhabbar machen
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

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ungenutzte Achsen (falls Raster größer als Panel-Zahl) ausblenden
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Losses over Epochs — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def denorm(tensor):
    return tensor * 0.5 + 0.5


def plot_samples(model, val_dataloader, device, epoch, n=4):
    model.eval()
    cloudy, label, mask = next(iter(val_dataloader))
    cloudy, label = cloudy.to(device), label.to(device)

    with torch.no_grad():
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


