from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import Rice2
import torch

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

def plot_losses(train_losses, val_losses,
                train_recon, val_recon,
                train_kl, val_kl,
                model_name="VAE", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Validation")
    axes[0].set_title("Total Loss")

    axes[1].plot(train_recon, label="Train")
    axes[1].plot(val_recon,   label="Validation")
    axes[1].set_title("Reconstruction Loss")

    axes[2].plot(train_kl, label="Train")
    axes[2].plot(val_kl,   label="Validation")
    axes[2].set_title("KL Divergence")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Losses over Epochs — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
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
