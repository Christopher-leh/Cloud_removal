from pyexpat import features

from torch import nn
import torch

# VAE-Architektur mit Encoder, Decoder und Reparameterisierungstrick -------------------------------------


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(
            6, 32, kernel_size=4, stride=2, padding=1
        )  # 256x256 -> 128x128
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 128x128 -> 64x64
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 64x64 -> 32x32
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 16x16
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.linear_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, cloudy, label):
        x = torch.cat((cloudy, label), dim=1)  # Concatenate along the channel dimension
        x = self.leaky_relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.batch_norm4(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar


def reparameterize(mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256 * 16 * 16)
        self.conv1 = nn.ConvTranspose2d(
            256 + 3, 128, kernel_size=4, stride=2, padding=1
        )
        """
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        """

        self.batch_norm1 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, z, label):
        x = self.leaky_relu(self.linear(z))
        x = x.view(-1, 256, 16, 16)
        label = nn.functional.interpolate(
            label, size=(16, 16), mode="bilinear", align_corners=False
        )
        x = torch.cat((x, label), dim=1)  # Concatenate along the channel dimension
        x = self.leaky_relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.upsample1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.upsample2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.upsample3(x)
        x = self.tanh(self.conv4(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, cloudy, label):
        mu, logvar = self.encoder(cloudy, label)
        z = reparameterize(mu, logvar)
        output = self.decoder(z, cloudy)
        return output, mu, logvar

    def generate(self, cloudy):
        z = torch.randn(cloudy.size(0), self.latent_dim).to(cloudy.device)
        output = self.decoder(z, cloudy)
        return output


# --------------------------------------------------------------------------

# GAN-Architektur mit Generator und Diskriminator -------------------------------------

c
