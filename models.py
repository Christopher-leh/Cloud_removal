from torch import nn
import torch

# VAE-Architektur mit Encoder, Decoder und Reparameterisierungstrick -------------------------------------


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, cond_channels=3):
        """
        cond_channels: Kanäle der Bedingung (3 = cloudy RGB, 4 = cloudy+mask).
        Encoder-Input = cond_channels (cloudy[+mask]) + 3 (label).
        """
        super().__init__()
        in_ch = cond_channels + 3
        self.conv1 = nn.Conv2d(
            in_ch, 32, kernel_size=4, stride=2, padding=1
        )  # 256 -> 128
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.linear_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, cond, label):
        # cond = cloudy (oder cloudy+mask), label = ground truth
        x = torch.cat((cond, label), dim=1)
        x = self.batch_norm1(self.leaky_relu(self.conv1(x)))
        x = self.batch_norm2(self.leaky_relu(self.conv2(x)))
        x = self.batch_norm3(self.leaky_relu(self.conv3(x)))
        x = self.batch_norm4(self.leaky_relu(self.conv4(x)))
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
    def __init__(self, latent_dim=128, cond_channels=3):
        super().__init__()
        self.cond_channels = cond_channels
        self.linear = nn.Linear(latent_dim, 256 * 16 * 16)
        # Bedingung wird auf 16x16 interpoliert und an den Bottleneck gehängt
        self.conv1 = nn.ConvTranspose2d(
            256 + cond_channels, 128, kernel_size=4, stride=2, padding=1
        )
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

    def forward(self, z, cond):
        x = self.leaky_relu(self.linear(z))
        x = x.view(-1, 256, 16, 16)
        cond = nn.functional.interpolate(
            cond, size=(16, 16), mode="bilinear", align_corners=False
        )
        x = torch.cat((x, cond), dim=1)
        x = self.batch_norm1(self.leaky_relu(self.conv1(x)))
        x = self.upsample1(x)
        x = self.batch_norm2(self.leaky_relu(self.conv2(x)))
        x = self.upsample2(x)
        x = self.batch_norm3(self.leaky_relu(self.conv3(x)))
        x = self.upsample3(x)
        x = self.tanh(self.conv4(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, cond_channels=3):
        """
        cond_channels=3  -> Bedingung ist cloudy RGB
        cond_channels=4  -> Bedingung ist cloudy+mask (use_masks=True)
        """
        super().__init__()
        self.encoder = Encoder(latent_dim, cond_channels)
        self.decoder = Decoder(latent_dim, cond_channels)
        self.latent_dim = latent_dim

    def forward(self, cond, label):
        # cond = cloudy[+mask], label = ground truth (nur im Training verfügbar)
        mu, logvar = self.encoder(cond, label)
        z = reparameterize(mu, logvar)
        output = self.decoder(z, cond)  # Decoder auf dieselbe Bedingung wie Encoder
        return output, mu, logvar

    def generate(self, cond):
        # Testzeit: kein label, z aus N(0,1)
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.decoder(z, cond)

    def reconstruct(self, cond, label):
        # Debug: z = mu (echter Encoder-Output mit Label), kein Rauschen
        mu, logvar = self.encoder(cond, label)
        return self.decoder(mu, cond)


# --------------------------------------------------------------------------

# GAN-Architektur mit Generator und Diskriminator -------------------------------------


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=64, depth=4):
        super().__init__()
        self.depth = depth

        class DoubleConv(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        self.encoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.pool = nn.MaxPool2d(2)
        self.tanh = nn.Tanh()

        current_in = in_channels
        current_out = feature_channels
        for i in range(depth):
            self.encoders.append(DoubleConv(current_in, current_out))
            current_in = current_out
            current_out *= 2

        self.bottleneck = DoubleConv(current_in, current_out)

        for i in range(depth):
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(
                        current_out, current_in, kernel_size=3, stride=1, padding=1
                    ),
                )
            )
            self.decoders.append(DoubleConv(current_in * 2, current_in))
            current_out = current_in
            current_in //= 2

        self.out_conv = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.tanh(self.out_conv(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, feature_channels, kernel_size=4, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(feature_channels)
        self.conv2 = nn.Conv2d(
            feature_channels, feature_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(feature_channels * 2)
        self.conv3 = nn.Conv2d(
            feature_channels * 2,
            feature_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.batch_norm3 = nn.BatchNorm2d(feature_channels * 4)
        self.conv4 = nn.Conv2d(
            feature_channels * 4,
            feature_channels * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.batch_norm4 = nn.BatchNorm2d(feature_channels * 8)
        self.conv5 = nn.Conv2d(
            feature_channels * 8,
            feature_channels * 16,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.batch_norm5 = nn.BatchNorm2d(feature_channels * 16)
        self.linear = nn.Linear(feature_channels * 16 * 8 * 8, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.batch_norm1(self.leaky_relu(self.conv1(x)))
        x = self.batch_norm2(self.leaky_relu(self.conv2(x)))
        x = self.batch_norm3(self.leaky_relu(self.conv3(x)))
        x = self.batch_norm4(self.leaky_relu(self.conv4(x)))
        x = self.batch_norm5(self.leaky_relu(self.conv5(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class Discriminator_patches(nn.Module):
    def __init__(self, in_channels=6, feature_channels=64):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            block(feature_channels, feature_channels * 2, 2),
            block(feature_channels * 2, feature_channels * 4, 2),
            block(feature_channels * 4, feature_channels * 8, 1),
            nn.Conv2d(feature_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
