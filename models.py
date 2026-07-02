from torch import nn
import torch

# CVAE Model with Encoder and Decoder


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, cond_channels=3):

        super().__init__()

        in_ch = (
            cond_channels + 3
        )  # 3 channels for the ground_truth (ground truth) and cond_channels for the condition (cloudy image or cloudy+mask)

        # 4 convolutional layers with stride 2 to downsample the input image from 256x256 to 16x16
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

        # linear layers to map the flattened feature map to the latent space (mu and logvar)
        self.linear_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.linear_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, cond, ground_truth):
        # cond = cloudy (oder cloudy+mask)
        x = torch.cat((cond, ground_truth), dim=1)
        x = self.batch_norm1(self.leaky_relu(self.conv1(x)))
        x = self.batch_norm2(self.leaky_relu(self.conv2(x)))
        x = self.batch_norm3(self.leaky_relu(self.conv3(x)))
        x = self.batch_norm4(self.leaky_relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar


# function for reparameterization trick to sample from the latent space
def reparameterize(mu, logvar):
    # clamping logvar and mu to avoid numerical issues
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, cond_channels=3):
        super().__init__()
        self.cond_channels = cond_channels

        # linear layer to map the latent space to a feature map of size 16x16 with 256 channels
        self.linear = nn.Linear(latent_dim, 256 * 16 * 16)

        # cond channels are concatenated to the feature map, so the input channels for the first conv layer is 256 + cond_channels
        self.conv1 = nn.ConvTranspose2d(
            256 + cond_channels, 128, kernel_size=4, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(128)

        # using bilinear upsampling instead of ConvTranspose2d for the next layers to avoid checkerboard artifacts
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

        # Cond is resized to 16x16 and concatenated to the feature map
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


# bring everything together in a VAE model with encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, cond_channels=3):

        super().__init__()
        self.encoder = Encoder(latent_dim, cond_channels)
        self.decoder = Decoder(latent_dim, cond_channels)
        self.latent_dim = latent_dim

    def forward(self, cond, ground_truth):
        # cond = cloudy[+mask]
        mu, logvar = self.encoder(cond, ground_truth)

        z = reparameterize(mu, logvar)

        output = self.decoder(z, cond)
        return output, mu, logvar

    # generate function for inference, where we sample z from N(0,1) and decode it with the condition
    def generate(self, cond):
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.decoder(z, cond)

    # function to reconstruct the input image from the latent space, using the encoder to get mu and logvar, and then decoding mu (without noise) with the condition
    def reconstruct(self, cond, ground_truth):
        mu, logvar = self.encoder(cond, ground_truth)
        return self.decoder(mu, cond)


# --------------------------------------------------------------------------

# GAN-Architektur mit Generator und Diskriminator -------------------------------------


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=64, depth=4):
        super().__init__()
        self.depth = depth

        # DoubleConv class for the UNet architecture
        # DoubleConv consists of two convolutional layers with batch normalization and ReLU activation
        class DoubleConv(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),  # tried inplace=True to save memory
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        # create the encoder and decoder parts of the UNet architecture using ModuleList
        self.encoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # pooling layer and tanh activation for the output
        self.pool = nn.MaxPool2d(2)
        self.tanh = nn.Tanh()

        # create the encoder part of the UNet architecture, where each encoder consists of a DoubleConv followed by a pooling layer
        # the number of channels doubles after each encoder, starting from feature_channels
        current_in = in_channels
        current_out = feature_channels
        for i in range(depth):
            self.encoders.append(DoubleConv(current_in, current_out))
            current_in = current_out
            current_out *= 2

        # bottleneck again is such a block
        self.bottleneck = DoubleConv(current_in, current_out)

        # create the decoder part of the UNet architecture, where each decoder consists of an upsampling layer followed by a DoubleConv
        # the number of channels halves after each decoder, starting from current_out (which is now the output of the bottleneck)
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
        # create a list to store the skip connections from the encoder part of the UNet architecture
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = up(x)
            x = torch.cat(
                [x, skip], dim=1
            )  # concatenate the skip connection from the encoder with the upsampled feature map from the decoder
            x = dec(x)
        return self.tanh(self.out_conv(x))


# first version of the discriminator, which takes the whole image as input and outputs a single value (real or fake)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()

        # 5 convolutional layers with stride 2 to downsample the input image from 256x256 to 8x8, followed by a linear layer to output a single value
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

        # adaptive pooling layer to downsample the feature map to 8x8
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


# second version of the discriminator, which takes patches of the image as input and outputs a value for each patch (real or fake)
class Discriminator_patches(nn.Module):
    def __init__(self, in_channels=6, feature_channels=64):
        super().__init__()

        # define a block function to create a convolutional block with Conv2d, BatchNorm2d and LeakyReLU
        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )

        # add the convolutional blocks to the model, with increasing number of feature channels and decreasing spatial dimensions
        # the input image is downsampled from 256x256 to 16x16, and the output is a single channel feature map of size 16x16, where each pixel corresponds to a patch of the input image
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
