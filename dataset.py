import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random


# dataset class with preprocessing and augmentation
class Rice2(Dataset):
    # flag augment: if True, apply random horizontal/vertical flips and rotations to the images
    def __init__(self, data_dir, augment=False):
        self.label_dir = data_dir + "/label"
        self.cloud_dir = data_dir + "/cloud"
        self.mask_dir = data_dir + "/mask"
        self.ids = sorted(
            os.listdir(self.cloud_dir)
        )  # sort the ids to ensure that the order of images is consistent across label, cloud, and mask directories
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        cloudy = Image.open(os.path.join(self.cloud_dir, name)).convert(
            "RGB"
        )  # convert to RGB to ensure that the images have 3 channels
        label = Image.open(os.path.join(self.label_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert(
            "L"
        )  # convert to grayscale to ensure that the mask has 1 channel

        # resize to 256x256 for training, since the original images are 512x512 a
        cloudy = F.resize(cloudy, (256, 256))
        label = F.resize(label, (256, 256))
        mask = F.resize(mask, (256, 256))

        # use augmentation if augment flag is True, with 50% probability for horizontal and vertical flips, and random rotation by 0, 90, 180, or 270 degrees
        if self.augment:
            if random.random() > 0.5:
                cloudy, label, mask = F.hflip(cloudy), F.hflip(label), F.hflip(mask)
            if random.random() > 0.5:
                cloudy, label, mask = F.vflip(cloudy), F.vflip(label), F.vflip(mask)
            k = random.choice([0, 90, 180, 270])
            if k:
                cloudy, label, mask = (
                    F.rotate(cloudy, k),
                    F.rotate(label, k),
                    F.rotate(mask, k),
                )

        # convert to tensor and normalize to [-1, 1] for training
        cloudy = F.to_tensor(cloudy) * 2 - 1
        label = F.to_tensor(label) * 2 - 1
        mask = (F.to_tensor(mask) > 0.5).float()

        return cloudy, label, mask
