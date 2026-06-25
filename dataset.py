import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class Rice2(Dataset):
    def __init__(self, data_dir):
        self.label_dir = data_dir + "/label"
        self.cloud_dir = data_dir + "/cloud"
        self.mask_dir = data_dir + "/mask"
        self.ids = sorted(os.listdir(self.cloud_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        cloudy = Image.open(os.path.join(self.cloud_dir, name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        cloudy = F.resize(cloudy, (256, 256))
        label = F.resize(label, (256, 256))
        mask = F.resize(mask, (256, 256))

        # *2-1 to scale from [0, 1] to [-1, 1]
        cloudy = F.to_tensor(cloudy) * 2 - 1
        label = F.to_tensor(label) * 2 - 1
        mask = (F.to_tensor(mask) > 0.5).float()

        return cloudy, label, mask


import torchvision.transforms.v2.functional as F
import random

class Rice2(Dataset):
    def __init__(self, data_dir, augment=False):
        self.label_dir = data_dir + "/label"
        self.cloud_dir = data_dir + "/cloud"
        self.mask_dir  = data_dir + "/mask"
        self.ids = sorted(os.listdir(self.cloud_dir))
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        cloudy = Image.open(os.path.join(self.cloud_dir, name)).convert("RGB")
        label  = Image.open(os.path.join(self.label_dir, name)).convert("RGB")
        mask   = Image.open(os.path.join(self.mask_dir,  name)).convert("L")

        cloudy = F.resize(cloudy, (256, 256))
        label  = F.resize(label,  (256, 256))
        mask   = F.resize(mask,   (256, 256))

        # --- Synchrone Augmentation ---
        if self.augment:
            if random.random() > 0.5:
                cloudy, label, mask = F.hflip(cloudy), F.hflip(label), F.hflip(mask)
            if random.random() > 0.5:
                cloudy, label, mask = F.vflip(cloudy), F.vflip(label), F.vflip(mask)
            k = random.choice([0, 90, 180, 270])
            if k:
                cloudy, label, mask = F.rotate(cloudy, k), F.rotate(label, k), F.rotate(mask, k)

        # *2-1 to scale from [0, 1] to [-1, 1]
        cloudy = F.to_tensor(cloudy) * 2 - 1
        label  = F.to_tensor(label)  * 2 - 1
        mask   = (F.to_tensor(mask) > 0.5).float()

        return cloudy, label, mask