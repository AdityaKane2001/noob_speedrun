from functools import partial
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from patchify import get_patches
from model import ViT

def flatten(patch):
    return torch.flatten(patch, start_dim=1)

def get_transform(*, train=True, dim=16):
    get_patches_transform = partial(get_patches, dim=dim)
    if train:
        transforms = T.Compose(
            [
                # T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Lambda(get_patches_transform), # (batch_size, num_patches, patch_side, patch_side, channels)
                T.Lambda(flatten), # (batch_size, num_patches, patch_side * patch_side * channels) -> nn.Linear() -> (batch_size, num_patches, hidden_dims)
            ]
        )
        return transforms
    else:
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Lambda(get_patches_transform),
                T.Lambda(flatten),
            ]
        )
        return transforms


train_ds = CIFAR10(
    root="./cifar10", train=True, download=True, transform=get_transform()
)
test_ds = CIFAR10(
    root="./cifar10", train=False, download=True, transform=get_transform(train=False)
)

def get_data(tr_bs=1024, te_bs=1024):
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=tr_bs, drop_last=True)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=te_bs, drop_last=True)
    return train_dl, test_dl