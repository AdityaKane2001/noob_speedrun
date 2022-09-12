from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from patchify import get_patches


def flatten(patch):
    return torch.flatten(patch, start_dim=1)

def get_transform(*, train=True, dim=16):
    get_patches_transform = partial(get_patches, dim=dim)
    if train:
        transforms = T.Compose(
            [
                T.RandomResizedCrop((224, 224)),
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

train_dl = DataLoader(train_ds, shuffle=True, batch_size=32, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=32, drop_last=True)

for i in train_dl:
    print(i[0].shape)
    break
