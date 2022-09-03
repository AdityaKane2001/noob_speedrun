# import numpy as np
import torch
import torch.nn.functional as F

CIFAR_RES = 32


def get_patches(image, dim=16):
    # CIFAR-10 has 32x32
    # dim = 16 would correspond to 16x16 patches == 4 patches
    assert CIFAR_RES % dim == 0, f"`dim` should exactly divide input dimension. Got {dim}"
    per_side_patches = int(CIFAR_RES / dim)
    patches = []
    for row in range(per_side_patches):
        row_patch = [
            image[:, row * dim : (row + 1) * dim, col * dim : (col + 1) * dim]
            for col in range(per_side_patches)
        ]
        patches.append(torch.stack(row_patch, dim=0))
    final_patches = torch.cat(patches, dim=0)
    # print(f"Final Patches Shape is: {final_patches.shape}")
    return final_patches
    # (4, 3, 16, 16)


if __name__ == "__main__":
    image = torch.randn(3, 32, 32)
    get_patches(image, dim=16)
    get_patches(image, dim=8)
    get_patches(image, dim=4)
    get_patches(image, dim=2)
