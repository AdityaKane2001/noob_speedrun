from turtle import forward
import torch
from torch import nn
from torchsummary import summary


class c7s1(nn.Module):
    def __init__(self, k, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, k, (7, 7), padding="same", padding_mode="reflect"
        )
        self.norm = nn.InstanceNorm2d(k)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class d(nn.Module):
    def __init__(self, k, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, k, (3, 3), stride=2, padding="valid", padding_mode="reflect"
        )  # TODO: see how reflection padding is used
        self.norm = nn.InstanceNorm2d(k)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class R(nn.Module):
    def __init__(self, k, in_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, k, (3, 3), padding="same", padding_mode="reflect"
        )
        self.conv2 = nn.Conv2d(
            in_channels, k, (3, 3), padding="same", padding_mode="reflect"
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return res + x


class u(nn.Module):
    def __init__(self, k, in_channels) -> None:
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2)
        # self.conv = nn.Conv2d(in_channels, k, kernel_size=(3, 3), padding="same")
        self.conv = nn.ConvTranspose2d(
            in_channels, k, kernel_size=(3, 3), stride=(2, 2)
        )
        self.norm = nn.InstanceNorm2d(k)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            c7s1(64, 3),
            d(128, 64),
            d(256, 128),
            #
            R(256, 256),
            R(256, 256),
            R(256, 256),
            R(256, 256),
            R(256, 256),
            R(256, 256),
            # ,
            u(128, 256),
            u(64, 128),
            c7s1(3, 64),
        )

    def forward(self, imgs):
        return self.stack(imgs)


class ck(nn.Module):
    """Used in Discriminator of CycleGAN"""

    def __init__(self, k, in_channels, instance_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=k, kernel_size=(4, 4), stride=2
        )
        self.instance_norm = nn.InstanceNorm2d(num_features=k)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inputs):
        out_conv = self.conv(inputs)
        if self.instance_norm:
            out_instance = self.instance_norm(out_conv)
            return self.relu(out_instance)
        return self.relu(out_conv)


class Discriminator(nn.Module):
    """Uses class ck"""

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.c64 = ck(k=64, in_channels=in_channels, instance_norm=False)
        self.c128 = ck(k=128, in_channels=64)
        self.c256 = ck(k=256, in_channels=128)
        self.c512 = ck(k=512, in_channels=256)
        self.conv = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=6
        )  # remove this hardcode bit

    def forward(self, inputs):
        out_c64 = self.c64(inputs)
        out_c128 = self.c128(out_c64)
        out_c256 = self.c256(out_c128)
        out_c512 = self.c512(out_c256)
        self.image_size = out_c512.shape[-1]
        return self.conv(out_c512).squeeze()


if __name__ == "__main__":
    import torch

    # inputs = torch.randn((8, 3, 469, 469))
    # inputs = torch.randn((8, 3, 500, 500))
    inputs = torch.randn((8, 3, 127, 127))
    gen = Generator()
    summary(gen, (3, 127, 127))
    print(gen(inputs).shape)

    inputs = torch.randn((8, 3, 127, 127))  # bs = 8, image: 200, 200, 3
    disc = Discriminator(in_channels=3)
    print(disc(inputs).shape)
