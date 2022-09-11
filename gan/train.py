import torch.nn as nn
import torch
from model import Generator, Discriminator

## ZERO: real
## ONE: fake


def cycle_loss(gen_G, gen_F, X_images, Y_images):
    return nn.L1Loss()(gen_F(gen_G(X_images)), X_images) + nn.L1Loss()(
        gen_G(gen_F(Y_images)), Y_images
    )


def disc_bce_loss(preds, gts):
    return nn.BCEWithLogitsLoss()(preds, gts)


def adversarial_loss(gen, disc, dom1_images, dom2_images):
    """
    in terms of OG GAN paper
    y == dom2_images
    z == dom1_images
    G(z) == gen(dom1_images)
    """
    gen_loss = disc_bce_loss(
        disc(gen(dom1_images)), torch.zeros(dom1_images.shape[0])
    )  # 1 - D(G(z)) == flipping the labels, G(z) is always fake, hence label is always real (ZERO)
    disc_loss = disc_bce_loss(
        disc(dom2_images), torch.zeros(dom2_images.shape[0])
    )  # D(y), y is always real, hence label is always real (ZERO)
    return gen_loss + disc_loss


def final_loss(gen_G, gen_F, disc_X, disc_Y, X_images, Y_images):
    ganG_loss = adversarial_loss(
        gen=gen_G, disc=disc_Y, dom1_images=X_images, dom2_images=Y_images
    )
    ganF_loss = adversarial_loss(
        gen=gen_F, disc=disc_X, dom1_images=Y_images, dom2_images=X_images
    )
    cyc_loss = cycle_loss(gen_G, gen_F, X_images, Y_images)
    return ganG_loss + ganF_loss + (cyc_loss * 10)


# Generators and Discriminators
gen_G = Generator()
disc_Y = Discriminator(in_channels=3)

gen_F = Generator()
disc_X = Discriminator(in_channels=3)

data_batch_X = torch.randn(8, 3, 127, 127)
data_batch_Y = torch.randn(8, 3, 127, 127)

optimizer = optim.Adam()

for epoch in range(50):
    combined_loss = final_loss(
        gen_G, gen_F, disc_X, disc_Y, data_batch_X, data_batch_Y
    )

# forward X --> Y.
if __name__ == "__main__":
    combined_loss = final_loss(
        gen_G, gen_F, disc_X, disc_Y, data_batch_X, data_batch_Y
    )
    print(combined_loss)
