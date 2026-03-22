from __future__ import annotations

import os

import torch
from torch.optim import Adam

from training.dcgan import Discriminator, Generator
from training.face_data_loader import load_face_data
from training.gan_config import GanTrainConfig


def train_gan(config: GanTrainConfig | None = None) -> None:
    config = config or GanTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    adversarial_loss = torch.nn.BCELoss()

    dataloader = load_face_data(config.batch_size, config.dataset_dir)

    for epoch in range(config.epochs):
        for i, imgs in enumerate(dataloader):
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), config.latent_dim, device=device)
            generated_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs.to(device)), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                f"Epoch {epoch}/{config.epochs} [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

        if epoch % 10 == 0:
            save_checkpoint(generator, discriminator, epoch, config)


def save_checkpoint(generator, discriminator, epoch: int, config: GanTrainConfig) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(generator.state_dict(), f"{config.checkpoint_dir}/generator_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{config.checkpoint_dir}/discriminator_{epoch}.pth")
