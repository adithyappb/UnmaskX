import torch
from torch.optim import Adam
from models.gan import Generator, Discriminator
from data.data_loader import load_data
from config.config import Config
import os

def train_gan():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Generator and Discriminator
    generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    dataloader = load_data(config.batch_size, config.dataset_dir)

    for epoch in range(config.epochs):
        for i, imgs in enumerate(dataloader):
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), config.latent_dim, device=device)
            generated_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs.to(device)), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(f"Epoch {epoch}/{config.epochs} [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            if epoch % 10 == 0:
                save_checkpoint(generator, discriminator, epoch)

def save_checkpoint(generator, discriminator, epoch):
    os.makedirs(Config().checkpoint_dir, exist_ok=True)
    torch.save(generator.state_dict(), f"{Config().checkpoint_dir}/generator_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{Config().checkpoint_dir}/discriminator_{epoch}.pth")
