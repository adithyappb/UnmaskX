import torch
from models.gan import Generator, Discriminator
from config.config import Config

def test_generator():
    config = Config()
    gen = Generator(config.latent_dim)
    z = torch.randn(1, config.latent_dim)
    generated_img = gen(z)
    assert generated_img.shape == (1, 3, 128, 128), "Generator output shape mismatch!"
    print("Generator test passed.")

def test_discriminator():
    config = Config()
    disc = Discriminator()
    img = torch.randn(1, 3, 128, 128)
    validity = disc(img)
    assert validity.shape == (1, 1), "Discriminator output shape mismatch!"
    print("Discriminator test passed.")

if __name__ == "__main__":
    test_generator()
    test_discriminator()
