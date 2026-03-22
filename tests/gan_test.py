import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.dcgan import Discriminator, Generator
from training.gan_config import GanTrainConfig


def test_generator() -> None:
    config = GanTrainConfig()
    gen = Generator(config.latent_dim)
    gen.eval()
    z = torch.randn(1, config.latent_dim)
    generated_img = gen(z)
    assert generated_img.shape == (1, 3, 128, 128), "Generator output shape mismatch!"
    print("Generator test passed.")


def test_discriminator() -> None:
    disc = Discriminator()
    disc.eval()
    img = torch.randn(1, 3, 128, 128)
    validity = disc(img)
    assert validity.shape == (1, 1), "Discriminator output shape mismatch!"
    print("Discriminator test passed.")


if __name__ == "__main__":
    test_generator()
    test_discriminator()
