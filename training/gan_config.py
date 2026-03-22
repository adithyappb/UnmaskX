"""Hyperparameters for legacy DCGAN training (research only; not the deployment restorer)."""


class GanTrainConfig:
    def __init__(self) -> None:
        self.epochs = 100
        self.batch_size = 64
        self.image_size = (128, 128)
        self.latent_dim = 256  # DCGAN noise dim
        self.checkpoint_dir = "./saved_models/"
        self.dataset_dir = "./data/"
        self.learning_rate = 0.0002
        self.beta1 = 0.5
