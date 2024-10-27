class Config:
    def __init__(self):
        self.epochs = 100  # Number of epochs
        self.batch_size = 64  # Batch size for better training
        self.image_size = (128, 128)  # Resized image for faster training
        self.latent_dim = 256  # Higher latent dimensions for GAN
        self.checkpoint_dir = './saved_models/'
        self.dataset_dir = './data/'
        self.learning_rate = 0.0002
        self.beta1 = 0.5  # For Adam optimizer