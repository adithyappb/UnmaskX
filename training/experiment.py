from training.gan_train import train_gan
from config.config import Config

def run_experiment(exp_name, config_overrides=None):
    config = Config()

    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    print(f"Running experiment: {exp_name}")
    train_gan()

def main():
    # Example of running experiments with different configurations
    experiment_configs = [
        {'epochs': 50, 'batch_size': 32},
        {'epochs': 100, 'batch_size': 64, 'latent_dim': 512}
    ]

    for i, cfg in enumerate(experiment_configs):
        exp_name = f"Experiment_{i}"
        run_experiment(exp_name, cfg)

if __name__ == "__main__":
    main()
