from training.gan_config import GanTrainConfig
from training.gan_train import train_gan


def run_experiment(exp_name: str, config_overrides: dict | None = None) -> None:
    config = GanTrainConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    print(f"Running experiment: {exp_name}")
    train_gan(config)


def main() -> None:
    experiment_configs = [
        {"epochs": 50, "batch_size": 32},
        {"epochs": 100, "batch_size": 64, "latent_dim": 512},
    ]

    for i, cfg in enumerate(experiment_configs):
        exp_name = f"Experiment_{i}"
        run_experiment(exp_name, cfg)


if __name__ == "__main__":
    main()
