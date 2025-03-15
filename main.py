import yaml
import logging
from train import train_model
from model import NNUE
import torch


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_logging(verbose):
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(asctime)s %(levelname)s %(message)s"
    )


if __name__ == "__main__":
    config = load_config()
    setup_logging(config["logging"]["verbose"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Run on device %s", device)
    model = NNUE().to(device)

    train_model(model, config, device)
    model.save(config["training"]["save_model"])
