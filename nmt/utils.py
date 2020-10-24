import os

from nmt.config import CHECKPOINT_DIR


def get_checkpoint_path(model, config):
    checkpoint_name = model.name
    for key, value in config.items():
        checkpoint_name += f"_{key}_{value}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    return checkpoint_path
