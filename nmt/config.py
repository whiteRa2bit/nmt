import os

# Data params
DATA_DIR = "/home/pafakanov/data/other/dl/"
RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "raw_data")
TRAIN_EN_PATH = os.path.join(RAW_DATA_DIR, "train.de-en.en")
TRAIN_DE_PATH = os.path.join(RAW_DATA_DIR, "train.de-en.de")
VAL_EN_PATH = os.path.join(RAW_DATA_DIR, "val.de-en.en")
VAL_DE_PATH = os.path.join(RAW_DATA_DIR, "val.de-en.de")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# WANDB params
WANDB_PROJECT = "dl_hse_nmt"

# Training params
TRAIN_SIZE = 0.95
TRAIN_CONFIG = {
    "lr": 3e-4,
    "lr_update_each": 50,
    "epochs_num": 8,
    "log_each": 25,
    "device": "cuda",
    "train_batch_size": 64,
    "val_batch_size": 128,
}

# Inference params
INFERENCE_BATCH_SIZE = 128
