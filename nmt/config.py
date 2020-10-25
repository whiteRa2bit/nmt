import os

# Data params
DATA_DIR = "/home/pafakanov/data/other/dl/"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
TRAIN_SRC_PATH = os.path.join(RAW_DATA_DIR, "train.de-en.en")
TRAIN_TRG_PATH = os.path.join(RAW_DATA_DIR, "train.de-en.de")
VAL_SRC_PATH = os.path.join(RAW_DATA_DIR, "val.de-en.en")
VAL_TRG_PATH = os.path.join(RAW_DATA_DIR, "val.de-en.de")
TEST_TRG_PATH = os.path.join(RAW_DATA_DIR, "test1.de-en.de")
TEST_SRC_PATH = os.path.join(RAW_DATA_DIR, "test1.de-en.en")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# WANDB params
WANDB_PROJECT = "dl_hse_nmt"
SAMPLES_NUM = 10

# Training params
TRAIN_SIZE = 0.95
CONFIG = {
    "data": {
        "src_vocab_size": 5000,
        "src_max_len": 45,
        "trg_vocab_size": 5000,
        "trg_max_len": 42
    },
    "train": {
        "lr": 1e-3,
        "epochs_num": 100,
        "enc_emb_size": 16,
        "enc_hid_size": 256,
        "dec_emb_size": 15,
        "dec_hid_size": 256,
        "log_each": 100,
        "device": "cuda:0",
        "train_batch_size": 64,
        "val_batch_size": 128,
    }
}

# Inference params
INFERSRCCE_BATCH_SIZE = 128
