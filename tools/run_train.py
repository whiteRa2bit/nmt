import torch

from nmt.vocab import get_src_vocab, get_trg_vocab
from nmt.dataset import TextDataset
from nmt.nets.baseline_net import BasicNet
from nmt.trainer import Trainer
from nmt.config import TRAIN_SRC_PATH, TRAIN_TRG_PATH, VAL_SRC_PATH, VAL_TRG_PATH, CONFIG
from scheduler import get_gpu_id


def main(config=CONFIG):
    gpu_id = get_gpu_id()
    data_config = config["data"]
    train_config = config["train"]
    train_config["device"] = f"cuda:{gpu_id}"

    src_vocab = get_src_vocab(TRAIN_SRC_PATH, data_config)
    trg_vocab = get_trg_vocab(TRAIN_TRG_PATH, data_config)
    train_dataset = TextDataset(TRAIN_SRC_PATH, TRAIN_TRG_PATH, src_vocab, trg_vocab)
    val_dataset = TextDataset(VAL_SRC_PATH, VAL_TRG_PATH, src_vocab, trg_vocab)

    model = BasicNet(src_vocab, train_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    trainer = Trainer(model, optimizer, train_dataset, val_dataset, config)
    trainer.train()


if __name__ == '__main__':
    main()
