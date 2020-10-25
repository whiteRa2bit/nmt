import torch

from nmt.vocab import get_src_vocab, get_trg_vocab
from nmt.dataset import TextDataset
from nmt.nets.baseline_net import Encoder, Decoder
from nmt.trainer import Trainer
from nmt.config import TRAIN_SRC_PATH, TRAIN_TRG_PATH, VAL_SRC_PATH, VAL_TRG_PATH, CONFIG


def main():
    data_config = CONFIG["data"]
    train_config = CONFIG["train"]

    src_vocab = get_src_vocab(TRAIN_SRC_PATH, data_config)
    trg_vocab = get_trg_vocab(TRAIN_TRG_PATH, data_config)
    train_dataset = TextDataset(TRAIN_SRC_PATH, TRAIN_TRG_PATH, src_vocab, trg_vocab)
    val_dataset = TextDataset(VAL_SRC_PATH, VAL_TRG_PATH, src_vocab, trg_vocab)

    encoder = Encoder(src_vocab, train_config)
    decoder = Decoder(trg_vocab, train_config)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=train_config["lr"])

    trainer = Trainer(encoder, decoder, optimizer, train_dataset, val_dataset, train_config)
    trainer.train()


if __name__ == '__main__':
    main()
