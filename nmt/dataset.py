import torch
from torch.utils.data import Dataset, DataLoader


def get_texts(path):
    with open(path) as file:
        return list(map(str.strip, file.readlines()))


# TODO: (@whiteRa2bit, 2020-08-30) Fix for only src_data
class TextDataset(Dataset):
    def __init__(self, src_data_path, trg_data_path, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        src_texts = get_texts(src_data_path)
        trg_texts = get_texts(trg_data_path)
        assert (len(src_texts) == len(trg_texts))

        self.src_idxs = self.src_vocab.vectorize_texts(src_texts)
        self.trg_idxs = self.trg_vocab.vectorize_texts(trg_texts)

    def __len__(self):
        return len(self.src_idxs)

    def __getitem__(self, idx):
        src_tensor = torch.from_numpy(self.src_idxs[idx])
        trg_tensor = torch.from_numpy(self.trg_idxs[idx])
        return src_tensor, trg_tensor
