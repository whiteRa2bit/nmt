import torch
from torch.utils.data import Dataset, DataLoader


def get_texts(path):
    try:
        with open(path) as file:
            return list(map(str.strip, file.readlines()))
    except FileNotFoundError:
        return []


# TODO: (@whiteRa2bit, 2020-08-30) Fix for only src_data
class TextDataset(Dataset):
    def __init__(self, src_data_path, trg_data_path, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_texts = get_texts(src_data_path)
        self.trg_texts = get_texts(trg_data_path)
        assert ((not self.trg_texts) or (len(self.src_texts) == len(self.trg_texts)))

        self.src_idxs = self.src_vocab.vectorize_texts(self.src_texts)
        self.trg_idxs = self.trg_vocab.vectorize_texts(self.trg_texts)

    def __len__(self):
        return len(self.src_idxs)

    def __getitem__(self, idx):
        src_tensor = torch.from_numpy(self.src_idxs[idx])
        trg_tensor = torch.from_numpy(self.trg_idxs[idx])
        return src_tensor, trg_tensor
