import numpy as np
from collections import defaultdict

from nmt.dataset import get_texts


class Vocab:
    _EOS = "_EOS_"
    _SOS = "_SOS_"
    _UNK = "_UNK_"

    def __init__(self, texts, vocab_size, max_len):
        token_tuples = self._get_token_tuples(texts)

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.special_symbols = [self._EOS, self._SOS, self._UNK]
        self.vocab = self.special_symbols + [pair[0] for pair in token_tuples]
        self.vocab = self.vocab[:vocab_size]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    @staticmethod
    def _get_token_tuples(texts):
        token_counts = defaultdict(int)
        for text in texts:
            for word in text.split():
                token_counts[word] += 1
        token_tuples = [(key, val) for key, val in token_counts.items()]
        token_tuples.sort(key=lambda x: x[1], reverse=True)

        return token_tuples

    def vectorize(self, text):
        tokens = [self._SOS] + text.split() + [self._EOS]
        idxs = np.zeros(self.max_len, dtype=int)

        for i in range(min(self.max_len, len(tokens))):
            token = tokens[i]
            if token in self.word_to_idx:
                idxs[i] = self.word_to_idx[tokens[i]]
            else:
                idxs[i] = self.word_to_idx[self._UNK]

        return idxs

    def vectorize_texts(self, texts):
        return [self.vectorize(text) for text in texts]

    def decode(self, idxs):
        tokens = []
        for idx in idxs[1:]:
            if idx == self.word_to_idx[self._EOS]:
                break
            else:
                token = self.idx_to_word[idx]
            tokens.append(token)
        return ' '.join(tokens).strip()

    def decode_idxs(self, idxs_list):
        texts = []
        for idxs in idxs_list:
            texts.append(self.decode(idxs))

        return texts


def _get_vocab(data_path, vocab_size, max_len):
    texts = get_texts(data_path)
    vocab = Vocab(texts, vocab_size, max_len)
    return vocab


def get_src_vocab(src_train_path, config):
    return _get_vocab(src_train_path, config["src_vocab_size"], config["src_max_len"])


def get_trg_vocab(trg_train_path, config):
    return _get_vocab(trg_train_path, config["trg_vocab_size"], config["trg_max_len"])
