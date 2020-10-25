import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()  # initialize base class to track sub-layers, trainable variables, etc.

        self.vocab = vocab
        self.eos_idx = vocab.word_to_idx[vocab._EOS]
        n_tokens = vocab.vocab_size
        self.emb = nn.Embedding(n_tokens, config["enc_emb_size"], self.eos_idx)
        self.lstm = nn.LSTM(config["enc_emb_size"], config["enc_hid_size"], 2, batch_first=True)

    def forward(self, inp):
        # Shape: (batch_size, seq_len, emb_size)
        emb = self.emb(inp.clone())
        # Shape: (batch_size, seq_len, hid_size)
        lengths = (inp != self.eos_idx).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        enc_seq, hidden = self.lstm(emb)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]

        return last_state
    
    @property
    def name(self):
        return "baseline_encoder"


class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocab = vocab
        self.eos_idx = vocab.word_to_idx[vocab._EOS]
        n_tokens = vocab.vocab_size
        self.emb = nn.Embedding(n_tokens, config["dec_emb_size"], self.eos_idx)
        self.gru = nn.GRU(config["dec_emb_size"], config["dec_hid_size"], batch_first=True)
        self.fc = nn.Linear(config["dec_hid_size"], n_tokens)

    def forward(self, inp, hidden):
        # Shape: (batch_size, seq_len, emb_size)
        emb = self.emb(inp.clone())

        output, _ = self.gru(emb, hidden.unsqueeze(0))
        output = self.fc(output)
        return output

    @property
    def name(self):
        return "baseline_decoder"
