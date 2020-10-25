import torch
import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocab = vocab
        self.eos_idx = vocab.word_to_idx[vocab._EOS]
        self.bos_idx = vocab.word_to_idx[vocab._SOS]
        n_tokens = vocab.vocab_size

        self.enc_emb = nn.Embedding(n_tokens, config["enc_emb_size"], self.eos_idx)
        self.enc_lstm = nn.LSTM(config["enc_emb_size"], config["enc_hid_size"], 2, batch_first=True)
        self.dec_emb = nn.Embedding(n_tokens, config["dec_emb_size"], self.eos_idx)
        self.dec_gru = nn.GRUCell(config["dec_emb_size"], config["dec_hid_size"])
        self.dec_fc = nn.Linear(config["dec_hid_size"], n_tokens)

    def forward(self, inp_tokens, out_tokens):
        """ Apply model in training mode """
        initial_state = self.encode(inp_tokens)
        return self.decode(initial_state, out_tokens)

    def encode(self, inp_tokens, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors, one or many
        """
        # Shape: (batch_size, seq_len, emb_size)
        emb = self.enc_emb(inp_tokens.clone())  # TODO: Remove clone
        # Shape: (batch_size)
        lengths = (inp_tokens != self.eos_idx).to(torch.int64).sum(dim=1).clamp_max(inp_tokens.shape[1] - 1)
        # Shape: (batch_size, seq_len, hid_size)
        enc_seq, _ = self.enc_lstm(emb)
        # Shape: (batch_size, hid_size)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]

        return last_state

    def decode_step(self, prev_state, prev_tokens):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(out_voc)]
        """
        # Shape: [batch_size, emb_size]
        prev_embs = self.dec_emb(prev_tokens)
        # Shape: [batch_size, hid_size]
        next_state = self.dec_gru(prev_embs, prev_state)
        # Shape: [batch_size, vocab_size]
        logits = self.dec_fc(next_state)

        return next_state, logits

    def decode(self, initial_state, out_tokens, **flags):
        """ 
        Iterate over reference tokens (out_tokens) with decode_step
        :param initial_state: Initial state for decoder with shape [batch_dize, hid_size]
        :param out_tokens: Output tokens, an int vector of [batch_size, seq_len]
        :return: A tensor with logits with shape [batch_size, seq_len, vocab_size]
        """
        batch_size = out_tokens.shape[0]
        state = initial_state

        logits_sequence = []
        for i in range(out_tokens.shape[1]):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Generate translations from model (greedy version) """
        batch_size, device = len(initial_state), initial_state.device
        state = initial_state

        outputs = [torch.full([batch_size], self.bos_idx, dtype=torch.int64, device=device)]
        # all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            # all_states.append(state)

        return torch.stack(outputs, dim=1)

    @property
    def name(self):
        return "baseline"
