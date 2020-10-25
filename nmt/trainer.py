import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb
from loguru import logger

from nmt.utils import get_checkpoint_path
from nmt.metrics import compute_bleu
from nmt.config import CONFIG, WANDB_PROJECT, SAMPLES_NUM, CHECKPOINT_DIR


class Trainer:
    def __init__(self, encoder, decoder, optimizer, train_dataset, val_dataset, config=CONFIG["train"]):
        self.encoder = encoder.to(config['device'])
        self.decoder = decoder.to(config['device'])
        self.optimizer = optimizer
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.src_vocab = train_dataset.src_vocab
        self.trg_vocab = train_dataset.trg_vocab
        self.train_dataloader = DataLoader(train_dataset, self.config["train_batch_size"], True)
        self.val_dataloader = DataLoader(val_dataset, self.config["val_batch_size"])

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.encoder)
        wandb.watch(self.decoder)

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self._initialize_wandb()

        best_val_bleu = 0
        for epoch in range(self.config['epochs_num']):
            logger.info(f"Epoch {epoch} started...")
            for idx, item in tqdm.tqdm(enumerate(self.train_dataloader)):
                inp, out = item
                inp = inp.to(self.config["device"])
                out = out.to(self.config["device"])

                self.optimizer.zero_grad()

                enc_out = self.encoder(inp)
                logits = self.decoder(out, enc_out)
                loss = self._compute_loss(logits, inp, out)

                loss.backward()
                self.optimizer.step()

                if idx % self.config['log_each'] == 0:
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_samples = val_metrics['samples']
                    val_bleu = val_metrics['bleu']

                    wandb.log({
                        "Train Loss": loss.item(), \
                        "Val Loss": val_loss.item(), \
                        "Val Bleu": val_bleu, \
                        "Translations": wandb.Table(data=val_samples, columns=["Real", "Translation"])
                    })

                    if val_bleu > best_val_bleu:
                        self._save_checkpoint(self.encoder)
                        self._save_checkpoint(self.decoder)
                        best_val_bleu = val_bleu

        logger.info(f"Training finished. Best validation bleu: {best_val_bleu}")

    def _compute_loss(self, logits, inp, out):
        eos_ix = self.trg_vocab.word_to_idx[self.trg_vocab._EOS]
        mask = self._compute_mask(out, eos_ix)
        out = out.clone()
        out[~mask.bool()] = -1
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        loss = criterion(logits[:, :-1].reshape(-1, logits.shape[-1]), out[:, 1:].flatten())

        return loss

    def _compute_metrics(self, dataloader, samples_num=SAMPLES_NUM):
        self.encoder.eval()
        self.decoder.eval()

        inp = []
        out = []
        logits = []
        for batch_item in dataloader:
            batch_inp, batch_out = batch_item
            batch_inp = batch_inp.to(self.config["device"])
            batch_out = batch_out.to(self.config["device"])

            with torch.no_grad():
                batch_enc_out = self.encoder(batch_inp)
                batch_logits = self.decoder(batch_out, batch_enc_out)

            inp.append(batch_inp)
            out.append(batch_out)
            logits.append(batch_logits)

        out = torch.cat(out)
        inp = torch.cat(inp)
        logits = torch.cat(logits)
        translations = self._translate(logits)

        loss = self._compute_loss(logits, inp, out)
        out = out.detach().cpu().numpy()
        real = self.trg_vocab.decode_idxs(out)
        bleu = compute_bleu(translations, real)

        sample_idxs = np.random.choice(range(len(dataloader.dataset)), samples_num)
        samples_real = np.array(real)[sample_idxs]
        samples_pred = np.array(translations)[sample_idxs]
        samples = list(zip(samples_real, samples_pred))

        self.encoder.train()
        self.decoder.train()

        return {"loss": loss, "bleu": bleu, "samples": samples}

    def _save_checkpoint(self, model):
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, wandb.run.id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model.name}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    def _translate(self, logits):  # TODO: (@whiteRa2bit, 2020-10-24) Move to another module
        pred_idxs = torch.argmax(logits, dim=2)
        pred_idxs = pred_idxs.detach().cpu().numpy()
        texts = self.trg_vocab.decode_idxs(pred_idxs)

        return texts

    @staticmethod
    def _compute_mask(input_ix, eos_ix):
        return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, \
                    pad=(1, 0, 0, 0), value=True)
