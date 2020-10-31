import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb
from loguru import logger

from nmt.metrics import compute_bleu
from nmt.config import CONFIG, WANDB_PROJECT, SAMPLES_NUM, CHECKPOINT_DIR


class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, config=CONFIG):
        self.full_config = config
        self.config = config["train"]
        self.model = model.to(self.config['device'])
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.src_vocab = train_dataset.src_vocab
        self.trg_vocab = train_dataset.trg_vocab
        self.train_dataloader = DataLoader(train_dataset, self.config["train_batch_size"], True)
        self.val_dataloader = DataLoader(val_dataset, self.config["val_batch_size"])

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.full_config, project=project_name)
        wandb.watch(self.model)

    def train(self):
        self.model.train()
        self._initialize_wandb()

        best_val_bleu = 0
        for epoch in range(self.config['epochs_num']):
            logger.info(f"Epoch {epoch} started...")
            for idx, item in tqdm.tqdm(enumerate(self.train_dataloader)):
                inp, out = item
                inp = inp.to(self.config["device"])
                out = out.to(self.config["device"])

                self.optimizer.zero_grad()

                enc_last_state = self.model.encode(inp)
                logits = self.model.decode(enc_last_state, out)
                loss = self._compute_loss(logits, inp, out)

                loss.backward()
                self.optimizer.step()

                if idx % self.config['log_each'] == 0:
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_samples = val_metrics['samples']
                    val_bleu = val_metrics['bleu']

                    wandb.log({
                        "Train Loss": loss.item(), \
                        "Val Bleu": val_bleu, \
                        "Translations": wandb.Table(data=val_samples, columns=["Real", "Translation"])
                    })

                    if val_bleu > best_val_bleu:
                        self._save_checkpoint(self.model)
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
        self.model.eval()

        out = []
        preds = []
        for batch_item in dataloader:
            batch_inp, batch_out = batch_item
            batch_inp = batch_inp.to(self.config["device"])
            batch_out = batch_out.to(self.config["device"])

            with torch.no_grad():
                batch_enc_last_state = self.model.encode(batch_inp)
                batch_preds = self.model.decode_inference(batch_enc_last_state)

            out.append(batch_out)
            preds.append(batch_preds)

        real = self._to_texts(out)
        translations = self._to_texts(preds)
        bleu = compute_bleu(translations, real)

        sample_idxs = np.random.choice(range(len(dataloader.dataset)), samples_num)
        samples_real = np.array(real)[sample_idxs]
        samples_pred = np.array(translations)[sample_idxs]
        samples = list(zip(samples_real, samples_pred))

        self.model.train()

        return {"bleu": bleu, "samples": samples}

    def _save_checkpoint(self, model):
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, wandb.run.id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model.name}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    def _to_texts(self, out_idxs):  # TODO: (@whiteRa2bit, 2020-10-24) Move to another module
        out_idxs = torch.cat(out_idxs)
        out_idxs = out_idxs.detach().cpu().numpy()
        texts = self.trg_vocab.decode_idxs(out_idxs)

        return texts

    @staticmethod
    def _compute_mask(input_ix, eos_ix):
        return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, \
                    pad=(1, 0, 0, 0), value=True)
