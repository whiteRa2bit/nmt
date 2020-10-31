import torch
import pandas as pd

from config import TRAIN_CONFIG, TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH, TEST_DIR, \
    TEST_LABELS_PATH, ID_COLUMN, LABEL_COLUMN, RANDOM_SEED
from dataset import ImageDataset
from model import Model
from trainer import Trainer
from predictor import Predictor


def _set_seed(seed=RANDOM_SEED):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def _transform_preds(pred, max_len=4):  # TODO: (@whiteRa2bit, 2020-09-20) Add to config
    pred = str(pred)
    pred = '0' * (max_len - len(pred)) + pred
    return pred


def main():
    # Train
    dataset = ImageDataset(TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH)
    model = Model(TRAIN_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    trainer = Trainer(model, optimizer, dataset, TRAIN_CONFIG)
    trainer.train()

    # Predict
    test_dataset = ImageDataset(TEST_DIR)
    model = Model(TRAIN_CONFIG)
    predictor = Predictor(model, TRAIN_CONFIG)
    preds, idxs = predictor.predict(test_dataset)

    preds = list(map(_transform_preds, preds))
    preds_df = pd.DataFrame({ID_COLUMN: idxs, LABEL_COLUMN: preds})
    preds_df.to_csv(TEST_LABELS_PATH, index=False)


if __name__ == '__main__':
    _set_seed()
    main()