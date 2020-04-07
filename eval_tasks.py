__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import get_device_setting
from eval_data_utils import load_data
from data_utils import tokenize
from model import BertSE
from typing import Tuple
from tqdm import tqdm


import torch.nn as nn
import numpy as np
import torch


def extract_features(encoder: BertSE, data_loader: DataLoader) -> np.ndarray:
    encoder.eval()

    x_embedding, labels = [], []

    for x, y in tqdm(data_loader):
        # Evaluation Mode On.
        embed = encoder(x, is_eval = True)
        x_embedding.append(embed.cpu().detach().numpy())
        labels.append(y)

    x_embedding = np.array(x_embedding)
    x_embedding = np.concatenate(x_embedding, axis=0)

    encoder.train()

    # scikit-learn is avaliable at CPU operation.
    return x_embedding, labels


def fit_lr(lr: LogisticRegression, train_x: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    lr.fit(train_x, train_y)
    return lr


def get_lr_score(lr: LogisticRegression, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[float, float]:
    acc = lr.score(test_x, test_y)

    pred_y = lr.predict(test_x)
    f1 = f1_score(test_y, pred_y)

    return acc, f1


if __name__ == '__main__':
    texts, labels = load_data('MPQA', './rsc/eval_data/mpqa.all')
    # encode(BertTokenizer.from_pretrained('bert-base-uncased'), texts, labels)
    pass