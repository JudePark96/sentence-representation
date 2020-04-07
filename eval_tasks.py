__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer
from config import get_device_setting
from eval_dataset import load_data
from data_utils import tokenize
from model import BertSE
from typing import Tuple
from tqdm import tqdm


import torch.nn as nn
import numpy as np
import torch


def extract_features(encoder: BertSE, texts: np.ndarray) -> np.ndarray:
    encoder.eval()

    # [bs x embed_dim(hid_dim)]
    sentence_embedding = encoder(torch.from_numpy(texts).long().to(get_device_setting()), is_eval=True)

    encoder.train()

    # scikit-learn is avaliable at CPU operation.
    return sentence_embedding.cpu().detach().numpy()


def fit_lr(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
                            random_state:int, c: float) -> Tuple[float, float]:

    lr = LogisticRegression(random_state=random_state, C=c)
    lr.fit(train_x, train_y)
    acc = lr.score(test_x, test_y)

    pred_y = lr.predict(test_x)
    f1 = f1_score(test_y, pred_y)

    return acc, f1


if __name__ == '__main__':
    texts, labels = load_data('MPQA', './rsc/eval_data/mpqa.all')
    # encode(BertTokenizer.from_pretrained('bert-base-uncased'), texts, labels)
    pass