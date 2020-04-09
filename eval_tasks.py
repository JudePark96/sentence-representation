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
from config import get_device_setting


import torch.nn as nn
import numpy as np
import torch


def extract_features(encoder: BertSE, data_loader: DataLoader) -> np.ndarray:
    encoder.eval()

    x_embedding, labels = [], []

    for x, y in tqdm(data_loader):
        # Evaluation Mode On.
        embed = encoder(x.to(get_device_setting()), is_eval = True)
        x_embedding.append(embed.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

    x_embedding = np.array(x_embedding)
    x_embedding = np.concatenate(x_embedding, axis=0)
    labels = np.array(labels)
    labels = np.concatenate(labels, axis=0)

    print(x_embedding.shape)
    print(type(x_embedding), type(labels))
    print(labels.shape)

    encoder.train()

    # scikit-learn is avaliable at CPU operation.
    return x_embedding, np.array(labels)


def fit_lr(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=49)
    
    # Torch -> Numpy
    # train_x, train_y = train_x.cpu().detach().numpy(), train_y.cpu().detach().numpy()
    # test_x, test_y = test_x.cpu().detach().numpy(), test_y.cpu().detach().numpy()
    lr.fit(train_x, train_y)

    # Test 에 대하여 점수를 낸다.
    acc = lr.score(test_x, test_y)
    pred_y = lr.predict(test_x)
    f1 = f1_score(test_y, pred_y)

    return acc, f1


if __name__ == '__main__':
    texts, labels = load_data('MPQA', './rsc/eval_data/mpqa.all')
    # encode(BertTokenizer.from_pretrained('bert-base-uncased'), texts, labels)
    pass