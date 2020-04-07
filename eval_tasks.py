__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer
from eval_dataset import load_data
from data_utils import tokenize
from tqdm import tqdm


import torch.nn as nn
import numpy as np

def encode(tokenizer: BertTokenizer, texts: list, labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encoding the eval task dataset
    :param tokenizer:
    :param text:
    :param labels:
    :return:
    """

    texts = np.array([tokenize(tokenizer, text, 50)['input_ids'] for text in tqdm(texts)])
    labels = np.array(labels)

    print(texts[0])
    print(texts[1])
    print(texts[2])
    return texts, labels


def extract_features(encoder: nn.Module, texts: np.ndarray, labels) -> None:


    pass


if __name__ == '__main__':
    texts, labels = load_data('MPQA', './rsc/eval_data/mpqa.all')
    encode(BertTokenizer.from_pretrained('bert-base-uncased'), texts, labels)
    pass