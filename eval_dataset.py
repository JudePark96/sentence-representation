__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from data_utils import tokenize
from typing import Tuple, Any
from tqdm import tqdm


import numpy as np
import random


def encode(tokenizer: BertTokenizer, sentence: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encoding the eval task dataset
    :param tokenizer:
    :param text:
    :param labels:
    :return:
    """

    texts = np.array(tokenize(tokenizer, sentence, 50)['input_ids'])

    return texts


def load_data(task:str, path:str, tokenizer: BertTokenizer) -> Tuple[Any, Any, Any, Any]:
    """load binary classification dataset for evaluation"""
    """https://github.com/AcademiaSinicaNLPLab/sentiment_dataset"""

    eval_tasks = ['MPQA', 'SUBJ']

    if task not in eval_tasks:
        raise ValueError('confirmed evaluation tasks are [MPQA, SUBJ].')

    def aggregate_data(path: str) -> Tuple[list, list]:
        texts, labels = [], []
        length = []
        with open(path, 'r') as f:
            # print(f.readline().split(' '))    => ['0', 'complaining\n'] (label, data)
            for data in tqdm(f):
                res = data.split(' ')
                labels.append(int(res[0]))
                texts.append(encode(tokenizer, ' '.join(res[1:]).strip()))
                length.append(len(res[1:]))

            f.close()

        # TODO => 문장 길이 출력.

        print(f'maximum length: {max(length)}')
        print(f'minimum length: {min(length)}')
        print(f'mean length: {sum(length) / len(length)}')
        return texts, labels

    def zip_data(texts: str, labels: str) -> list:
        zipped = []
        for text, label in zip(texts, labels):
            zipped.append((text, label))
        return zipped

    def unzip_data(zipped_data: list) -> Tuple[list, list]:
        texts, labels = [], []

        for data in zipped_data:
            texts.append(data[0])
            labels.append(data[1])

        return texts, labels

    texts, labels = aggregate_data(path)
    zipped_data = zip_data(texts, labels)

    random.seed(49)
    random.shuffle(zipped_data)

    texts, labels = unzip_data(zipped_data)

    train_x, test_x, train_y, test_y = train_test_split(texts, labels, test_size=0.25)  # Default size.

    # 훈련 데이터 / 테스트 데이터 / 훈련 검증 데이터 / 테스트 검증 데이터
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    task = ['MPQA', 'SUBJ']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_x, test_x, train_y, test_y = load_data(task[0], './rsc/eval_data/mpqa.all', tokenizer)
    print(train_y[:10])
    print(train_x[:10])
    # print(tokenizer.decode(train_x[0]))