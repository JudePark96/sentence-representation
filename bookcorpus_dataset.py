__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data_utils import tokenize
from config import data_config
from typing import Any
from tqdm import tqdm


import torch as T


class BookCorpusDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, corpus: list, max_len: int) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index: int) -> Any:
        return T.LongTensor(tokenize(self.tokenizer, self.corpus[index], self.max_len)['input_ids'])

    def __len__(self) -> int:
        return len(self.corpus)


def get_data_loader(dataset: Dataset, bs: int) -> DataLoader:
    return DataLoader(dataset, batch_size=bs, shuffle=False)


if __name__ == '__main__':
    corpus = []

    with open(data_config['bookcorpus_path'], 'r') as f:
        for seq in tqdm(f):
            corpus.append(seq.strip())

        f.close()
