__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Tuple, Any

from torch.utils.data import Dataset

from data_utils import tokenize
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


class EvalDataset(Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()


def load_data(task:str, path:str) -> Tuple[list, list]:
    """load binary classification dataset for evaluation"""
    eval_tasks = ['MPQA', 'SUBJ']

    if task not in eval_tasks:
        raise ValueError('confirmed evaluation tasks is [MPQA, SUBJ].')

    def aggregate_data(path: str) -> Tuple[list, list]:
        texts, labels = [], []
        with open(path, 'r') as f:
            # print(f.readline().split(' '))    => ['0', 'complaining\n'] (label, data)
            for data in tqdm(f):
                res = data.split(' ')
                labels.append(int(res[0]))
                texts.append(res[1].strip())

            f.close()

        text, labels = shuffle(texts, labels, random_state=49)
        return text, labels

    return aggregate_data(path)


if __name__ == '__main__':
    task = ['MPQA', 'SUBJ']
    load_data(task[0], './rsc/eval_data/mpqa.all')