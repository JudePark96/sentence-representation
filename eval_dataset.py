__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from torch.utils.data import Dataset, DataLoader


class EvalDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        assert len(self.x) == len(self.y)
        return len(self.x)


def get_eval_loader(dataset: Dataset, bs: int) -> DataLoader:
    return DataLoader(dataset, batch_size=bs, shuffle=False)
