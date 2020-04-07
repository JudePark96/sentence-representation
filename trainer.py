__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from bookcorpus_dataset import get_data_loader, BookCorpusDataset
from transformers import BertModel, BertTokenizer
from config import get_device_setting, data_config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model import BertSE
from tqdm import tqdm


import torch.optim as optim
import torch.nn as nn
import torch as T


class Trainer(object):
    def __init__(self, model: BertSE, epoch:int) -> None:
        self.device = get_device_setting()
        self.model = model.to(self.device)
        self.epoch = epoch
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-5)
        self.loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.writer = SummaryWriter()


    def train(self, train_loader: DataLoader):
        steps = 0
        for iter in tqdm(range(self.epoch)):
            for idx, input_ids in tqdm(enumerate(train_loader)):
                steps += 1
                self.optimizer.zero_grad()
                input_ids = input_ids.to(self.device)
                bs, seq_len = input_ids.size()
                target = self.model.generate_smooth_targets(bs)
                output = self.model(input_ids)

                kl_loss = self.loss(output, target)
                kl_loss.backward()

                # Gradient Clipping
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 2.0)
                self.optimizer.step()

                if (steps) % 100 == 0:
                    self.writer.add_scalar('kl_loss/train', kl_loss.item(), steps)
                    print(f'kl_loss: {kl_loss}')

                if (steps + 1) % 1000 == 0:
                    T.save(self.model.state_dict(), './checkpoint/' + f'model-{iter}-{idx}.pt')

    def evaluate(self):

        pass


if __name__ == '__main__':
    corpus = []

    with open(data_config['bookcorpus_path'], 'r') as f:
        for seq in tqdm(f):
            corpus.append(seq.strip())

        f.close()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dataset = BookCorpusDataset(tokenizer, corpus, 50)
    train_loader = get_data_loader(dataset, 32)
    model = BertSE(bert_model, False)

    Trainer(model, 5).train(train_loader)

    pass
