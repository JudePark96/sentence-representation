__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from sklearn.linear_model import LogisticRegression

from bookcorpus_dataset import get_data_loader, BookCorpusDataset
from eval_dataset import EvalDataset, get_eval_loader
from eval_tasks import extract_features, fit_lr, get_lr_score
from config import get_device_setting, data_config
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from eval_data_utils import load_data
from model import BertSE
from tqdm import tqdm


import torch.optim as optim
import torch.nn as nn
import torch as T


class Trainer(object):
    def __init__(self,
                 model: BertSE,
                 tokenizer: BertTokenizer,
                 train_loader: DataLoader,
                 eval_train_loader: DataLoader,
                 eval_test_loader: DataLoader,
                 epoch:int) -> None:
        self.device = get_device_setting()
        self.epoch = epoch
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-5)
        self.loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.writer = SummaryWriter()
        self.train_loader = train_loader
        self.eval_train_loader = eval_train_loader
        self.eval_test_loader = eval_test_loader


    def train(self):
        steps = 0
        for iter in tqdm(range(self.epoch)):
            for idx, input_ids in tqdm(enumerate(self.train_loader)):
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

                # TODO => 잘되나 확인
                self.evaluate(steps=steps)

                if (steps) % 100 == 0:
                    self.writer.add_scalar('Train/KL_Loss', kl_loss.item(), steps)
                    print(f'kl_loss: {kl_loss}')

                if (steps + 1) % 1000 == 0:
                    T.save(self.model.state_dict(), './checkpoint/' + f'model-{iter}-{idx}.pt')

    def evaluate(self, eval_train_loader: DataLoader, eval_test_loader: DataLoader, steps: int) -> None:
        lr = LogisticRegression(C=1.0)

        print('*************** eval_training ... ***************')

        for i, (x, y) in enumerate(eval_train_loader):
            train_x_embedding = extract_features(self.model, x)
            lr = fit_lr(lr, train_x_embedding, y)

        for i, (x, y) in enumerate(eval_test_loader):
            test_x_embedding = extract_features(self.model, x)
            acc, f1 = get_lr_score(lr, test_x_embedding, y)

            self.writer.add_scalar('Eval/MRQA_acc', acc, i)
            self.writer.add_scalar('Eval/MRQA_f1', f1, i)


if __name__ == '__main__':
    corpus = []

    with open(data_config['bookcorpus_path'], 'r') as f:
        for i, seq in tqdm(enumerate(f)):
            corpus.append(seq.strip())
            if i == 10000:
                # 빠른 테스트를 위해서
                break

        f.close()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertSE(bert_model, False)

    train_x, test_x, train_y, test_y = load_data('MPQA', './rsc/eval_data/mpqa.all', tokenizer)

    # 학습 데이터
    dataset = BookCorpusDataset(tokenizer, corpus, 50)
    train_loader = get_data_loader(dataset, 32)

    # 검증 - 학습 데이터
    eval_train_dataset = EvalDataset(train_x, train_y)
    eval_train_loader = get_eval_loader(eval_train_dataset, 32)

    # 검증 - 테스트 데이터
    eval_test_dataset = EvalDataset(test_x, test_y)
    eval_test_loader = get_eval_loader(eval_test_dataset, 32)


    Trainer(model, tokenizer, train_loader, eval_train_loader, eval_test_loader, 5).train()

    pass
