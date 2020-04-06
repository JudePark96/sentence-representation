__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch as T
import torch.nn as nn

from transformers import BertModel
from typing import Any

class BertSE(nn.Module):
    """
    Bert Sentence Embedding through Quick Though Model.
    """
    def __init__(self, bert: BertModel, is_lstm: bool = False) -> None:
        super().__init__()
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size
        self.lstm_hidden_dim = 256
        self.is_lstm = is_lstm

        if self.is_lstm:
            self.lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.lstm_hidden_dim,
                bidirectional=True,
                batch_first=True
            )

    def forward(self, input_ids: T.Tensor) -> Any:
        if self.is_lstm:
            bs, seq_len = input_ids.size()
            """
            Deep Contextualized Embedding through BI-LSTM
            """
            # [bs x seq_len x hidden_dim]
            ctx_seqs, tgt_seqs = self.get_last_hidden_states(input_ids), self.get_last_hidden_states(input_ids)

            # hidden => [2 x bs x hidden_dim]
            ctx, ctx_hid = self.lstm(ctx_seqs)
            tgt, tgt_hid = self.lstm(tgt_seqs)

            ctx_hid = ctx_hid[-1]
            tgt_hid = tgt_hid[-1]

            # [bs x (2 * lstm_hidden_dim)]
            ctx_hid = ctx_hid.view((bs, 2 * self.lstm_hidden_dim))
            tgt_hid = tgt_hid.view((bs, 2 * self.lstm_hidden_dim))

            scores = T.mm(ctx_hid, tgt_hid.T)
            mask = torch.eye(len(scores)).to().bool()
            scores = scores.masked_fill_(mask, 0)

            return nn.LogSoftmax(dim=1)(scores)
        else:
            # [bs x hidden_dim]
            ctx_seqs, tgt_seqs = self.get_pooled_output(input_ids), self.get_pooled_output(input_ids)

            # [bs x bs]
            scores = T.mm(ctx_seqs, tgt_seqs.T)
            mask = torch.eye(len(scores)).to().bool()
            scores = scores.masked_fill_(mask, 0)

            return nn.LogSoftmax(dim=1)(scores)

    def get_pooled_output(self, input_ids):
        # Reference => https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1]
        return pooled_output

    def get_last_hidden_states(self, input_ids):
        # TODO => [CLS] 토큰을 뺄지 말지.
        output = self.bert(input_ids=input_ids)
        last_hidden_states = output[0]
        return last_hidden_states

    def generate_targets(self, bs: int):
        # TODO => Label Smoothing 을 추가하는 것도 좋은 전략 중 하나다.
        return T.diag(T.ones(bs - 1), 1)


if __name__ == '__main__':
    import torch
    from transformers import BertTokenizer
    from data_utils import tokenize

    text = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = torch.LongTensor([tokenize(tokenizer, t)['input_ids'] for t in text])
    model = BertSE(BertModel.from_pretrained('bert-base-uncased'), False)
    output = model(text)
    print(output.shape)
    print(output)
    bert_lstm_model = BertSE(BertModel.from_pretrained('bert-base-uncased'), True)
    output = bert_lstm_model(text)
    print(output.shape)
    print(output)




