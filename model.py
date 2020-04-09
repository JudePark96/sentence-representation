# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

"""
An Efficient Framework for Learning Sentence Representations (ICLR 2018)
Advanced Implementation
"""


import torch
import torch.nn as nn

from config import get_device_setting
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
        self.device = get_device_setting()

        if self.is_lstm:
            self.ctx_lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.lstm_hidden_dim,
                bidirectional=True,
                batch_first=True
            )

            self.tgt_lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.lstm_hidden_dim,
                bidirectional=True,
                batch_first=True
            )

    def forward(self, input_ids: torch.Tensor, is_eval:bool = False) -> torch.Tensor:
        bs, seq_len = input_ids.size()

        if is_eval:
            if self.is_lstm:
                ctx_seqs, tgt_seqs = self.get_last_hidden_states(input_ids), self.get_last_hidden_states(input_ids)
                ctx, ctx_hid = self.ctx_lstm(ctx_seqs)
                tgt, tgt_hid = self.tgt_lstm(tgt_seqs)

                ctx_hid = ctx_hid[-1]
                tgt_hid = tgt_hid[-1]

                # [bs x (2 * lstm_hidden_dim)]
                ctx_hid = ctx_hid.view((bs, 2 * self.lstm_hidden_dim))
                tgt_hid = tgt_hid.view((bs, 2 * self.lstm_hidden_dim))

                # [bs x (2 * (ctx_hid + tgt_hid))]
                ctx_tgt_hid = torch.cat((ctx_hid, tgt_hid), dim=1)

                return ctx_tgt_hid
            else:
                return self.get_pooled_output(input_ids)
        else:
            if self.is_lstm:
                """
                Deep Contextualized Embedding through BI-LSTM
                """
                # [bs x seq_len x hidden_dim]
                ctx_seqs, tgt_seqs = self.get_last_hidden_states(input_ids), self.get_last_hidden_states(input_ids)

                # hidden => [2 x bs x hidden_dim]
                ctx, ctx_hid = self.ctx_lstm(ctx_seqs)
                tgt, tgt_hid = self.tgt_lstm(tgt_seqs)

                ctx_hid = ctx_hid[-1]
                tgt_hid = tgt_hid[-1]

                # [bs x (2 * lstm_hidden_dim)]
                ctx_hid = ctx_hid.view((bs, 2 * self.lstm_hidden_dim))
                tgt_hid = tgt_hid.view((bs, 2 * self.lstm_hidden_dim))

                scores = torch.matmul(ctx_hid, tgt_hid.transpose(0, 1))
                mask = torch.eye(len(scores)).to(self.device).bool()
                scores = scores.masked_fill(mask, 0)

                # [bs x bs]
                return nn.LogSoftmax(dim=1).to(get_device_setting())(scores)
            else:
                # [bs x hidden_dim]
                ctx_seqs, tgt_seqs = self.get_pooled_output(input_ids), self.get_pooled_output(input_ids)

                # [bs x bs]
                scores = torch.matmul(ctx_seqs, tgt_seqs.transpose(0, 1))
                mask = torch.eye(len(scores)).to(self.device).bool()
                scores = scores.masked_fill(mask, 0)

                # [bs x bs]
                return nn.LogSoftmax(dim=1).to(get_device_setting())(scores)

    def get_pooled_output(self, input_ids) -> torch.Tensor:
        # Reference => https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1]
        return pooled_output

    def get_last_hidden_states(self, input_ids) -> torch.Tensor:
        # TODO => [CLS] 토큰을 뺄지 말지.
        output = self.bert(input_ids=input_ids)
        last_hidden_states = output[0]
        return last_hidden_states

    def generate_targets(self, bs: int) -> torch.Tensor:
        return torch.diag(torch.ones(bs - 1), 1).to(get_device_setting())

    def generate_smooth_targets(self, bs, offsetlist=[1], smooth_rate=0.1) -> torch.Tensor:
        targets = torch.zeros(bs, bs, device=self.device).fill_(smooth_rate)
        for offset in offsetlist:
            targets += torch.diag(torch.ones(bs-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets

    def generate_smooth_scaling_targets(self, bs):
        targets = torch.zeros(bs, bs, device=self.device)
        for offset, scale in zip([-3, -2, -1, 1, 2, 3], [5, 10, 10, 5]):
            targets += scale * torch.diag(torch.ones(bs-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets


if __name__ == '__main__':
    pass




