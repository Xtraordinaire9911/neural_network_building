# -*- coding: utf-8 -*-
# ---
# @File: model_transformer.py
# @Author: Ruiyao.J
# @Time: 6月 16, 2021
# ---

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_model, num_classes):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Embedding(nn.Module):
    def __init__(self, len_seq, d_model):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(in_features=len_seq, out_features=d_model)
        self.softmax = nn.Softmax(dim=d_model)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class PositionalEncoding:
    def __init__(self, len_seq, d_model):
        """"""
        super(PositionalEncoding, self).__init__()
        self.pe_sin = torch.sin_(torch.Tensor([[(pos/10000) ** ((2*i) / d_model) for pos in range(len_seq)]
                                               for i in range(d_model)]))
        self.pe_cos = torch.cos_(torch.Tensor([[(pos/10000) ** ((2*i) / d_model) for pos in range(len_seq)]
                                               for i in range(d_model)]))

    def forward(self, x):
        x = self.pe_sin(x)
        x = self.pe_cos(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, Q, K, V):
        super(ScaledDotProductAttention, self).__init__()
        self.query = Q
        self.key = K
        self.value = V
        self.d_k = torch.Tensor(len(K.size()))
        self.softmax = nn.Softmax()
        self.output = torch.matmul(self.softmax(torch.matmul(self.query, self.key.permute())
                                                / torch.sqrt_(self.d_k)), self.value)

    def forward(self):
        return self.output


class MultiHeadAttention(nn.Module):
    def __init__(self, Q, W_q, K, W_k, V, W_v, W_o, h=8, d_model = 512):
        super(MultiHeadAttention, self).__init__()
        self.d_k = len(K.size())
        """by default, d_k = d_model / h = 64"""
        self.d_v = len(V.size())
        """by default, d_v = d_model / h = 64"""
        self.branch_q = torch.Tensor([nn.Linear(self.d_k, self.d_k)(Q) for _ in range(h)])
        self.branch_k = torch.Tensor([nn.Linear(self.d_k, self.d_k)(K) for _ in range(h)])
        self.branch_v = torch.Tensor([nn.Linear(self.d_v, self.d_v)(V) for _ in range(h)])
        self.attention = torch.Tensor([ScaledDotProductAttention(
                        self.branch_q(torch.matmul(Q[i], W_q[i])),
                        self.branch_k(torch.matmul(K[i], W_k[i])),
                        self.branch_v(torch.matmul(V[i], W_v[i]))) for i in range(h)])
        self.concat = torch.matmul(torch.cat(self.attention, dim=-1), W_o)
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        return self.linear


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super(AddNorm, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=d_model)

    def forward(self, x, y):
        ret = self.norm(x + y)
        return ret


class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super(PointwiseFeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=d_model * 4, out_features=d_model)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


"""
Encoder, EncoderLayer
Decoder, DecoderLayer
SublayerLogic
DecoderGenerator
"""

