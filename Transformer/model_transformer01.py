# -*- coding: utf-8 -*-
# ---
# @File: model_transformer.py
# @Author: Ruiyao.J
# @Ref: 《Attention is all you need》, by A.Vaswani et al.
# @Time: 6月 16, 2021
# ---


import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        """
        Args:
            vocab_size: torch.Tensor
                    the length of the vocabulary to be embedded
            model_dim: int
                    the dimension of the model, i.e. the dimension of outputs of sub-layers
                    (Page 3, Chapter 3.1, Encoder and Decoder Stacks - Encoder)
        """
        super(Embedding, self).__init__()
        self.model_dim = model_dim
        self.embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        # ↑ of shape (vocab_size, model_dim)

    def forward(self, vocab_to_embed):
        """
        Inputs:
            vocab_to_embed: torch.Tensor (ndim == 2)
                    vocabulary to be embedded, of shape (batch_size, seq_len)
        Returns:
            embedding_table: nn.Layer
                    with weights being torch.Tensor (ndim == 3), of shape (batch_size, seq_len, model_dim)
        """
        embedding_table = self.embedding_table(vocab_to_embed) * np.sqrt(self.model_dim)
        return embedding_table


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout_prob, max_seq_len=5000):
        """
        Args:
            model_dim: int
            dropout_prob: float
            max_seq_len: int
        """
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.dropout = nn.Dropout(p=dropout_prob)
        self.encoding_table = torch.zeros(size=(max_seq_len, model_dim))
        for pos in range(max_seq_len):
            for i in range(model_dim):
                if i % 2 == 0:
                    self.encoding_table[pos, i] = np.sin(pos / (10000 ** (i / model_dim)))
                else:
                    self.encoding_table[pos, i] = np.cos(pos / (10000 ** ((i-1) / model_dim)))

    def forward(self, embedding_table):
        """
        Args:
            embedding_table: nn.Layer
                    with weights being torch.Tensor (ndim==3), of shape (batch_size, vocab_len, model_dim)

        Returns:
            embedding_encoding_table: nn.Layer
                    with weights being torch.Tensor (ndim == 3), of shape (batch_size, vocab_len, model_dim),
                    where vocab_len is a init_param of class Embedding
        """
        assert embedding_table.ndim == 3 and embedding_table[-1] == self.encoding_table[-1], \
            "Expected embedding_table's of shape (batch_size, {}, {}), got {}".format(self.max_seq_len, self.model_dim, embedding_table.shape)
        self.encoding_table = self.encoding_table[:embedding_table.shape[1], :]
        embedding_encoding_table = self.dropout(embedding_table + self.encoding_table)
        # in the addition ↑, self.encoding_table will be broadcast into shape (batch_size, ... , ...)
        return embedding_encoding_table


class MultiHeadAttention(nn.Module):
    def __init__(self, q, k, v, mask, model_dim, num_heads, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.q = q
        self.k = k
        self.v = v
        self.mask = mask
        self.head_dim = int(model_dim / num_heads)
        self.linear1 = nn.Sequential(*[nn.Linear(in_features=self.model_dim, out_features=model_dim) for _ in range(3)])
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(in_features=model_dim, out_features=model_dim)

    def scaled_dot_product_attention(self):
        """
        Returns:
            weighted_score: torch.Tensor
                    of shape (batch_size, )
        """
        score = torch.matmul(self.Q, self.K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if self.mask is not None:
            score.masked_fill_(self.mask == torch.Tensor(False), float("-inf"))
        score = self.dropout(self.softmax(score))
        weighted_score = torch.matmul(score, self.V)
        return weighted_score

    def forward(self):
        pass


class PointwiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PointwiseFeedForwardNet, self).__init__()
        pass


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        pass


if __name__ == "__main__":
    pass























