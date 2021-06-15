import torch
import torch.nn as nn
import numpy as np


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        
    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        pass


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        pass


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        pass
