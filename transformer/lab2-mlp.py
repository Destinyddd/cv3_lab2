import paddle
import time
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
import numpy as np
import paddle.optimizer.lr as lrScheduler
from paddle.vision.transforms import BaseTransform
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os


class Mlp(nn.Layer):
    def __init__(self, feats, mlp_hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(feats, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, feats)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
