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

class AutoTransforms(BaseTransform):
    def __init__(self, transforms=None, keys=None):
        super(AutoTransforms, self).__init__(keys)
        self.transforms = transforms

    def _apply_image(self, image):
        if self.transforms is None: 
            return image
        choose=np.random.randint(0, len(self.transforms))
        return self.transforms[choose](image)

# 训练集数据增强
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transforms_list= [
    transforms.BrightnessTransform(0.5),
    transforms.SaturationTransform(0.5),
    transforms.ContrastTransform(0.5),
    transforms.HueTransform(0.5),
    transforms.RandomRotation(15,
                              expand=True,
                              fill=128),
    transforms.ColorJitter(0.5,0.5,0.5,0.5),
    transforms.Grayscale(3)
]

train_tx = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           AutoTransforms(transforms_list),
           transforms.RandomCrop(32),
           transforms.RandomVerticalFlip(),
           transforms.Transpose(),
           transforms.Normalize(0.0, 255.0),
           transforms.Normalize(mean, std)
])

val_tx = transforms.Compose([
         transforms.Transpose(),
         transforms.Normalize(0.0, 255.0),
         transforms.Normalize(mean, std)
])

cifar100_train = paddle.vision.datasets.Cifar100(mode='train', transform=train_tx, download=True)
cifar100_test = paddle.vision.datasets.Cifar100(mode='test', transform=val_tx, download=True)

print('训练集数量:', len(cifar100_train), '训练集图像尺寸', cifar100_train[0][0].shape)
print('测试集数量:', len(cifar100_test), '测试集图像尺寸', cifar100_test[0][0].shape)
plot_num_images(25, cifar100_train)
