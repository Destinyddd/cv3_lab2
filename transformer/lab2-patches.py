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

class Patches(paddle.nn.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        patches = F.unfold(images, self.patch_size, self.patch_size)
        return patches.transpose([0,2,1])

image_size = 32
patch_size = 8

image = anti_normalize(paddle.to_tensor(cifar100_test[np.random.choice(len(cifar100_test))][0]))
fig=plt.figure(figsize=(8, 4))
grid = plt.GridSpec(4, 8, wspace=0.5, figure=fig)
plt.subplot(grid[:4, :4])
plt.imshow(image)
plt.axis("off")

patches = Patches(patch_size)(image.transpose([2, 0, 1]).unsqueeze(0))

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

for i, patch in enumerate(patches[0]):
    plt.subplot(grid[i // 4, i % 4 + 4])
    patch_img = patch.reshape([3, patch_size, patch_size]).transpose([1,2,0])
    plt.imshow(patch_img)
    plt.axis("off")
