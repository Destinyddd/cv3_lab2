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

EPOCHS = 200
BATCH_SIZE = 128
NUM_CLASSES = 100
WARMUP_EPOCHS = 5
LR = 1e-3

scheduler = get_scheduler(epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, learning_rate=LR)
optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=Model.parameters(), weight_decay=5e-5)
criterion = LabelSmoothingCrossEntropyLoss(NUM_CLASSES, smoothing=0.1)

train_loader = DataLoader(cifar100_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(cifar100_test, batch_size=BATCH_SIZE * 16, shuffle=False, num_workers=0, drop_last=False)

def train_epoch(model, epoch, interval=20):
    acc_num = 0
    total_samples = 0
    nb = len(train_loader)
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=nb, colour='red', disable=((epoch + 1) % interval != 0))
    pbar.set_description(f'EPOCH: {epoch:3d}')
    for _, (_, data) in enumerate(pbar):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        loss = criterion(predicts, y_data)
        loss_item = loss.item()
        acc_num += paddle.sum(predicts.argmax(1) == y_data).item()
        total_samples += y_data.shape[0]
        total_acc = acc_num / total_samples
        current_lr = optim.get_lr()
        loss.backward()
        pbar.set_postfix(train_loss=f'{loss_item:5f}', train_acc=f'{total_acc:5f}', train_lr=f'{current_lr:5f}')
        optim.step()
        optim.clear_grad()
    scheduler.step()

@paddle.no_grad()
def validation(model, epoch, interval=20):
    model.eval()
    acc_num = 0
    total_samples = 0
    nb = len(test_loader)
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=nb, colour='green', disable=((epoch + 1) % interval != 0))
    pbar.set_description(f'EVAL')
    for _, (_, data) in enumerate(pbar):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        acc_num += paddle.sum(predicts.argmax(1) == y_data).item()
        total_samples += y_data.shape[0]
        batch_acc = paddle.metric.accuracy(predicts, y_data.unsqueeze(1)).item()
        total_acc = acc_num / total_samples
        pbar.set_postfix(eval_batch_acc=f'{batch_acc:4f}', total_acc=f'{total_acc:4f}')

start = time.time()
for epoch in range(EPOCHS):
    train_epoch(Model, epoch)
    validation(Model, epoch)
    if (epoch + 1) % 50 == 0:
        paddle.save(Model.state_dict(), str(epoch + 1) + '.pdparams')
paddle.save(Model.state_dict(), 'finished.pdparams')
end = time.time()
print('Training Cost ', (end-start) / 60, 'minutes')
