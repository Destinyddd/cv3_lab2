class LabelSmoothingCrossEntropyLoss(nn.Layer):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, axis=self.dim)
        with paddle.no_grad():
            true_dist = paddle.ones_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.put_along_axis_(target.unsqueeze(1), self.confidence, 1)
        return paddle.mean(paddle.sum(-true_dist * pred, axis=self.dim))

def get_scheduler(epochs, warmup_epochs, learning_rate):
    base_scheduler = lrScheduler.CosineAnnealingDecay(learning_rate=learning_rate, T_max=epochs, eta_min=1e-5, verbose=False)
    scheduler = lrScheduler.LinearWarmup(base_scheduler, warmup_epochs, 1e-5, learning_rate, last_epoch=-1, verbose=False)
    return scheduler
