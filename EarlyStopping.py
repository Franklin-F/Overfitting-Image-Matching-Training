class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # 容忍的 epoch 数
        self.min_delta = min_delta  # 验证损失的最小变化量
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
