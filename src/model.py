import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class AudioClassificationModel(L.LightningModule):
    def __init__(self, num_classes, input_shape, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.network = BaseNetwork(num_classes, input_shape)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.lr = lr

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class BaseNetwork(nn.Module):
    def __init__(self, num_classes, input_shape=None):
        super().__init__()
        in_channels = input_shape[0]

        def conv_block(in_f, out_f, use_pool=True, use_dropout=False):
            layers = [
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
            ]
            if use_pool:
                layers.append(nn.MaxPool2d(2))
            if use_dropout:
                layers.append(nn.Dropout(0.2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256, use_pool=False),
            conv_block(256, 512, use_dropout=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_flatten = 512

        self.fc = nn.Sequential(
            nn.Linear(self.n_flatten, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
