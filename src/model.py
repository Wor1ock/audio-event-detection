import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models import resnet18


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

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
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

        self.model = resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes))

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        return self.model(x)
