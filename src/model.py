import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from transformers import ASTModel


class ASTClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ASTAudioClassifier(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
        )

        self.backbone.encoder.gradient_checkpointing = True

        hidden_size = self.backbone.config.hidden_size

        self.head = ASTClassificationHead(hidden_size=hidden_size, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        outputs = self.backbone(input_values=x)
        logits = self.head(outputs.last_hidden_state)
        return logits


class AudioTrainingSystem(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        lr: float,  # noqa: ARG002
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        f1 = self.train_f1(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        f1 = self.val_f1(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-5,
        )

        milestones = [4, 7, 11]
        gamma = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return [optimizer], [scheduler]
