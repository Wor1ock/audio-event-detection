import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import ASTForAudioClassification


class ASTClassificationModel(L.LightningModule):
    def __init__(self, num_classes: int, _lr: float = 1e-5, _weight_decay: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model = ASTForAudioClassification.from_pretrained(
            "/home/ext-yankin@ad.speechpro.com/Документы/labs/audio-event-detection/models/ast_base",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1(logits, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4, 6, 8], gamma=0.5)

        return [optimizer], [scheduler]
