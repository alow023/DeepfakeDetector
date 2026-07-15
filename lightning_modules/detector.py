import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
import numpy as np

class DeepfakeDetector(pl.LightningModule):
    def __init__(self, model, lr=1e-4, num_classes=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.train_precision = Precision(task=task, num_classes=num_classes)
        self.val_precision = Precision(task=task, num_classes=num_classes)
        self.train_recall = Recall(task=task, num_classes=num_classes)
        self.val_recall = Recall(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task=task, num_classes=num_classes)
        self.val_auroc = AUROC(task=task, num_classes=num_classes)
        self.val_confusion = ConfusionMatrix(task=task, num_classes=num_classes)

        # To store epoch-wise metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_aurocs = []
        self.val_confusions = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.train_acc(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)
        self.training_step_outputs.append(loss)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        self.train_losses.append(avg_train_loss.item())
        self.training_step_outputs.clear()

        self.log("train_acc_epoch", self.train_acc.compute(), prog_bar=True)
        self.log("train_precision_epoch", self.train_precision.compute())
        self.log("train_recall_epoch", self.train_recall.compute())
        self.log("train_f1_epoch", self.train_f1.compute())

        self.train_accs.append(self.train_acc.compute().item())
        self.train_precisions.append(self.train_precision.compute().item())
        self.train_recalls.append(self.train_recall.compute().item())
        self.train_f1s.append(self.train_f1.compute().item())

        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)
        self.val_confusion(preds, y)
        self.validation_step_outputs.append(loss)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.val_losses.append(avg_val_loss.item())
        self.validation_step_outputs.clear()

        self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True)
        self.log("val_precision_epoch", self.val_precision.compute())
        self.log("val_recall_epoch", self.val_recall.compute())
        self.log("val_f1_epoch", self.val_f1.compute())
        self.log("val_auroc_epoch", self.val_auroc.compute())

        self.val_accs.append(self.val_acc.compute().item())
        self.val_precisions.append(self.val_precision.compute().item())
        self.val_recalls.append(self.val_recall.compute().item())
        self.val_f1s.append(self.val_f1.compute().item())
        self.val_aurocs.append(self.val_auroc.compute().item())
        self.val_confusions.append(self.val_confusion.compute().cpu().numpy())

        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_confusion.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
