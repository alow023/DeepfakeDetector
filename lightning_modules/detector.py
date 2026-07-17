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
        self.train_precisions = []  # NEW: Initialize train_precisions
        self.val_precisions = []
        self.train_recalls = []     # NEW: Initialize train_recalls
        self.val_recalls = []
        self.train_f1s = []         # NEW: Initialize train_f1s
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

        # Compute all metrics once
        train_acc_val = self.train_acc.compute().item()
        train_precision_val = self.train_precision.compute().item()
        train_recall_val = self.train_recall.compute().item()
        train_f1_val = self.train_f1.compute().item()

        # Log metrics
        self.log("train_acc_epoch", train_acc_val, prog_bar=True)
        self.log("train_precision_epoch", train_precision_val)
        self.log("train_recall_epoch", train_recall_val)
        self.log("train_f1_epoch", train_f1_val)

        # Save to history
        self.train_accs.append(train_acc_val)
        self.train_precisions.append(train_precision_val)
        self.train_recalls.append(train_recall_val)
        self.train_f1s.append(train_f1_val)

        # Reset all metrics
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

        # For binary AUROC, we need the positive class probability (index 1)
        pos_probs = probs[:, 1] if self.num_classes == 2 else probs

        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(pos_probs, y)
        self.val_confusion(preds, y)
        self.validation_step_outputs.append(loss)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.val_losses.append(avg_val_loss.item())
        self.validation_step_outputs.clear()

        # Compute all metrics once
        val_acc_val = self.val_acc.compute().item()
        val_precision_val = self.val_precision.compute().item()
        val_recall_val = self.val_recall.compute().item()
        val_f1_val = self.val_f1.compute().item()
        val_auroc_val = self.val_auroc.compute().item()
        val_confusion_val = self.val_confusion.compute().cpu().numpy()

        # Log metrics
        self.log("val_acc_epoch", val_acc_val, prog_bar=True)
        self.log("val_precision_epoch", val_precision_val)
        self.log("val_recall_epoch", val_recall_val)
        self.log("val_f1_epoch", val_f1_val)
        self.log("val_auroc_epoch", val_auroc_val)

        # Save to history
        self.val_accs.append(val_acc_val)
        self.val_precisions.append(val_precision_val)
        self.val_recalls.append(val_recall_val)
        self.val_f1s.append(val_f1_val)
        self.val_aurocs.append(val_auroc_val)
        self.val_confusions.append(val_confusion_val)

        # Reset all metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_confusion.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
