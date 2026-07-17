import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

# Fix Windows unicode encoding issue
os.environ["PYTHONIOENCODING"] = "utf-8"

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector
from models import EfficientNetWithCBAM
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import create_results_dir, plot_training_curves, plot_confusion_matrix, save_metrics_to_csv


if __name__ == "__main__":
    # === Load YAML config ===
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # === Dataset Paths ===
    train_sources = [(p, None) for p in cfg["train_paths"]]
    val_sources = [(p, None) for p in cfg["val_paths"]]

    # === Datasets & Loaders ===
    train_dataset = HybridDeepfakeDataset(train_sources, transform=transform)
    val_dataset = HybridDeepfakeDataset(val_sources, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # === Model Architecture ===
    use_cbam = cfg.get("use_cbam", True)
    experiment_name = "efficientnet_b0_cbam" if use_cbam else "baseline_efficientnet_b0"
    backbone = EfficientNetWithCBAM(use_cbam=use_cbam)

    # Print model summary
    print("=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    summary(backbone, input_size=(1, 3, 224, 224), device="cpu")
    print("=" * 60)

    model = DeepfakeDetector(backbone, lr=cfg["lr"])

    # === Callbacks ===
    checkpoint = ModelCheckpoint(
        monitor=cfg.get("monitor_metric", "val_loss"),
        dirpath="models",
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val_loss"),
        patience=3,
        mode="min"
    )

    # === Check Debug Mode ===
    debug_mode = cfg.get("DEBUG_MODE", False)
    if debug_mode:
        print("="*42)
        print("DEBUG MODE ENABLED")
        print("Training on 10 batches")
        print("Validation on 10 batches")
        print("Epochs: 1")
        print("="*42)
        max_epochs = 1
        limit_train_batches = 10
        limit_val_batches = 10
    else:
        max_epochs = cfg["num_epochs"]
        limit_train_batches = 1.0
        limit_val_batches = 1.0

    # === Trainer ===
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint, early_stop],
        enable_progress_bar=True,  # Disable progress bar to avoid Windows encoding issues
        log_every_n_steps=cfg.get("log_every_n_steps", 1),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_model_summary=False,  # We already printed our own model summary
        num_sanity_val_steps=0  # Skip sanity check to go faster in debug
    )

    # === Start Training ===
    print("=== Starting trainer.fit() ===")
    trainer.fit(model, train_loader, val_loader)
    print("=== trainer.fit() completed ===")

    # === Save Results ===
    print("=== Starting to save results ===")
    save_dir = os.path.join("results", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Print the collected metrics lengths
    print("=== Collected metrics ===")
    print(f"train_losses: {len(model.train_losses)}")
    print(f"val_losses: {len(model.val_losses)}")
    print(f"train_accs: {len(model.train_accs)}")
    print(f"val_accs: {len(model.val_accs)}")
    print(f"train_precisions: {len(model.train_precisions)}")
    print(f"val_precisions: {len(model.val_precisions)}")
    print(f"train_recalls: {len(model.train_recalls)}")
    print(f"val_recalls: {len(model.val_recalls)}")
    print(f"train_f1s: {len(model.train_f1s)}")
    print(f"val_f1s: {len(model.val_f1s)}")
    print(f"val_aurocs: {len(model.val_aurocs)}")

    metrics_dict = {
        "epoch": list(range(1, len(model.val_losses) + 1)),
        "train_loss": model.train_losses,
        "val_loss": model.val_losses,
        "train_acc": model.train_accs,
        "val_acc": model.val_accs,
        "train_precision": model.train_precisions,
        "val_precision": model.val_precisions,
        "train_recall": model.train_recalls,
        "val_recall": model.val_recalls,
        "train_f1": model.train_f1s,
        "val_f1": model.val_f1s,
        "val_auroc": model.val_aurocs
    }

    save_metrics_to_csv(metrics_dict, save_dir, experiment_name)
    print(f"CSV saved to {os.path.join(save_dir, f'{experiment_name}_metrics.csv')}")
    
    plot_training_curves(
        model.train_losses, model.val_losses,
        model.train_accs, model.val_accs,
        model.train_precisions, model.val_precisions,
        model.train_recalls, model.val_recalls,
        model.train_f1s, model.val_f1s,
        model.val_aurocs,
        save_dir, experiment_name
    )
    print(f"Training curves saved to {os.path.join(save_dir, f'{experiment_name}_training_curves.png')}")
    
    if model.val_confusions:
        plot_confusion_matrix(model.val_confusions[-1], save_dir, experiment_name)
        print(f"Confusion matrix saved to {os.path.join(save_dir, f'{experiment_name}_confusion_matrix.png')}")
    print("=== All results saved ===")
