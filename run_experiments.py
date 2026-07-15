import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector
from models import EfficientNetWithCBAM
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import create_results_dir, plot_training_curves, plot_confusion_matrix, save_metrics_to_csv


def run_experiment(experiment_name, use_cbam, cfg):
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}\n")

    # Create save directory
    save_dir = os.path.join("results", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Dataset Paths
    train_sources = [(p, None) for p in cfg["train_paths"]]
    val_sources = [(p, None) for p in cfg["val_paths"]]

    # Datasets & Loaders
    train_dataset = HybridDeepfakeDataset(train_sources, transform=transform)
    val_dataset = HybridDeepfakeDataset(val_sources, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # Model Architecture
    backbone = EfficientNetWithCBAM(use_cbam=use_cbam)

    # Print model summary
    print(f"\nModel Summary for {experiment_name}:")
    print("-"*60)
    summary(backbone, input_size=(1, 3, 224, 224), device="cpu")
    print("-"*60)

    model = DeepfakeDetector(backbone, lr=cfg["lr"])

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor=cfg.get("monitor_metric", "val_loss"),
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val_loss"),
        patience=3,
        mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint, early_stop],
        enable_progress_bar=True,
        log_every_n_steps=cfg.get("log_every_n_steps", 1),
        logger=False
    )

    # Start Training
    trainer.fit(model, train_loader, val_loader)

    # Extract metrics from the model
    metrics_dict = {
        "epoch": list(range(1, len(model.val_losses) + 1)),
        "train_loss": model.train_losses,
        "val_loss": model.val_losses,
        "train_acc": model.train_accs,
        "val_acc": model.val_accs,
        "val_precision": model.val_precisions,
        "val_recall": model.val_recalls,
        "val_f1": model.val_f1s,
        "val_auroc": model.val_aurocs
    }

    # Save results
    save_metrics_to_csv(metrics_dict, save_dir, experiment_name)
    plot_training_curves(
        model.train_losses, model.val_losses,
        model.train_accs, model.val_accs,
        save_dir, experiment_name
    )
    if model.val_confusions:
        plot_confusion_matrix(model.val_confusions[-1], save_dir, experiment_name)

    print(f"\n{'='*60}")
    print(f"Experiment {experiment_name} completed!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")

    return metrics_dict


if __name__ == "__main__":
    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Run both experiments
    baseline_metrics = run_experiment("baseline_efficientnet_b0", use_cbam=False, cfg=cfg)
    cbam_metrics = run_experiment("efficientnet_b0_cbam", use_cbam=True, cfg=cfg)

    # Create comparison CSV
    print("\nCreating comparison table...")
    comparison_data = []
    for exp_name, metrics in [("baseline", baseline_metrics), ("cbam", cbam_metrics)]:
        best_idx = metrics["val_acc"].index(max(metrics["val_acc"]))
        comparison_data.append({
            "experiment": exp_name,
            "best_epoch": best_idx + 1,
            "best_val_acc": metrics["val_acc"][best_idx],
            "best_val_precision": metrics["val_precision"][best_idx],
            "best_val_recall": metrics["val_recall"][best_idx],
            "best_val_f1": metrics["val_f1"][best_idx],
            "best_val_auroc": metrics["val_auroc"][best_idx]
        })

    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join("results", "experiment_comparison.csv"), index=False)
    print("\nComparison table saved to results/experiment_comparison.csv")
    print("\nComparison Results:")
    print(comparison_df)
