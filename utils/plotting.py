import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_results_dir(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_precisions=None, val_precisions=None,
                         train_recalls=None, val_recalls=None, train_f1s=None, val_f1s=None, val_aurocs=None,
                         save_dir=None, experiment_name=None):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    n_plots = 4  # Loss, Accuracy, Precision/Recall/F1, AUROC

    plt.figure(figsize=(16, 10))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} - Loss Curves")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, label="Training Accuracy", marker='o')
    plt.plot(epochs, val_accs, label="Validation Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Accuracy Curves")
    plt.legend()
    plt.grid(True)

    # Precision, Recall, F1 plot
    plt.subplot(2, 2, 3)
    if train_precisions:
        plt.plot(epochs, train_precisions, label="Training Precision", marker='o', linestyle='--')
    if val_precisions:
        plt.plot(epochs, val_precisions, label="Validation Precision", marker='s', linestyle='--')
    if train_recalls:
        plt.plot(epochs, train_recalls, label="Training Recall", marker='^', linestyle='-.')
    if val_recalls:
        plt.plot(epochs, val_recalls, label="Validation Recall", marker='d', linestyle='-.')
    if train_f1s:
        plt.plot(epochs, train_f1s, label="Training F1", marker='*', linestyle=':')
    if val_f1s:
        plt.plot(epochs, val_f1s, label="Validation F1", marker='X', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{experiment_name} - Precision, Recall, F1")
    plt.legend()
    plt.grid(True)

    # AUROC plot
    plt.subplot(2, 2, 4)
    if val_aurocs:
        plt.plot(epochs, val_aurocs, label="Validation AUROC", marker='o', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title(f"{experiment_name} - AUROC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(confusion_matrix, save_dir, experiment_name, class_names=["Real", "Fake"]):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{experiment_name} - Confusion Matrix")
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_metrics_to_csv(metrics_dict, save_dir, experiment_name):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(save_dir, f"{experiment_name}_metrics.csv"), index=False)
    return df
