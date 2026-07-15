import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_results_dir(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir, experiment_name):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} - Loss Curves")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Training Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Accuracy Curves")
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
