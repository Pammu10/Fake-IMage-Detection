import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


sns.set_style("whitegrid")


def ensure_output_dirs(base_dir: str = "outputs") -> Dict[str, str]:
    graphs_dir = os.path.join(base_dir, "graphs")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return {"base": base_dir, "graphs": graphs_dir, "models": models_dir}


def save_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, save_path: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return float(roc_auc)


def save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, save_path: str) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return float(pr_auc)


def save_confidence_distribution(
    y_true: np.ndarray, y_score_fake: np.ndarray, class_names: List[str], save_path: str
) -> None:
    fig = plt.figure(figsize=(8, 5))
    # y_score_fake is probability for class index 1 in our training pipeline.
    # We convert class name label text accordingly so the chart stays readable.
    sns.histplot(
        y_score_fake[y_true == 0],
        bins=30,
        color="#f39c12",
        alpha=0.6,
        label=f"True {class_names[0]}",
        stat="density",
    )
    sns.histplot(
        y_score_fake[y_true == 1],
        bins=30,
        color="#3498db",
        alpha=0.6,
        label=f"True {class_names[1]}",
        stat="density",
    )
    plt.xlabel(f"Predicted confidence for class: {class_names[1]}")
    plt.ylabel("Density")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table_as_image(df: pd.DataFrame, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.4), max(2.5, len(df) * 0.5)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_and_save_artifacts(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    graphs_dir: str,
) -> Dict:
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = float(accuracy_score(y_true, y_pred))

    try:
        auc_score = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        auc_score = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, os.path.join(graphs_dir, "confusion_matrix.png"))

    roc_auc = save_roc_curve(
        y_true,
        y_prob[:, 1],
        os.path.join(graphs_dir, "roc_curve.png"),
    )
    pr_auc = save_pr_curve(
        y_true,
        y_prob[:, 1],
        os.path.join(graphs_dir, "precision_recall_curve.png"),
    )
    save_confidence_distribution(
        y_true,
        y_prob[:, 1],
        class_names,
        os.path.join(graphs_dir, "confidence_distribution.png"),
    )

    return {
        "accuracy": accuracy,
        "auc": auc_score,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_from_predictions_and_save_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    graphs_dir: str,
) -> Dict:
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))

    try:
        auc_score = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        auc_score = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, os.path.join(graphs_dir, "confusion_matrix.png"))

    roc_auc = save_roc_curve(
        y_true,
        y_prob[:, 1],
        os.path.join(graphs_dir, "roc_curve.png"),
    )
    pr_auc = save_pr_curve(
        y_true,
        y_prob[:, 1],
        os.path.join(graphs_dir, "precision_recall_curve.png"),
    )
    save_confidence_distribution(
        y_true,
        y_prob[:, 1],
        class_names,
        os.path.join(graphs_dir, "confidence_distribution.png"),
    )

    return {
        "accuracy": accuracy,
        "auc": auc_score,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def write_results_json(results: Dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
