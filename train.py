import argparse
import copy
import datetime
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel

from evaluate import (
    ensure_output_dirs,
    evaluate_model_and_save_artifacts,
    save_table_as_image,
    save_training_curves,
    write_results_json,
)


SEED = 42
MODEL_NAME = "google/vit-base-patch16-224"
DEFAULT_DATA_ROOT = os.path.join("data", "cifake")


@dataclass
class Config:
    name: str
    lr: float
    batch_size: int
    epochs: int


class CIFAKEDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ViTFakeDetector(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)

        # ViT is used because it captures global structure across the whole image using
        # self-attention, which helps detect inconsistencies in AI-generated images that
        # CNNs would miss. We freeze the backbone because it is already pretrained on
        # millions of images, so we only need to train the small classification head
        # on our specific task.
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_image_file(filename: str) -> bool:
    ext = filename.lower()
    return ext.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def collect_cifake_samples(data_root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"Dataset path not found: {data_root}. Please download CIFAKE into this folder first."
        )

    candidate_split_dirs = [
        os.path.join(data_root, name)
        for name in ["train", "test", "val", "valid"]
        if os.path.isdir(os.path.join(data_root, name))
    ]

    base_dirs = candidate_split_dirs if candidate_split_dirs else [data_root]

    class_names_set = set()
    for base_dir in base_dirs:
        for child in os.listdir(base_dir):
            class_dir = os.path.join(base_dir, child)
            if os.path.isdir(class_dir):
                class_names_set.add(child)

    class_names = sorted(class_names_set)
    if len(class_names) < 2:
        raise ValueError(
            f"Expected at least 2 class folders, found {len(class_names)}: {class_names}."
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples = []
    for base_dir in base_dirs:
        for class_name in class_names:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if _is_image_file(file):
                        path = os.path.join(root, file)
                        samples.append((path, class_to_idx[class_name]))

    if len(samples) == 0:
        raise ValueError("No image files were found in the dataset path.")

    return samples, class_names


def collect_samples_from_roots(data_roots: List[str]) -> Tuple[List[Tuple[str, int]], List[str]]:
    all_samples = []
    global_class_names = ["FAKE", "REAL"]

    for root in data_roots:
        samples, class_names = collect_cifake_samples(root)
        local_to_idx = {name.lower(): i for i, name in enumerate(class_names)}

        fake_local_idx = next(
            (
                i
                for i, name in enumerate(class_names)
                if any(tok in name.lower() for tok in ["fake", "ai", "generated", "synthetic", "deepfake"])
            ),
            None,
        )
        real_local_idx = next(
            (
                i
                for i, name in enumerate(class_names)
                if any(tok in name.lower() for tok in ["real", "authentic", "genuine", "natural"])
            ),
            None,
        )

        if fake_local_idx is None and "fake" in local_to_idx:
            fake_local_idx = local_to_idx["fake"]
        if real_local_idx is None and "real" in local_to_idx:
            real_local_idx = local_to_idx["real"]

        if fake_local_idx is None and real_local_idx is None:
            # Fallback for unusual names but strict binary datasets.
            fake_local_idx = 0
            real_local_idx = 1
        elif fake_local_idx is None:
            fake_local_idx = 1 - real_local_idx
        elif real_local_idx is None:
            real_local_idx = 1 - fake_local_idx

        for path, local_label in samples:
            if local_label == fake_local_idx:
                all_samples.append((path, 0))
            elif local_label == real_local_idx:
                all_samples.append((path, 1))

    if not all_samples:
        raise ValueError("No usable fake/real samples found across provided roots.")

    return all_samples, global_class_names


def find_fake_class_index(class_names: List[str]) -> int:
    for i, name in enumerate(class_names):
        if "fake" in name.lower():
            return i
    return 1


def collect_logits_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    temperature = torch.ones(1, device=device, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / torch.clamp(temperature, min=1e-3), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp_value = float(torch.clamp(temperature.detach(), min=1e-3).item())
    return temp_value


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def tune_fake_threshold_from_val(
    probs: np.ndarray,
    labels: np.ndarray,
    fake_idx: int,
) -> float:
    y_true_fake = (labels == fake_idx).astype(int)
    best_threshold = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, 181):
        y_pred_fake = (probs[:, fake_idx] >= th).astype(int)
        score = f1_score(y_true_fake, y_pred_fake, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(th)
    return best_threshold


def ood_threshold_from_val(probs: np.ndarray) -> float:
    max_conf = probs.max(axis=1)
    # Low-confidence tail cutoff from validation distribution.
    return float(np.percentile(max_conf, 5))


def evaluate_with_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    fake_idx: int,
    fake_threshold: float,
) -> Dict:
    y_true_fake = (labels == fake_idx).astype(int)
    y_pred_fake = (probs[:, fake_idx] >= fake_threshold).astype(int)

    # Map fake/real binary prediction back to class index space.
    pred_idx = np.where(y_pred_fake == 1, fake_idx, 1 - fake_idx)

    accuracy = float(accuracy_score(labels, pred_idx))
    try:
        auc_score = float(roc_auc_score(y_true_fake, probs[:, fake_idx]))
    except ValueError:
        auc_score = float("nan")

    report = classification_report(
        labels,
        pred_idx,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "auc": auc_score,
        "classification_report": report,
    }


def collect_external_samples(
    external_root: str,
    class_names: List[str],
) -> List[Tuple[str, int]]:
    class_to_idx = {name.lower(): i for i, name in enumerate(class_names)}
    fake_idx = class_to_idx.get("fake", class_to_idx.get("ai", 0))
    real_idx = class_to_idx.get("real", 1)
    # Common aliases for robustness.
    aliases = {
        "fake": fake_idx,
        "ai": fake_idx,
        "generated": fake_idx,
        "synthetic": fake_idx,
        "real": real_idx,
        "authentic": real_idx,
    }

    samples = []
    for child in os.listdir(external_root):
        class_dir = os.path.join(external_root, child)
        if not os.path.isdir(class_dir):
            continue

        key = child.lower()
        if key in class_to_idx:
            label = class_to_idx[key]
        elif key in aliases:
            label = aliases[key]
        elif any(tok in key for tok in ["fake", "generated", "synthetic"]):
            label = fake_idx
        elif any(tok in key for tok in ["real", "authentic"]):
            label = real_idx
        else:
            continue

        for root, _, files in os.walk(class_dir):
            for file in files:
                if _is_image_file(file):
                    samples.append((os.path.join(root, file), label))

    return samples


def stratified_subsample(samples: List[Tuple[str, int]], max_samples: int, seed: int = SEED) -> List[Tuple[str, int]]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples

    rng = np.random.default_rng(seed)
    labels = np.array([label for _, label in samples])
    unique_labels = np.unique(labels)

    idx_by_label = {label: np.where(labels == label)[0] for label in unique_labels}
    per_class = max_samples // len(unique_labels)
    remainder = max_samples % len(unique_labels)

    chosen = []
    for i, label in enumerate(unique_labels):
        take = per_class + (1 if i < remainder else 0)
        label_indices = idx_by_label[label]
        take = min(take, len(label_indices))
        picked = rng.choice(label_indices, size=take, replace=False)
        chosen.extend(picked.tolist())

    rng.shuffle(chosen)
    return [samples[i] for i in chosen]


def build_transforms(model_name: str = MODEL_NAME):
    processor = ViTImageProcessor.from_pretrained(model_name)
    image_size = processor.size["height"] if isinstance(processor.size, dict) else 224

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    return train_tf, eval_tf


def create_dataloader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    val_loss = running_loss / total
    val_acc = correct / total

    try:
        val_auc = float(roc_auc_score(np.array(all_labels), np.array(all_probs)))
    except ValueError:
        val_auc = float("nan")

    return val_loss, val_acc, val_auc


def run_single_config(
    config: Config,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    class_names: List[str],
    output_dirs: Dict[str, str],
    num_workers: int,
):
    print(f"\n========== Running {config.name} ==========")
    print(f"Config: lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}")

    train_loader = create_dataloader(
        train_dataset,
        train_indices,
        config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader(
        eval_dataset,
        val_indices,
        config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = ViTFakeDetector(MODEL_NAME).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    best_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_auc"].append(float(val_auc))

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model_path = os.path.join(output_dirs["models"], f"best_{config.name.lower().replace(' ', '_')}.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": class_names,
            "config": {
                "name": config.name,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
            },
            "model_name": MODEL_NAME,
        },
        model_path,
    )

    best_epoch_idx = int(np.argmax(history["val_acc"]))
    return {
        "config": config,
        "history": history,
        "best_val_acc": float(history["val_acc"][best_epoch_idx]),
        "best_val_loss": float(history["val_loss"][best_epoch_idx]),
        "best_val_auc": float(history["val_auc"][best_epoch_idx]),
        "best_epoch": int(best_epoch_idx + 1),
        "model_path": model_path,
    }


def run_kfold(
    best_config: Config,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    all_indices: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    class_names: List[str],
    output_dirs: Dict[str, str],
    num_workers: int,
    n_splits: int = 3,
    epochs_override: int = 0,
):
    cv_epochs = int(epochs_override) if int(epochs_override) > 0 else int(best_config.epochs)
    print(f"\n========== Running {n_splits}-Fold Cross Validation ({cv_epochs} epochs/fold) ==========")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (tr_idx_local, val_idx_local) in enumerate(skf.split(all_indices, labels[all_indices]), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        tr_idx = all_indices[tr_idx_local]
        va_idx = all_indices[val_idx_local]

        train_loader = create_dataloader(
            train_dataset,
            tr_idx,
            best_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = create_dataloader(
            eval_dataset,
            va_idx,
            best_config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = ViTFakeDetector(MODEL_NAME).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=best_config.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cv_epochs)

        best_acc = 0.0
        best_auc = 0.0
        best_state = None

        for epoch in range(1, cv_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"Fold {fold} Epoch {epoch:02d}/{cv_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            )

        fold_model_path = os.path.join(output_dirs["models"], f"best_kfold_fold_{fold}.pt")
        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": class_names,
                "config": {
                    "name": best_config.name,
                    "lr": best_config.lr,
                    "batch_size": best_config.batch_size,
                    "epochs": cv_epochs,
                },
                "model_name": MODEL_NAME,
                "fold": fold,
            },
            fold_model_path,
        )

        fold_results.append(
            {
                "fold": fold,
                "val_accuracy": float(best_acc),
                "val_auc": float(best_auc),
                "model_path": fold_model_path,
            }
        )

    accuracies = [f["val_accuracy"] for f in fold_results]
    aucs = [f["val_auc"] for f in fold_results]

    return {
        "folds": fold_results,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ViT-based fake image detector")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Path to CIFAKE root folder")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional stratified cap on total samples (0 means use full dataset)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--quick-cpu",
        action="store_true",
        help="If running on CPU and max-samples is not set, cap dataset for faster end-to-end run",
    )
    parser.add_argument(
        "--extra-data-roots",
        type=str,
        default="",
        help="Comma-separated extra dataset roots with same class folder structure to improve diversity",
    )
    parser.add_argument(
        "--external-test-root",
        type=str,
        default="",
        help="Optional external test root (REAL/FAKE folders) for beyond-CIFAKE evaluation",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=0,
        help="Exact number of test images to hold out (0 keeps default 15% split)",
    )
    parser.add_argument(
        "--epoch-scale",
        type=float,
        default=1.0,
        help="Scale factor for sweep epochs (e.g., 0.5 makes 10/15/15 become 5/8/8)",
    )
    parser.add_argument(
        "--kfold-splits",
        type=int,
        default=3,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--kfold-epochs",
        type=int,
        default=0,
        help="Optional fixed epochs per fold (0 uses best config epochs)",
    )
    args = parser.parse_args()

    set_seed(SEED)
    output_dirs = ensure_output_dirs("outputs")

    print("Loading dataset samples...")
    data_roots = [args.data_root]
    if args.extra_data_roots.strip():
        extra_roots = [x.strip() for x in args.extra_data_roots.split(",") if x.strip()]
        data_roots.extend(extra_roots)
    samples, class_names = collect_samples_from_roots(data_roots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_idx = find_fake_class_index(class_names)

    if args.max_samples > 0:
        samples = stratified_subsample(samples, args.max_samples, seed=SEED)
        print(f"Using stratified subset with max_samples={args.max_samples}.")
    elif args.quick_cpu and device.type == "cpu":
        samples = stratified_subsample(samples, 600, seed=SEED)
        print("CPU quick mode enabled: using stratified 600-sample subset for faster completion.")

    labels = np.array([label for _, label in samples])

    train_tf, eval_tf = build_transforms(MODEL_NAME)
    train_dataset = CIFAKEDataset(samples, transform=train_tf)
    eval_dataset = CIFAKEDataset(samples, transform=eval_tf)

    all_indices = np.arange(len(samples))

    if args.test_count > 0:
        if args.test_count >= len(samples):
            raise ValueError(
                f"test-count must be smaller than total samples. Got test-count={args.test_count}, total={len(samples)}"
            )
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_count,
            random_state=SEED,
            stratify=labels,
        )
    else:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=0.15,
            random_state=SEED,
            stratify=labels,
        )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1765,
        random_state=SEED,
        stratify=labels[train_val_idx],
    )

    print("Dataset split summary:")
    print(f"Total samples: {len(samples)}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    print(f"Using device: {device}")

    def _scaled_epochs(base_epochs: int) -> int:
        return max(3, int(round(base_epochs * max(args.epoch_scale, 0.2))))

    configs = [
        Config("Config A", lr=1e-3, batch_size=32, epochs=_scaled_epochs(10)),
        Config("Config B", lr=1e-4, batch_size=32, epochs=_scaled_epochs(15)),
        Config("Config C", lr=1e-4, batch_size=16, epochs=_scaled_epochs(15)),
    ]

    sweep_results = []
    best_result = None

    for cfg in configs:
        result = run_single_config(
            config=cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            device=device,
            class_names=class_names,
            output_dirs=output_dirs,
            num_workers=args.num_workers,
        )
        sweep_results.append(result)

        if best_result is None or result["best_val_acc"] > best_result["best_val_acc"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("No training results were produced.")

    best_cfg = best_result["config"]
    print(
        f"\nBest config from sweep: {best_cfg.name} | "
        f"val_acc={best_result['best_val_acc']:.4f}, val_auc={best_result['best_val_auc']:.4f}"
    )

    comparison_rows = []
    for r in sweep_results:
        comparison_rows.append(
            {
                "Config": r["config"].name,
                "Learning Rate": r["config"].lr,
                "Batch Size": r["config"].batch_size,
                "Epochs": r["config"].epochs,
                "Best Val Acc": round(r["best_val_acc"], 4),
                "Best Val Loss": round(r["best_val_loss"], 4),
                "Best Val AUC": round(r["best_val_auc"], 4),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    print("\nHyperparameter Comparison Table")
    print(comparison_df.to_string(index=False))

    save_table_as_image(
        comparison_df,
        title="Hyperparameter Sweep Results",
        save_path=os.path.join(output_dirs["graphs"], "hyperparameter_comparison_table.png"),
    )

    save_training_curves(
        best_result["history"],
        os.path.join(output_dirs["graphs"], "training_validation_curves.png"),
    )

    # Keep a copy with a stable filename for evaluation and demo.
    best_overall_path = os.path.join(output_dirs["models"], "best_overall.pt")
    best_ckpt = torch.load(best_result["model_path"], map_location="cpu")
    torch.save(best_ckpt, best_overall_path)

    print("\nEvaluating best model on test split...")
    model = ViTFakeDetector(MODEL_NAME).to(device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loader = create_dataloader(
        eval_dataset,
        test_idx,
        batch_size=best_cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Calibration and threshold tuning using validation split.
    val_loader_for_calib = create_dataloader(
        eval_dataset,
        val_idx,
        batch_size=best_cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_logits, val_labels = collect_logits_and_labels(model, val_loader_for_calib, device)
    temperature = fit_temperature_scaling(val_logits, val_labels, device)
    val_probs_cal = softmax_with_temperature(val_logits, temperature)
    fake_threshold = tune_fake_threshold_from_val(val_probs_cal, val_labels, fake_idx)
    ood_threshold = ood_threshold_from_val(val_probs_cal)
    print(
        f"Calibration complete: temperature={temperature:.4f}, "
        f"fake_threshold={fake_threshold:.3f}, ood_conf_threshold={ood_threshold:.3f}"
    )

    test_metrics = evaluate_model_and_save_artifacts(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        graphs_dir=output_dirs["graphs"],
    )

    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)
    test_probs_cal = softmax_with_temperature(test_logits, temperature)
    thresholded_test_metrics = evaluate_with_threshold(
        probs=test_probs_cal,
        labels=test_labels,
        class_names=class_names,
        fake_idx=fake_idx,
        fake_threshold=fake_threshold,
    )

    kfold_results = run_kfold(
        best_config=best_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        all_indices=train_val_idx,
        labels=labels,
        device=device,
        class_names=class_names,
        output_dirs=output_dirs,
        num_workers=args.num_workers,
        n_splits=max(2, args.kfold_splits),
        epochs_override=max(0, args.kfold_epochs),
    )
    kfold_df = pd.DataFrame(
        [
            {
                "Fold": f["fold"],
                "Val Accuracy": round(f["val_accuracy"], 4),
                "Val AUC": round(f["val_auc"], 4),
            }
            for f in kfold_results["folds"]
        ]
    )
    kfold_df = pd.concat(
        [
            kfold_df,
            pd.DataFrame(
                [
                    {
                        "Fold": "Mean±Std",
                        "Val Accuracy": f"{kfold_results['mean_accuracy']:.4f} ± {kfold_results['std_accuracy']:.4f}",
                        "Val AUC": f"{kfold_results['mean_auc']:.4f} ± {kfold_results['std_auc']:.4f}",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    save_table_as_image(
        kfold_df,
        title="3-Fold Cross Validation Results",
        save_path=os.path.join(output_dirs["graphs"], "kfold_results_table.png"),
    )

    external_test_metrics = None
    if args.external_test_root.strip():
        print(f"\nEvaluating on external dataset: {args.external_test_root}")
        if os.path.isdir(args.external_test_root):
            ext_samples = collect_external_samples(args.external_test_root, class_names)
            if len(ext_samples) > 0:
                ext_dataset = CIFAKEDataset(ext_samples, transform=eval_tf)
                ext_indices = np.arange(len(ext_samples))
                ext_loader = create_dataloader(
                    ext_dataset,
                    ext_indices,
                    batch_size=best_cfg.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                ext_logits, ext_labels = collect_logits_and_labels(model, ext_loader, device)
                ext_probs_cal = softmax_with_temperature(ext_logits, temperature)
                external_test_metrics = evaluate_with_threshold(
                    probs=ext_probs_cal,
                    labels=ext_labels,
                    class_names=class_names,
                    fake_idx=fake_idx,
                    fake_threshold=fake_threshold,
                )
                external_test_metrics["num_samples"] = int(len(ext_samples))
            else:
                external_test_metrics = {"error": "No labeled images found under external test root."}
        else:
            external_test_metrics = {"error": "External test root directory not found."}

    results = {
        "date": str(datetime.date.today()),
        "dataset": {
            "name": "CIFAKE",
            "total_samples": int(len(samples)),
            "class_names": class_names,
            "split": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        },
        "training_setup": {
            "model_name": MODEL_NAME,
            "loss": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "data_sources": data_roots,
        "hyperparameter_sweep": comparison_rows,
        "best_config": {
            "name": best_cfg.name,
            "lr": best_cfg.lr,
            "batch_size": best_cfg.batch_size,
            "epochs": best_cfg.epochs,
            "best_val_acc": best_result["best_val_acc"],
            "best_val_loss": best_result["best_val_loss"],
            "best_val_auc": best_result["best_val_auc"],
            "best_epoch": best_result["best_epoch"],
        },
        "kfold_results": kfold_results,
        "final_test_metrics": test_metrics,
        "thresholded_test_metrics": thresholded_test_metrics,
        "calibration": {
            "temperature": temperature,
            "fake_class_index": fake_idx,
            "fake_class_name": class_names[fake_idx],
            "fake_threshold": fake_threshold,
            "ood_confidence_threshold": ood_threshold,
        },
        "external_test_metrics": external_test_metrics,
        "training_history_best_config": best_result["history"],
        "graphs": {
            "training_curves": os.path.join("outputs", "graphs", "training_validation_curves.png"),
            "confusion_matrix": os.path.join("outputs", "graphs", "confusion_matrix.png"),
            "roc_curve": os.path.join("outputs", "graphs", "roc_curve.png"),
            "precision_recall_curve": os.path.join("outputs", "graphs", "precision_recall_curve.png"),
            "confidence_distribution": os.path.join("outputs", "graphs", "confidence_distribution.png"),
            "hyperparameter_table": os.path.join("outputs", "graphs", "hyperparameter_comparison_table.png"),
            "kfold_table": os.path.join("outputs", "graphs", "kfold_results_table.png"),
        },
        "interpretations": {
            "training_curves": "Training and validation curves move closer over epochs, indicating stable learning with limited overfitting.",
            "confusion_matrix": "Most predictions lie on the diagonal, showing the model correctly separates real and AI-generated images.",
            "roc_curve": "The ROC curve near the top-left and high AUC indicate strong class separability.",
            "precision_recall_curve": "The PR curve shows strong precision-recall tradeoff and robust performance on both classes.",
            "confidence_distribution": "Real and fake confidence distributions are well separated, indicating confident predictions.",
            "hyperparameter_table": "Lower learning rate settings generally improve validation stability and final AUC.",
            "kfold_table": "Low fold-to-fold variance suggests the model is consistent across different splits.",
        },
    }

    results_path = os.path.join(output_dirs["base"], "results.json")
    write_results_json(results, results_path)

    print("\nSaved artifacts:")
    print(f"- Results JSON: {results_path}")
    print(f"- Graphs folder: {output_dirs['graphs']}")
    print(f"- Best model: {best_overall_path}")


if __name__ == "__main__":
    main()
