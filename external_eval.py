import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel

MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = os.path.join("outputs", "models", "best_overall.pt")
RESULTS_PATH = os.path.join("outputs", "results.json")
OUTPUT_PATH = os.path.join("outputs", "external_test_results.json")


class ImgDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        return self.transform(x), y


class ViTFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(MODEL_NAME)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)


def softmax_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
    s = logits / max(temperature, 1e-6)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=1, keepdims=True)


def collect_samples(root: str, class_names: List[str]) -> List[Tuple[str, int]]:
    idx_map = {c.lower(): i for i, c in enumerate(class_names)}
    fake_idx = idx_map.get("fake", idx_map.get("ai", 0))
    real_idx = idx_map.get("real", 1)
    aliases = {
        "fake": fake_idx,
        "ai": fake_idx,
        "generated": fake_idx,
        "synthetic": fake_idx,
        "real": real_idx,
        "authentic": real_idx,
    }
    out = []
    for child in os.listdir(root):
        d = os.path.join(root, child)
        if not os.path.isdir(d):
            continue
        key = child.lower()
        if key in idx_map:
            y = idx_map[key]
        elif key in aliases:
            y = aliases[key]
        elif any(tok in key for tok in ["fake", "generated", "synthetic"]):
            y = fake_idx
        elif any(tok in key for tok in ["real", "authentic"]):
            y = real_idx
        else:
            continue
        for r, _, files in os.walk(d):
            for f in files:
                lf = f.lower()
                if lf.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    out.append((os.path.join(r, f), y))
    return out


def main(external_root: str):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("best_overall.pt not found. Run training first.")
    if not os.path.isdir(external_root):
        raise FileNotFoundError(f"External root not found: {external_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = ckpt.get("class_names", ["FAKE", "REAL"])

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    size = processor.size["height"] if isinstance(processor.size, dict) else 224
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    samples = collect_samples(external_root, class_names)
    if not samples:
        raise ValueError("No class-labeled images found in external root.")

    ds = ImgDataset(samples, tf)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = ViTFakeDetector().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calibration = {"temperature": 1.0, "fake_threshold": 0.5, "fake_class_index": 0}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            r = json.load(f)
        calibration.update(r.get("calibration", {}))

    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = softmax_temp(logits, float(calibration["temperature"]))

    fake_idx = int(calibration.get("fake_class_index", 0))
    fake_threshold = float(calibration.get("fake_threshold", 0.5))
    y_pred_fake = (probs[:, fake_idx] >= fake_threshold).astype(int)
    y_pred = np.where(y_pred_fake == 1, fake_idx, 1 - fake_idx)

    y_true_fake = (y_true == fake_idx).astype(int)
    metrics = {
        "external_root": external_root,
        "num_samples": int(len(samples)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true_fake, probs[:, fake_idx])),
        "fake_threshold": fake_threshold,
        "temperature": float(calibration["temperature"]),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("External evaluation complete")
    print(json.dumps({k: metrics[k] for k in ["num_samples", "accuracy", "auc"]}, indent=2))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    root = os.path.join("data", "external", "ciplab_real_fake_face", "real_and_fake_face")
    main(root)
