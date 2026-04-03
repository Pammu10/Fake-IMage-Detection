import os
import json

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel


MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = os.path.join("outputs", "models", "best_overall.pt")
RESULTS_PATH = os.path.join("outputs", "results.json")


class ViTFakeDetector(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
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


def load_model_and_transforms():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Please run train.py first."
        )

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    image_size = processor.size["height"] if isinstance(processor.size, dict) else 224

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = ViTFakeDetector(MODEL_NAME).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", ["FAKE", "REAL"])
    return model, transform, class_names, device


MODEL, TRANSFORM, CLASS_NAMES, DEVICE = load_model_and_transforms()


def load_results_summary():
    if not os.path.exists(RESULTS_PATH):
        return "results.json not found. Run train.py first.", []

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best_config", {})
    final = data.get("final_test_metrics", {})
    kfold = data.get("kfold_results", {})

    summary = (
        f"Best Config: {best.get('name', 'N/A')}\n"
        f"Val Acc: {best.get('best_val_acc', 'N/A')} | Val AUC: {best.get('best_val_auc', 'N/A')}\n"
        f"Test Acc: {final.get('accuracy', 'N/A')} | Test AUC: {final.get('auc', 'N/A')}\n"
        f"K-Fold Mean Acc: {kfold.get('mean_accuracy', 'N/A')} ± {kfold.get('std_accuracy', 'N/A')}\n"
        f"K-Fold Mean AUC: {kfold.get('mean_auc', 'N/A')} ± {kfold.get('std_auc', 'N/A')}"
    )

    graphs = data.get("graphs", {})
    preferred_order = [
        "training_curves",
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "confidence_distribution",
        "hyperparameter_table",
        "kfold_table",
    ]

    image_paths = []
    for key in preferred_order:
        p = graphs.get(key)
        if p and os.path.exists(p):
            image_paths.append(p)

    return summary, image_paths


def predict_image(image: Image.Image):
    if image is None:
        return "Please upload an image.", {}

    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    pred_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    friendly_label = "AI-Generated" if "fake" in pred_label.lower() else "Real"
    summary = f"Prediction: {friendly_label} | Confidence: {confidence:.2%}"

    class_probabilities = {
        ("AI-Generated" if "fake" in name.lower() else "Real"): float(prob)
        for name, prob in zip(CLASS_NAMES, probs)
    }

    return summary, class_probabilities


def main():
    summary_text, graph_images = load_results_summary()

    with gr.Blocks(title="AI vs Real Image Detector") as demo:
        gr.Markdown("# AI vs Real Image Detector")
        gr.Markdown("Upload an image for prediction, and review the latest training curves and evaluation figures.")

        with gr.Tab("Prediction"):
            with gr.Row():
                in_img = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    out_text = gr.Textbox(label="Prediction")
                    out_probs = gr.Label(label="Class Probabilities")
            run_btn = gr.Button("Predict")
            run_btn.click(fn=predict_image, inputs=in_img, outputs=[out_text, out_probs])

        with gr.Tab("Training Dashboard"):
            gr.Textbox(value=summary_text, label="Latest Results Summary", lines=7)
            if graph_images:
                gr.Gallery(
                    value=graph_images,
                    label="Generated Curves and Evaluation Figures",
                    columns=2,
                    rows=4,
                    object_fit="contain",
                    height="auto",
                )
            else:
                gr.Markdown("No graph files found yet. Run train.py first.")

    demo.launch()


if __name__ == "__main__":
    main()
