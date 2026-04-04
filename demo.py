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


def softmax_with_temperature(logits_np, temperature: float):
    scaled = logits_np / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = torch.tensor(scaled).exp().numpy()
    return exp / exp.sum(axis=1, keepdims=True)


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

CALIBRATION = {
    "temperature": 1.0,
    "fake_threshold": 0.5,
    "ood_confidence_threshold": 0.55,
    "fake_class_index": 0 if any("fake" in c.lower() for c in CLASS_NAMES) else 1,
}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        _r = json.load(f)
    CALIBRATION.update(_r.get("calibration", {}))


def load_results_summary():
    if not os.path.exists(RESULTS_PATH):
        return "results.json not found. Run train.py first.", []

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best_config", {})
    final = data.get("final_test_metrics", {})
    kfold = data.get("kfold_results", {})
    dataset = data.get("dataset", {})
    external = data.get("external_test_metrics", {})

    # Format percentages
    best_val_acc = best.get('best_val_acc', 0)
    best_val_auc = best.get('best_val_auc', 0)
    test_acc = final.get('accuracy', 0)
    test_auc = final.get('auc', 0)
    kfold_acc = kfold.get('mean_accuracy', 0)
    kfold_std = kfold.get('std_accuracy', 0)
    kfold_auc = kfold.get('mean_auc', 0)
    kfold_auc_std = kfold.get('std_auc', 0)
    ext_acc = external.get('accuracy', 0)
    ext_auc = external.get('auc', 0)

    summary = (
        f"📊 TRAINING RESULTS SUMMARY\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Dataset: {dataset.get('name', 'CIFAKE')} ({dataset.get('total_samples', '?')} samples)\n"
        f"Model: Vision Transformer (ViT-base-patch16-224)\n\n"
        f"🏆 Best Configuration: {best.get('name', 'Config A')}\n"
        f"   LR: {best.get('lr', 'N/A')} | Batch: {best.get('batch_size', 'N/A')} | Epochs: {best.get('epochs', 'N/A')}\n\n"
        f"✅ Validation Performance\n"
        f"   Accuracy: {best_val_acc:.2%} | AUC: {best_val_auc:.4f}\n\n"
        f"🎯 Test Set Performance\n"
        f"   Accuracy: {test_acc:.2%} | AUC: {test_auc:.4f}\n\n"
        f"📈 K-Fold Cross-Validation (3-Fold)\n"
        f"   Mean Accuracy: {kfold_acc:.2%} ± {kfold_std:.2%}\n"
        f"   Mean AUC: {kfold_auc:.4f} ± {kfold_auc_std:.4f}\n\n"
        f"🔍 External OOD Test (Domain Shift Expected)\n"
        f"   Accuracy: {ext_acc:.2%} | AUC: {ext_auc:.4f} ({external.get('num_samples', '?')} samples)"
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


def load_results_data():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_image(image: Image.Image):
    if image is None:
        return "Please upload an image first.", {"AI-Generated": 0.0, "Real": 0.0}

    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        logits_np = logits.cpu().numpy()

    probs = softmax_with_temperature(logits_np, CALIBRATION.get("temperature", 1.0)).flatten()

    fake_idx = int(CALIBRATION.get("fake_class_index", 0))
    real_idx = 1 - fake_idx
    fake_prob = float(probs[fake_idx])
    real_prob = float(probs[real_idx])
    max_conf = float(max(fake_prob, real_prob))

    ood_th = float(CALIBRATION.get("ood_confidence_threshold", 0.55))
    if max_conf < ood_th:
        summary = (
            f"⚠️ UNCERTAIN PREDICTION\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"This image may be out-of-distribution or difficult to classify.\n"
            f"Max Confidence: {max_conf:.2%} (below threshold: {ood_th:.2%})\n\n"
            f"Model is uncertain. Consider manual review.\n"
            f"AI-Generated: {fake_prob:.2%} | Real: {real_prob:.2%}"
        )
        class_probabilities = {
            "AI-Generated": fake_prob,
            "Real": real_prob,
        }
        return summary, class_probabilities

    fake_threshold = float(CALIBRATION.get("fake_threshold", 0.5))
    pred_is_fake = fake_prob >= fake_threshold
    pred_label = "AI-Generated 🤖" if pred_is_fake else "Real 📸"
    confidence = fake_prob if pred_is_fake else real_prob

    # Confidence level indicator
    if confidence >= 0.9:
        confidence_level = "Very High (90%+) ✓✓✓"
    elif confidence >= 0.75:
        confidence_level = "High (75-90%) ✓✓"
    elif confidence >= 0.6:
        confidence_level = "Moderate (60-75%) ✓"
    else:
        confidence_level = f"Low ({confidence:.0%})"

    summary = (
        f"🔍 PREDICTION RESULT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Classification: {pred_label}\n"
        f"Confidence: {confidence:.2%} ({confidence_level})\n"
        f"Threshold Used: {fake_threshold:.3f}\n\n"
        f"Probabilities:\n"
        f"  AI-Generated: {fake_prob:.2%}\n"
        f"  Real: {real_prob:.2%}"
    )

    class_probabilities = {
        "AI-Generated": fake_prob,
        "Real": real_prob,
    }

    return summary, class_probabilities


def main():
    summary_text, graph_images = load_results_summary()
    results_data = load_results_data()
    dataset = results_data.get("dataset", {})
    final = results_data.get("final_test_metrics", {})
    kfold = results_data.get("kfold_results", {})
    external = results_data.get("external_test_metrics", {})

    with gr.Blocks(title="AI vs Real Image Detector") as demo:
        gr.Markdown("# 🎨 AI vs Real Image Detector")
        gr.Markdown(
            "Upload an image to determine whether it's **AI-generated** or a **real photograph**. "
            "Review training metrics and evaluation curves in the dashboard."
        )

        with gr.Tab("🔍 Prediction"):
            gr.Markdown("### Upload an image and get instant prediction")
            with gr.Row():
                in_img = gr.Image(
                    type="pil",
                    label="Upload Image (JPG, PNG, etc.)",
                    scale=2
                )
                with gr.Column(scale=1):
                    out_text = gr.Textbox(
                        label="Prediction Result",
                        lines=10,
                        interactive=False
                    )
                    out_probs = gr.Label(label="Class Probabilities")
            run_btn = gr.Button("🚀 Predict", variant="primary", size="lg")
            run_btn.click(fn=predict_image, inputs=in_img, outputs=[out_text, out_probs])
            
            gr.Markdown("""
            #### How it works
            - **Temperature Scaling**: Calibrated confidence estimates for reliable predictions
            - **Fake Threshold**: {:.3f} (tuned on validation set)
            - **OOD Detection**: Flags uncertain/out-of-distribution images (confidence < {:.2%})
            """.format(
                float(CALIBRATION.get("fake_threshold", 0.5)),
                float(CALIBRATION.get("ood_confidence_threshold", 0.55))
            ))

        with gr.Tab("📊 Training Dashboard"):
            gr.Markdown("### Model Performance & Training Results")
            gr.Textbox(
                value=summary_text,
                label="Results Summary",
                lines=14,
                interactive=False
            )
            if graph_images:
                gr.Markdown("### Evaluation Metrics")
                gr.Gallery(
                    value=graph_images,
                    label="Training Curves & Evaluation Figures",
                    columns=2,
                    rows=4,
                    object_fit="contain",
                    height="auto",
                )
            else:
                gr.Markdown("⚠️ No graph files found. Run `train.py` first.")
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ### Model Information
            - **Architecture**: Vision Transformer (ViT-base-patch16-224)
            - **Framework**: PyTorch + Hugging Face Transformers
            - **Dataset**: {dataset.get('name', 'CIFAKE')} ({dataset.get('total_samples', '?')} samples) + External Sources
            - **Device**: {DEVICE}
            
            ### Calibration Parameters
            - **Temperature**: {CALIBRATION.get('temperature', 1.0):.4f}
            - **Fake Threshold**: {CALIBRATION.get('fake_threshold', 0.5):.3f}
            - **OOD Threshold**: {CALIBRATION.get('ood_confidence_threshold', 0.55):.4f}
            
            ### Performance Summary
            From latest training run on CIFAKE dataset:
            - **Test Accuracy**: {final.get('accuracy', 0):.2%}
            - **Test AUC**: {final.get('auc', 0):.4f}
            - **K-Fold Robustness**: {kfold.get('mean_accuracy', 0):.2%} ± {kfold.get('std_accuracy', 0):.2%}
            - **External OOD Accuracy**: {external.get('accuracy', 0):.2%}
            
            ### Limitations
            - Model trained primarily on CIFAKE dataset
            - May underperform on significantly different image domains
            - Best results on high-quality images (224×224 minimum recommended)
            """)

    try:
        # Gradio 6+: theme moved from Blocks(...) to launch(...)
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            theme=gr.themes.Soft(),
        )
    except TypeError:
        # Backward compatibility for older Gradio versions
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
