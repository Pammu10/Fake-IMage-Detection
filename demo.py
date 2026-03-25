import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel


MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = os.path.join("outputs", "models", "best_overall.pt")


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
    description = (
        "Drag and drop any image to classify it as Real or AI-generated using a ViT-based model."
    )

    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Label(label="Class Probabilities"),
        ],
        title="AI vs Real Image Detector",
        description=description,
        allow_flagging="never",
    )

    demo.launch()


if __name__ == "__main__":
    main()
