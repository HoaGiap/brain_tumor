"""
app.py — Flask API Backend
Cung cấp endpoint /predict nhận ảnh MRI và trả về:
  - Phân loại (4 classes)
  - Confidence scores
  - Grad-CAM heatmap (base64)
  - Ensemble ResNet50 + EfficientNet-B0

Cách chạy:
  python app.py --resnet  checkpoints/resnet50/best.pth \
                --effnet  checkpoints/efficientnet/best.pth \
                --port    5000
"""

import os
import sys
import io
import base64
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from src.models import (
    BrainTumorModel, GradCAM, GradCAMPlusPlus,
    apply_gradcam_overlay, build_resnet50, build_efficientnet, get_gradcam
)
from src.dataset import CLASS_NAMES, CLASS_VI, IMG_SIZE, get_val_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Global Models ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models: dict[str, BrainTumorModel] = {}
gradcam_engines: dict[str, GradCAM] = {}


def load_models(resnet_path: str | None, effnet_path: str | None):
    global models, gradcam_engines

    for name, path, builder in [
        ("resnet50",     resnet_path, build_resnet50),
        ("efficientnet", effnet_path, build_efficientnet),
    ]:
        if path and Path(path).exists():
            logger.info(f"Loading {name} from {path}...")
            m = builder(pretrained=False)
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            models[name] = m
            gradcam_engines[name] = get_gradcam(m, variant="gradcam++")
            logger.info(f"  {name} loaded ✓")
        else:
            logger.warning(f"  {name}: checkpoint not found at {path}, skipping.")


# ─── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor: preprocessed tensor (1, 3, H, W)
        orig_np: original numpy array (H, W, 3) uint8 RGB
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_np = img_rgb.copy()

    transform = get_val_transforms(IMG_SIZE)
    aug = transform(image=img_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
    return tensor, orig_np


# ─── Single Model Prediction ───────────────────────────────────────────────────
@torch.no_grad()
def predict_single(
    model: BrainTumorModel,
    tensor: torch.Tensor,
) -> np.ndarray:
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    return probs  # (num_classes,)


# ─── Grad-CAM Generation ───────────────────────────────────────────────────────
def generate_gradcam_image(
    model_name: str,
    tensor: torch.Tensor,
    orig_np: np.ndarray,
    class_idx: int,
) -> str:
    """Trả về Grad-CAM overlay ảnh dạng base64 PNG."""
    engine = gradcam_engines.get(model_name)
    if engine is None:
        return ""

    cam, _, _ = engine.generate(tensor.squeeze(0), class_idx=class_idx)
    overlay = apply_gradcam_overlay(orig_np, cam, img_size=IMG_SIZE)

    # Convert to base64
    img_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "device": str(device),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: multipart/form-data với field 'image'
    Optional query params:
        - gradcam_model: 'resnet50' | 'efficientnet' | 'both' (default: 'both')
        - gradcam_variant: 'gradcam' | 'gradcam++' (default: 'gradcam++')
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    gradcam_model_req = request.args.get("gradcam_model", "both")

    try:
        pil_img = Image.open(file.stream)
        tensor, orig_np = preprocess_image(pil_img)
    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 400

    if not models:
        return jsonify({"error": "No models loaded. Please provide checkpoint paths."}), 503

    # ── Predictions ──────────────────────────────────────────────────────────
    all_probs = {}
    for name, m in models.items():
        all_probs[name] = predict_single(m, tensor).tolist()

    # Ensemble (weighted average if both available, else single)
    if len(all_probs) == 2:
        ensemble_probs = np.array([
            0.5 * np.array(all_probs["resnet50"]) +
            0.5 * np.array(all_probs["efficientnet"])
        ])[0]
    else:
        ensemble_probs = np.array(list(all_probs.values())[0])

    pred_class_idx = int(np.argmax(ensemble_probs))
    pred_class     = CLASS_NAMES[pred_class_idx]
    confidence     = float(ensemble_probs[pred_class_idx])
    has_tumor      = pred_class != "notumor"

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    gradcam_images = {}
    models_for_cam = (
        list(models.keys())
        if gradcam_model_req == "both"
        else [gradcam_model_req]
        if gradcam_model_req in models
        else list(models.keys())[:1]
    )

    for name in models_for_cam:
        if name in models:
            gradcam_images[name] = generate_gradcam_image(
                name, tensor, orig_np, pred_class_idx
            )

    # ── Response ─────────────────────────────────────────────────────────────
    response = {
        "prediction": {
            "class":      pred_class,
            "class_vi":   CLASS_VI.get(pred_class, pred_class),
            "confidence": round(confidence * 100, 2),
            "has_tumor":  has_tumor,
        },
        "probabilities": {
            CLASS_NAMES[i]: {
                "score_pct": round(float(ensemble_probs[i]) * 100, 2),
                "label_vi":  CLASS_VI[CLASS_NAMES[i]],
            }
            for i in range(len(CLASS_NAMES))
        },
        "per_model": {
            name: {
                CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
            for name, probs in all_probs.items()
        },
        "gradcam": gradcam_images,
        "severity": _get_severity(pred_class, confidence),
        "recommendation": _get_recommendation(pred_class),
    }
    return jsonify(response)


def _get_severity(pred_class: str, confidence: float) -> dict:
    severity_map = {
        "notumor":    ("none",   "Bình thường"),
        "meningioma": ("medium", "Trung bình"),
        "pituitary":  ("medium", "Trung bình"),
        "glioma":     ("high",   "Cao"),
    }
    level, label = severity_map.get(pred_class, ("unknown", "Không xác định"))
    return {"level": level, "label": label, "confidence_pct": round(confidence * 100, 2)}


def _get_recommendation(pred_class: str) -> str:
    recs = {
        "notumor":    "Không phát hiện bất thường. Tiếp tục theo dõi định kỳ theo lịch hẹn.",
        "meningioma": "Phát hiện dấu hiệu u màng não. Cần chụp MRI có cản quang và tham vấn bác sĩ thần kinh.",
        "pituitary":  "Phát hiện dấu hiệu u tuyến yên. Cần xét nghiệm hormone và tham vấn chuyên khoa nội tiết.",
        "glioma":     "Phát hiện dấu hiệu u thần kinh đệm. Cần sinh thiết và tham vấn bác sĩ thần kinh-ung thư ngay.",
    }
    return recs.get(pred_class, "Vui lòng tham vấn bác sĩ chuyên khoa.")


# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Brain Tumor MRI Classifier API")
    p.add_argument("--resnet",  default=None, help="Path to ResNet50 checkpoint")
    p.add_argument("--effnet",  default=None, help="Path to EfficientNet checkpoint")
    p.add_argument("--port",    type=int, default=5000)
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--debug",   action="store_true")
    args = p.parse_args()

    logger.info(f"Device: {device}")
    load_models(args.resnet, args.effnet)

    if not models:
        logger.warning("⚠ No models loaded! API will return 503 for /predict.")
        logger.warning("  Provide checkpoint paths via --resnet and/or --effnet")

    logger.info(f"Starting Flask on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
