import argparse
import logging
import os
import glob
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.inference.engine import InferenceEngine
from src.config import SERVER_CONFIG, CLASS_NAMES, CLASS_VI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Global Engine ─────────────────────────────────────────────────────────────
engine = InferenceEngine()

def get_latest_checkpoint(model_name: str):
    """Finds the most recent .pth file in checkpoints/{model_name}/**/"""
    base_dir = os.path.join("checkpoints", model_name)
    if not os.path.exists(base_dir):
        return None
    
    # Tìm tất cả file .pth
    pth_files = glob.glob(f"{base_dir}/**/*.pth", recursive=True)
    if not pth_files:
        return None
        
    # Lấy file có thời gian sửa đổi gần nhất
    return max(pth_files, key=os.path.getmtime)

def load_models_to_engine(args):
    """Load models based on CLI arguments or auto-detect."""
    configs = [
        # ("resnet50",           args.resnet),
        # ("efficientnet",       args.effnet),
        # ("convnext_small",     args.convnext),
        ("efficientnet_v2_s",  args.effnet_v2),
        # ("swin_t",             args.swin_t),
        # ("swin_b",             args.swin_b),
    ]
    for name, arg_path in configs:
        # Nếu user không truyền path vào, tự đi tìm file mới nhất
        path = arg_path if arg_path else get_latest_checkpoint(name)
        
        if path:
            if engine.load_model(name, path):
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"Failed to load {name} from {path}")

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(engine.models.keys()),
        "device": str(engine.device),
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    gradcam_model_req = request.args.get("gradcam_model", "both")

    try:
        pil_img = Image.open(file.stream)
        result = engine.predict(pil_img, gradcam_request=gradcam_model_req)
        
        # Add severity and recommendation (business logic)
        pred_class = result["prediction" if "prediction" in result else "class"]
        confidence = result["confidence"]
        
        response = {
            "prediction": {
                "class": pred_class,
                "class_vi": result["class_vi"],
                "confidence": confidence,
                "has_tumor": pred_class != "notumor",
            },
            "probabilities": {
                name: {"score_pct": score, "label_vi": CLASS_VI[name]}
                for name, score in result["probabilities"].items()
            },
            "per_model": result["per_model"],
            "gradcam": result["gradcam"],
            "severity": _get_severity(pred_class, confidence / 100.0),
            "recommendation": _get_recommendation(pred_class),
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
    p = argparse.ArgumentParser(description="Brain Tumor MRI Classifier API (Refactored)")
    p.add_argument("--resnet",   default=None)
    p.add_argument("--effnet",   default=None)
    p.add_argument("--convnext", default=None)
    p.add_argument("--effnet_v2", default=None)
    p.add_argument("--swin_t",    default=None)
    p.add_argument("--swin_b",    default=None)
    p.add_argument("--port",    type=int, default=SERVER_CONFIG["port"])
    p.add_argument("--host",    default=SERVER_CONFIG["host"])
    p.add_argument("--debug",   action="store_true", default=SERVER_CONFIG["debug"])
    args = p.parse_args()

    load_models_to_engine(args)
    if not engine.models:
        logger.warning("⚠ No models loaded! API will return 500/error for /predict.")

    logger.info(f"Starting Flask on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
