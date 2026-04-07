import io
import base64
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.base import BrainTumorModel
from src.models.registry import (
    build_resnet50, build_efficientnet, build_convnext_small, 
    build_efficientnet_v2_s, get_gradcam
)
from src.utils.gradcam import apply_gradcam_overlay
from src.config import CLASS_NAMES, CLASS_VI, IMG_SIZE, DEVICE, SERVER_CONFIG
from src.dataset import get_val_transforms

class InferenceEngine:
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.models: Dict[str, BrainTumorModel] = {}
        self.gradcam_engines = {}
        self.transforms = get_val_transforms(IMG_SIZE)

    def load_model(self, name: str, path: str):
        if not Path(path).exists():
            print(f"[WARN] Checkpoint not found: {path}")
            return False
        
        builders = {
            # "resnet50": build_resnet50,
            # "efficientnet": build_efficientnet,
            # "convnext_small": build_convnext_small,
            "efficientnet_v2_s": build_efficientnet_v2_s,
        }
        
        if name not in builders:
            return False

        model = builders[name](pretrained=False)
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        
        self.models[name] = model
        self.gradcam_engines[name] = get_gradcam(model, variant=SERVER_CONFIG["gradcam_variant"])
        return True

    def preprocess(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        img_rgb = np.array(pil_image.convert("RGB"))
        orig_np = img_rgb.copy()
        aug = self.transforms(image=img_rgb)
        tensor = aug["image"].unsqueeze(0).to(self.device)
        return tensor, orig_np

    def predict(self, pil_image: Image.Image, gradcam_request: str = "both") -> Dict:
        if not self.models:
            raise RuntimeError("No models loaded.")

        tensor, orig_np = self.preprocess(pil_image)
        
        all_probs = {}
        for name, model in self.models.items():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            all_probs[name] = probs

        # Ensemble
        ensemble_probs = np.mean(list(all_probs.values()), axis=0) if len(all_probs) > 1 else list(all_probs.values())[0]
        
        pred_idx = int(np.argmax(ensemble_probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(ensemble_probs[pred_idx])

        # Grad-CAM
        gradcam_images = {}
        target_models = list(self.models.keys()) if gradcam_request == "both" else [gradcam_request]
        
        for name in target_models:
            if name in self.gradcam_engines:
                cam, _, _ = self.gradcam_engines[name].generate(tensor.squeeze(0), class_idx=pred_idx)
                overlay = apply_gradcam_overlay(orig_np, cam, img_size=IMG_SIZE)
                
                # Base64
                img_pil = Image.fromarray(overlay)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                gradcam_images[name] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "class": pred_class,
            "class_vi": CLASS_VI.get(pred_class, pred_class),
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(ensemble_probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "per_model": {
                name: {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
                for name, probs in all_probs.items()
            },
            "gradcam": gradcam_images
        }
