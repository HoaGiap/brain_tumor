"""
models.py — ResNet50 & EfficientNet-B0 với Grad-CAM
Hỗ trợ: fine-tuning, feature extraction, ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np
import cv2
from typing import Optional

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("[WARN] efficientnet_pytorch not installed. Using torchvision EfficientNet.")

from src.config import NUM_CLASSES, CLASS_NAMES


# ─── Base Model Wrapper ───────────────────────────────────────────────────────
class BrainTumorModel(nn.Module):
    """
    Wrapper chung cho ResNet50 và EfficientNet-B0.
    Hỗ trợ 3 chế độ training:
      - 'finetune': unfreeze toàn bộ
      - 'feature':  freeze backbone, chỉ train classifier
      - 'partial':  chỉ unfreeze N layer cuối
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        pretrained: bool = True,
        mode: str = "finetune",       # 'finetune' | 'feature' | 'partial'
        unfreeze_layers: int = 2,     # dùng khi mode='partial'
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.mode = mode

        if backbone == "resnet50":
            self._build_resnet50(pretrained, dropout, num_classes)
        elif backbone == "efficientnet":
            self._build_efficientnet(pretrained, dropout, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'efficientnet'")

        self._apply_freeze_mode(mode, unfreeze_layers)

    # ── ResNet50 ──────────────────────────────────────────────────────────────
    def _build_resnet50(self, pretrained, dropout, num_classes):
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv_models.resnet50(weights=weights)

        # Giữ nguyên backbone, thay classifier
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool           # AdaptiveAvgPool2d

        in_features = base.fc.in_features     # 2048
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # Target layer cho Grad-CAM (last conv block)
        self.gradcam_layer = self.feature_extractor[-1]   # layer4

    # ── EfficientNet-B0 ───────────────────────────────────────────────────────
    def _build_efficientnet(self, pretrained, dropout, num_classes):
        if EFFICIENTNET_AVAILABLE:
            if pretrained:
                base = EfficientNet.from_pretrained("efficientnet-b0")
            else:
                base = EfficientNet.from_name("efficientnet-b0")

            self.feature_extractor = base
            # Thay _fc
            in_features = base._fc.in_features
            base._fc = nn.Identity()          # remove original fc
            self.avgpool = nn.AdaptiveAvgPool2d(1)

            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.gradcam_layer = base._blocks[-1]   # last MBConv block
        else:
            # Fallback: torchvision EfficientNet-B0
            weights = (tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
                       if pretrained else None)
            base = tv_models.efficientnet_b0(weights=weights)
            self.feature_extractor = base.features
            self.avgpool = base.avgpool

            in_features = base.classifier[-1].in_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.gradcam_layer = self.feature_extractor[-1]

    # ── Freeze / Unfreeze ─────────────────────────────────────────────────────
    def _apply_freeze_mode(self, mode: str, unfreeze_layers: int):
        if mode == "feature":
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            for p in self.classifier.parameters():
                p.requires_grad = True

        elif mode == "partial":
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            # Unfreeze N layer cuối của feature_extractor
            children = list(self.feature_extractor.children())
            for child in children[-unfreeze_layers:]:
                for p in child.parameters():
                    p.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True

        else:  # 'finetune' — unfreeze all
            for p in self.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self.mode = "finetune"

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "efficientnet" and EFFICIENTNET_AVAILABLE:
            # EfficientNet từ efficientnet_pytorch
            x = self.feature_extractor.extract_features(x)
            x = self.feature_extractor._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.feature_extractor._dropout(x)
        else:
            x = self.feature_extractor(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return self.classifier(x)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Grad-CAM Implementation ──────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Ref: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: BrainTumorModel, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        smooth: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """
        Sinh Grad-CAM heatmap.

        Returns:
            cam: heatmap ndarray (H, W) normalized [0, 1]
            class_idx: predicted class index
            confidence: softmax confidence của predicted class
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor

        # Forward
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidences = probs[0].cpu().detach().numpy()

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Compute Grad-CAM
        grads = self._gradients[0]        # (C, H, W)
        acts  = self._activations[0]      # (C, H, W)

        # Global average pooling của gradients
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)                 # ReLU — chỉ lấy positive contributions

        cam = cam.cpu().detach().numpy()
        if smooth:
            cam = cv2.GaussianBlur(cam, (7, 7), 0)

        # Normalize after smoothing to preserve peak
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        cam = np.clip(cam, 0, 1)

        return cam, int(class_idx), confidences


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ — cải thiện độ chính xác vùng quan tâm."""

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        smooth: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidences = probs[0].cpu().detach().numpy()

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        grads = self._gradients[0]        # (C, H, W)
        acts  = self._activations[0]      # (C, H, W)

        # Grad-CAM++ weighting
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(1, 2), keepdim=True)    # (C, 1, 1)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = alpha_num / alpha_denom

        relu_grad = F.relu(score.exp() * grads)
        weights = (alpha * relu_grad).sum(dim=(1, 2))    # (C,)

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam).cpu().detach().numpy()
        if smooth:
            cam = cv2.GaussianBlur(cam, (7, 7), 0)

        # Normalize after smoothing to preserve peak
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = np.clip(cam, 0, 1)

        return cam, int(class_idx), confidences


# ─── Overlay Heatmap onto Image ───────────────────────────────────────────────
def apply_gradcam_overlay(
    original_img: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.45,
    img_size: int = 224,
) -> np.ndarray:
    """
    Ghép Grad-CAM heatmap lên ảnh gốc.

    Args:
        original_img: ảnh gốc (H, W, 3) uint8 RGB
        cam: Grad-CAM ndarray (H', W') [0,1]
        colormap: OpenCV colormap
        alpha: độ trong suốt của heatmap
        img_size: kích thước output

    Returns:
        overlay: ảnh ghép (img_size, img_size, 3) uint8 RGB
    """
    img_resized = cv2.resize(original_img, (img_size, img_size))
    cam_resized = cv2.resize(cam, (img_size, img_size))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(img_resized) * (1 - alpha) + np.float32(heatmap) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ─── Ensemble Model ───────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    """
    Ensemble ResNet50 + EfficientNet-B0.
    Kết hợp bằng weighted average của softmax probabilities.
    """

    def __init__(
        self,
        resnet_path: str,
        efficientnet_path: str,
        device: torch.device,
        resnet_weight: float = 0.5,
        efficientnet_weight: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.w_r = resnet_weight
        self.w_e = efficientnet_weight

        self.resnet = BrainTumorModel(backbone="resnet50")
        self.resnet.load_state_dict(
            torch.load(resnet_path, map_location=device)["model_state_dict"]
        )
        self.resnet.to(device).eval()

        self.efficientnet = BrainTumorModel(backbone="efficientnet")
        self.efficientnet.load_state_dict(
            torch.load(efficientnet_path, map_location=device)["model_state_dict"]
        )
        self.efficientnet.to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_r = F.softmax(self.resnet(x), dim=1)
        p_e = F.softmax(self.efficientnet(x), dim=1)
        return self.w_r * p_r + self.w_e * p_e   # weighted average


# ─── Quick Factory Functions ──────────────────────────────────────────────────
def build_resnet50(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="resnet50", pretrained=pretrained, mode=mode)


def build_efficientnet(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="efficientnet", pretrained=pretrained, mode=mode)


def get_gradcam(model: BrainTumorModel, variant: str = "gradcam") -> GradCAM:
    """Khởi tạo Grad-CAM cho model."""
    target_layer = model.gradcam_layer
    if model.backbone_name == "resnet50":
        # Target layer cụ thể hơn cho ResNet50 để tăng độ chính xác (last conv của block cuối)
        target_layer = model.feature_extractor[7][-1].conv3

    if variant == "gradcam++":
        return GradCAMPlusPlus(model, target_layer)
    return GradCAM(model, target_layer)
