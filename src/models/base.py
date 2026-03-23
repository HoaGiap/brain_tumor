import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from src.config import NUM_CLASSES

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

class BrainTumorModel(nn.Module):
    """
    Wrapper chung cho ResNet50, EfficientNet, ConvNeXt
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        pretrained: bool = True,
        mode: str = "finetune",
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.mode = mode

        if backbone == "resnet50":
            self._build_resnet50(pretrained, dropout, num_classes)
        elif backbone == "efficientnet":
            self._build_efficientnet(pretrained, dropout, num_classes)
        elif backbone == "convnext_small":
            self._build_convnext_small(pretrained, dropout, num_classes)
        elif backbone == "efficientnet_v2_s":
            self._build_efficientnet_v2_s(pretrained, dropout, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self._apply_freeze_mode(mode, unfreeze_layers)

    def _build_resnet50(self, pretrained, dropout, num_classes):
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv_models.resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        in_features = base.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.gradcam_layer = self.feature_extractor[-1]

    def _build_efficientnet(self, pretrained, dropout, num_classes):
        if EFFICIENTNET_AVAILABLE:
            base = EfficientNet.from_pretrained("efficientnet-b0") if pretrained else EfficientNet.from_name("efficientnet-b0")
            self.feature_extractor = base
            in_features = base._fc.in_features
            base._fc = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.gradcam_layer = base._blocks[-1]
        else:
            weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
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

    def _build_convnext_small(self, pretrained, dropout, num_classes):
        weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.convnext_small(weights=weights)
        self.feature_extractor = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[-1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.gradcam_layer = self.feature_extractor[-1]

    def _build_efficientnet_v2_s(self, pretrained, dropout, num_classes):
        weights = tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.efficientnet_v2_s(weights=weights)
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

    def _apply_freeze_mode(self, mode: str, unfreeze_layers: int):
        if mode == "feature":
            for p in self.feature_extractor.parameters(): p.requires_grad = False
        elif mode == "partial":
            for p in self.feature_extractor.parameters(): p.requires_grad = False
            children = list(self.feature_extractor.children())
            for child in children[-unfreeze_layers:]:
                for p in child.parameters(): p.requires_grad = True
        for p in self.classifier.parameters(): p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "efficientnet" and EFFICIENTNET_AVAILABLE:
            x = self.feature_extractor.extract_features(x)
            x = self.feature_extractor._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.feature_extractor._dropout(x)
        else:
            x = self.feature_extractor(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return self.classifier(x)
