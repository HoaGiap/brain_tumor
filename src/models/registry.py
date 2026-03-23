from src.models.base import BrainTumorModel
from src.utils.gradcam import GradCAM, GradCAMPlusPlus

def build_resnet50(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="resnet50", pretrained=pretrained, mode=mode)

def build_efficientnet(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="efficientnet", pretrained=pretrained, mode=mode)

def build_convnext_small(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="convnext_small", pretrained=pretrained, mode=mode)

def build_efficientnet_v2_s(pretrained=True, mode="finetune") -> BrainTumorModel:
    return BrainTumorModel(backbone="efficientnet_v2_s", pretrained=pretrained, mode=mode)

def get_gradcam(model: BrainTumorModel, variant: str = "gradcam") -> GradCAM:
    target_layer = model.gradcam_layer
    if model.backbone_name == "resnet50":
        target_layer = model.feature_extractor[7][-1].conv3
    elif model.backbone_name == "convnext_small":
        target_layer = model.feature_extractor[7][-1]
    elif model.backbone_name == "efficientnet_v2_s":
        target_layer = model.feature_extractor[-1]

    if variant == "gradcam++":
        return GradCAMPlusPlus(model, target_layer)
    return GradCAM(model, target_layer)
