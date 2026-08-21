import torch.nn as nn
import torchvision.models as tvm


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
