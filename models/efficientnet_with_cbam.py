import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .cbam import CBAM


class EfficientNetWithCBAM(nn.Module):
    def __init__(self, use_cbam=True, num_classes=2, dropout=0.4, weights=EfficientNet_B0_Weights.IMAGENET1K_V1):
        super().__init__()
        self.use_cbam = use_cbam

        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        in_channels = backbone.classifier[1].in_features

        if self.use_cbam:
            self.cbam = CBAM(in_channels=in_channels)
        else:
            self.cbam = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_efficientnet_with_cbam(checkpoint_path=None, use_cbam=True, num_classes=2, dropout=0.4, device='cpu'):
    model = EfficientNetWithCBAM(use_cbam=use_cbam, num_classes=num_classes, dropout=dropout)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)

        if 'model' in state_dict:
            state_dict = state_dict['model']

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if 'cbam' in str(e):
                print("Warning: Checkpoint doesn't have CBAM layers. Loading with strict=False and CBAM disabled.")
                model = EfficientNetWithCBAM(use_cbam=False, num_classes=num_classes, dropout=dropout)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise e

    model.eval()
    return model.to(device)
