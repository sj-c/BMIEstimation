import torch
import torch.nn as nn

class DinoBMIModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load DINOv2
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg'
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # CLS embedding
        features = self.backbone(x)  # (B, 768)

        out = self.head(features)
        return out.squeeze(1)