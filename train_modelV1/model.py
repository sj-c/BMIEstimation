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
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Get all tokens
        outputs = self.backbone.forward_features(x)

        # CLS token
        cls = outputs["x_norm_clstoken"]        # (B, 768)

        # Patch tokens
        patch_tokens = outputs["x_norm_patchtokens"]  # (B, N, 768)

        # Mean pooling over patches
        patch_mean = patch_tokens.mean(dim=1)  # (B, 768)

        # Combine CLS + patch mean
        features = torch.cat([cls, patch_mean], dim=1)  # (B, 1536)

        out = self.head(features)
        return out.squeeze(1)