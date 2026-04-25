import torch
import torch.nn as nn

class DinoBMIModelV1(nn.Module): #mean + max patch + CLS
    def __init__(self):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg'
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # CLS + avg patch + max patch = 768 * 3 = 2304
        self.head = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        outputs = self.backbone.forward_features(x)

        cls = outputs["x_norm_clstoken"]              # (B, 768)
        patch_tokens = outputs["x_norm_patchtokens"]  # (B, N, 768)

        patch_mean = patch_tokens.mean(dim=1)         # (B, 768)
        patch_max = patch_tokens.max(dim=1).values    # (B, 768)

        features = torch.cat(
            [cls, patch_mean, patch_max],
            dim=1
        )                                             # (B, 2304)

        out = self.head(features)
        return out.squeeze(1)
    
############VERSION 2: ATTENTION POOLING
import torch
import torch.nn as nn

class DinoBMIModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg'
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attention pooling over patch tokens
        self.attn_pool = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # CLS + attention-pooled patch = 768 * 2 = 1536
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
        outputs = self.backbone.forward_features(x)

        cls = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]

        attn_scores = self.attn_pool(patch_tokens)          # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (B, N, 1)
        patch_attn = (patch_tokens * attn_weights).sum(dim=1)  # (B, 768)

        features = torch.cat([cls, patch_attn], dim=1)      # (B, 1536)

        out = self.head(features)
        return out.squeeze(1)

############VERSION 3: ATTENTION POOLING + CLS + MAX + AVG

class DinoBMIModelV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg'
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attention pooling over patch tokens
        self.attn_pool = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # CLS + attention + avg + max = 768 * 4 = 3072
        self.head = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        outputs = self.backbone.forward_features(x)

        cls = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]

        patch_mean = patch_tokens.mean(dim=1)
        patch_max = patch_tokens.max(dim=1).values

        attn_scores = self.attn_pool(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)
        patch_attn = (patch_tokens * attn_weights).sum(dim=1)

        features = torch.cat(
            [cls, patch_attn, patch_mean, patch_max],
            dim=1
        )  # (B, 3072)

        out = self.head(features)
        return out.squeeze(1)