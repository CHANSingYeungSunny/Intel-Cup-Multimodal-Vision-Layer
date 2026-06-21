"""
Swin Transformer Health Classifier
==================================
Swin-Base backbone with a custom 3-class head.  Returns both logits and
a pooled feature vector for the Fusion Layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor

from config import (
    SWIN_MODEL,
    NUM_CLASSES,
    FEATURE_DIM,
    DEVICE,
)


class SwinHealthClassifier(nn.Module):
    """
    Wraps a torchvision Swin Transformer for 3-class health classification.

    Args:
        pretrained  – load ImageNet-22k weights (default True)
        freeze_stages – number of Swin stages to freeze (0 = all trainable)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_stages: int = 0,
    ):
        super().__init__()

        # Resolve model name → torchvision constructor.
        # Prefer ImageNet-22K weights, fall back to ImageNet-1K.
        if pretrained:
            try:
                weights = tv_models.Swin_T_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"
        else:
            weights = None
        self.backbone = tv_models.swin_t(weights=weights)

        # Replace the pretrained classifier head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # remove original head
        self.feature_dim = in_features

        # Optional freezing
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        # New classifier head
        self.norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(in_features, num_classes)

        # Initialise head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze the first *num_stages* Swin stages (0–3)."""
        stage_names = ["features.0", "features.2", "features.4", "features.6"]
        for i in range(min(num_stages, 4)):
            prefix = stage_names[i]
            for name, param in self.backbone.named_parameters():
                if name.startswith(prefix):
                    param.requires_grad = False
            print(f"  Frozen Swin stage {i}  ({prefix})")

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x  –  (B, 3, 224, 224)

        Returns:
            logits         – (B, num_classes)
            feature_vector – (B, feature_dim)   [before final linear]
        """
        # Swin backbone → pooled feature
        features = self.backbone(x)            # (B, 1024)
        pooled = self.norm(features)
        pooled = self.dropout(pooled)

        logits = self.head(pooled)             # (B, 3)

        return logits, pooled


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    pretrained: bool = True,
    freeze_stages: int = 0,
    device: str = DEVICE,
) -> SwinHealthClassifier:
    """Create and move the model to *device*."""
    model = SwinHealthClassifier(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        freeze_stages=freeze_stages,
    )
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SwinHealthClassifier: {total:,} params ({trainable:,} trainable)")
    return model


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha  – per-class weighting tensor (C,) or scalar.
        gamma  – focusing parameter (0 = plain CE).
    """

    def __init__(self, alpha: Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)          # p_t for the true class
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    logits, feats = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Logits: {logits.shape}  {logits.argmax(dim=-1)}")
    print(f"Feats:  {feats.shape}")

    # Focal loss smoke test
    fl = FocalLoss(gamma=2.0)
    targets = torch.tensor([0, 2], device=DEVICE)
    loss = fl(logits, targets)
    print(f"FocalLoss: {loss.item():.4f}")
