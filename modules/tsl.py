"""Text-guided Semantic Localization (TSL).

Stage 1:
    image + report text -> predicted CAM
    train with CAM label supervision.

Design in this version:
    ConvNeXt image encoder      : pretrained from vision_type, trainable by default.
    CXR-BERT text encoder       : pretrained from bert_type, frozen by default and kept in eval mode.
    Text-image fusion           : FiLM modulation on multi-scale ConvNeXt features.
    Decoder                     : light FPN / U-Net-like decoder -> 1-channel CAM logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from einops import repeat
except Exception:
    repeat = None


@dataclass
class CAMGeneratorOutput:
    cam_logits: torch.Tensor              # [B, 1, H, W]
    cam_prob: torch.Tensor                # [B, 1, H, W]
    text_embedding: torch.Tensor          # [B, text_dim]
    selected_feature_shapes: List[Tuple[int, ...]]


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation.

    Given image feature X_l and text embedding e_t, generate gamma and beta:
        X_l_text = (1 + gamma_l) * X_l + beta_l

    gamma and beta are channel-wise, shared over spatial positions.
    """

    def __init__(self, text_dim: int, channels: int, hidden_dim: int = 512):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        # Initially close to identity modulation. This avoids destroying the
        # pretrained ConvNeXt features at the beginning of training.
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(text_emb)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * x + beta


class TextConditionedFPNDecoder(nn.Module):
    """
    A light FPN / U-Net-like decoder.

    Inputs are four projected feature maps, ordered from high resolution to
    low resolution:
        x1: [B, D, 56, 56]
        x2: [B, D, 28, 28]
        x3: [B, D, 14, 14]
        x4: [B, D,  7,  7]

    Output is 1-channel CAM logits at target_size, normally 224x224.
    """

    def __init__(self, channels: int = 256, out_channels: int = 1):
        super().__init__()
        self.smooth4 = ConvBNAct(channels, channels)
        self.smooth3 = ConvBNAct(channels, channels)
        self.smooth2 = ConvBNAct(channels, channels)
        self.smooth1 = ConvBNAct(channels, channels)
        self.head = nn.Sequential(
            ConvBNAct(channels, channels // 2),
            nn.Conv2d(channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(features)}.")
        x1, x2, x3, x4 = features

        p4 = self.smooth4(x4)
        p3 = x3 + F.interpolate(p4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)
        p2 = x2 + F.interpolate(p3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)
        p1 = x1 + F.interpolate(p2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        logits = self.head(p1)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


class TextGuidedSemanticLocalization(nn.Module):
    """
    Image-and-report conditioned TSL network.

    This version follows the user's intended setting:
        text_encoder  = CXR-BERT from pretrained weights, frozen and not updated.
        vision_encoder = ConvNeXt from pretrained weights, trainable and updated.

    Notes:
        - The text encoder is kept in eval mode even when the whole model is set to train().
        - Gradients are disabled through CXR-BERT to reduce memory.
        - ConvNeXt remains trainable by default.
    """

    def __init__(
        self,
        vision_type: str,
        bert_type: str,
        cam_dim: int = 256,
        text_pooling: str = "mean",
        target_size: Optional[Tuple[int, int]] = (224, 224),
        freeze_vision: bool = False,
        freeze_text: bool = True,
        trust_remote_code: bool = True,
        convnext_channels: Sequence[int] = (96, 192, 384, 768),
    ):
        super().__init__()
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.cam_dim = int(cam_dim)
        self.text_pooling = str(text_pooling).lower()
        self.target_size = target_size
        self.freeze_text = bool(freeze_text)
        self.freeze_vision = bool(freeze_vision)

        # Use AutoModel to match your current model.py style:
        #   AutoModel.from_pretrained(vision_type, output_hidden_states=True)
        #   AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.vision_encoder = AutoModel.from_pretrained(
            vision_type,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
        )
        self.text_encoder = AutoModel.from_pretrained(
            bert_type,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
        )

        if self.freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()
        else:
            for p in self.vision_encoder.parameters():
                p.requires_grad = True

        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

        # ConvNeXt-Tiny channels are [96, 192, 384, 768], matching your model.py comments.
        self.expected_in_channels = list(convnext_channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, self.cam_dim, kernel_size=1) for c in self.expected_in_channels
        ])

        text_dim = self._infer_text_dim()
        self.film_blocks = nn.ModuleList([
            FiLMBlock(text_dim=text_dim, channels=self.cam_dim) for _ in range(4)
        ])
        self.decoder = TextConditionedFPNDecoder(channels=self.cam_dim, out_channels=1)

    def train(self, mode: bool = True):
        """Keep frozen CXR-BERT in eval mode even when parent model is train()."""
        super().train(mode)
        if self.freeze_text:
            self.text_encoder.eval()
        if self.freeze_vision:
            self.vision_encoder.eval()
        return self

    def _infer_text_dim(self) -> int:
        if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "hidden_size"):
            return int(self.text_encoder.config.hidden_size)
        return 768

    @staticmethod
    def _select_last_four_2d_features(hidden_states: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Select four 4D feature maps from ConvNeXt hidden states."""
        feats = [h for h in hidden_states if isinstance(h, torch.Tensor) and h.ndim == 4]
        if len(feats) < 4:
            raise RuntimeError(
                "Could not find four 4D feature maps from ConvNeXt hidden_states. "
                f"Found shapes: {[tuple(h.shape) for h in hidden_states if isinstance(h, torch.Tensor)]}"
            )
        feats = feats[-4:]
        # Ensure order high-res -> low-res.
        feats = sorted(feats, key=lambda t: t.shape[-1], reverse=True)
        return feats

    def _pool_text(self, text_outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(text_outputs, "last_hidden_state"):
            h = text_outputs.last_hidden_state
        elif isinstance(text_outputs, dict) and "last_hidden_state" in text_outputs:
            h = text_outputs["last_hidden_state"]
        elif hasattr(text_outputs, "hidden_states") and text_outputs.hidden_states is not None:
            h = text_outputs.hidden_states[-1]
        elif isinstance(text_outputs, dict) and "hidden_states" in text_outputs:
            h = text_outputs["hidden_states"][-1]
        else:
            raise RuntimeError("Text encoder output does not contain last_hidden_state or hidden_states.")

        if self.text_pooling == "cls":
            return h[:, 0]

        if self.text_pooling == "pooler" and hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            return text_outputs.pooler_output

        # Default: attention-mask-aware mean pooling.
        mask = attention_mask.to(dtype=h.dtype).unsqueeze(-1)  # [B, L, 1]
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _encode_text_frozen(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_text:
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return text_outputs

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        target_size = tuple(image.shape[-2:]) if self.target_size is None else tuple(self.target_size)

        # If image is grayscale, repeat to 3 channels, consistent with your existing model.py.
        if image.ndim == 4 and image.shape[1] == 1:
            if repeat is not None:
                image = repeat(image, "b 1 h w -> b c h w", c=3)
            else:
                image = image.repeat(1, 3, 1, 1)

        # ConvNeXt is trainable by default, so no torch.no_grad() here.
        vision_outputs = self.vision_encoder(
            image,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = vision_outputs.hidden_states if hasattr(vision_outputs, "hidden_states") else vision_outputs["hidden_states"]
        image_feats = self._select_last_four_2d_features(hidden_states)

        # Project image features to same dimension.
        projected = []
        for i, (feat, conv) in enumerate(zip(image_feats, self.lateral_convs)):
            if feat.shape[1] != conv.in_channels:
                raise RuntimeError(
                    f"ConvNeXt feature channel mismatch at stage {i + 1}: "
                    f"got {feat.shape[1]}, but lateral conv expects {conv.in_channels}. "
                    "For ConvNeXt-Tiny this should be [96, 192, 384, 768]. "
                    "If you use another ConvNeXt size, set MODEL.convnext_channels in yaml."
                )
            projected.append(conv(feat))

        # CXR-BERT is frozen. The output does not carry gradients through text_encoder,
        # but FiLM MLPs still receive gradients normally.
        text_outputs = self._encode_text_frozen(input_ids, attention_mask)
        text_emb = self._pool_text(text_outputs, attention_mask)

        # Text-conditioned modulation.
        modulated = [film(x, text_emb) for film, x in zip(self.film_blocks, projected)]

        cam_logits = self.decoder(modulated, target_size=target_size)
        cam_prob = torch.sigmoid(cam_logits)

        if not return_dict:
            return cam_logits

        return CAMGeneratorOutput(
            cam_logits=cam_logits,
            cam_prob=cam_prob,
            text_embedding=text_emb,
            selected_feature_shapes=[tuple(f.shape) for f in image_feats],
        )


def set_text_encoder_frozen(model: TextGuidedSemanticLocalization) -> None:
    """Utility for sanity checks in train script."""
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    model.text_encoder.eval()
    model.freeze_text = True
