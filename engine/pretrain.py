"""Paper-aligned supervised pre-training for the CAST-Seg segmentation network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceCELoss

from models.cast_seg import CASTSegNetwork


class CASTSegPretrainModule(pl.LightningModule):
    """Initialize the Teacher with labeled data and TSL-generated CAM weights."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = CASTSegNetwork(args.vision_type, args.project_dim)
        self.lr = float(args.lr)
        self.cam_prior_alpha = float(getattr(args, "cam_prior_alpha", 0.9))
        self.use_cam_prior = bool(getattr(args, "use_cam_prior", False))
        self.background_loss = DiceCELoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(self.args, "max_epochs", 200)),
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _parse_batch_x(x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError("Expected [image, ground_truth, optional_cam].")
        image, ground_truth = x[0], x[1]
        cam = x[2] if len(x) >= 3 and torch.is_tensor(x[2]) else None
        return image, ground_truth, cam

    @staticmethod
    def _weighted_dice_bce(prediction, target, weight, smooth=1e-5, eps=1e-6):
        prediction = prediction.float().clamp(eps, 1.0 - eps)
        target = target.float()
        weight = weight.float().detach()
        bce = -(
            target * torch.log(prediction)
            + (1.0 - target) * torch.log(1.0 - prediction)
        )
        bce = (bce * weight).sum() / (weight.sum() + eps)
        intersection = (prediction * target * weight).sum()
        denominator = ((prediction + target) * weight).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        return dice + bce

    def _cam_weight(self, cam, reference):
        if not self.use_cam_prior or cam is None:
            return torch.ones_like(reference)
        cam = cam.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        if cam.shape[-2:] != reference.shape[-2:]:
            cam = F.interpolate(cam, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        alpha = min(1.0, max(0.0, self.cam_prior_alpha))
        return (alpha + (1.0 - alpha) * cam).clamp(alpha, 1.0)

    def _shared_step(self, batch, stage):
        x, target, _ = batch
        image, ground_truth, cam = self._parse_batch_x(x)
        foreground, background, _, _ = self.model([image, ground_truth])
        foreground_weight = self._cam_weight(cam, foreground)
        foreground_loss = self._weighted_dice_bce(
            foreground,
            ground_truth,
            foreground_weight,
        )
        background_loss = self.background_loss(background, 1 - ground_truth)
        loss = foreground_loss + background_loss
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=stage != "train",
            on_step=stage == "train",
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
