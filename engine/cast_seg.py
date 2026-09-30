from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex

from models.cast_seg import CASTSegNetwork
from modules.acc import AgreementAwareConfidenceCalibration
from modules.wema import weighted_ema_update


class CASTSegModule(pl.LightningModule):
    """
    CAST-Seg dual-student semi-supervised training module.

    Networks with identical architecture:
      - self.student_a: Student-A, learns from Teacher pseudo-labels through ACC.
      - self.student_b: Student-B, learns from MDAA-adapted SAM expert masks.
      - self.teacher:   Teacher, updated from both students through WEMA.

    Main training logic:
      labeled samples:
        Student-A and Student-B both learn GT.
      unlabeled samples:
        Student-A uses teacher predictions as the pseudo-label source.
        Student-B uses SAM expert masks as the pseudo-label source.
        Apart from pseudo-label source, both students use the same pipeline:
          agreement-confidence filtering -> CAM residual reweight -> masked Dice+BCE.
      after each train step:
        Teacher <- EMA(Teacher, weighted_average(Student-A, Student-B)).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.student_a = CASTSegNetwork(args.vision_type, args.project_dim)
        self.student_b = CASTSegNetwork(args.vision_type, args.project_dim)
        self.teacher = deepcopy(self.student_a)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.lr = float(args.lr)
        self.history = {}
        self.loss_fn = DiceCELoss()

        # Agreement-aware Confidence Calibration (ACC).
        self.pseudo_agree_threshold = float(getattr(args, "pseudo_agree_threshold", 0.30))
        self.pseudo_conflict_threshold = float(getattr(args, "pseudo_conflict_threshold", 0.70))
        self.pseudo_conf_margin = float(getattr(args, "pseudo_conf_margin", 0.20))
        self.acc = AgreementAwareConfidenceCalibration(
            decision_threshold=float(getattr(args, "pseudo_label_threshold", 0.5)),
            agreement_threshold=self.pseudo_agree_threshold,
            conflict_threshold=self.pseudo_conflict_threshold,
            confidence_margin=self.pseudo_conf_margin,
        )

        # TSL/ASR CAM residual prior shared by both students.
        self.use_cam_prior = bool(getattr(args, "use_cam_prior", False))
        self.cam_prior_min_value = float(getattr(args, "cam_prior_min_value", 0.05))
        self.cam_prior_zero_eps = float(getattr(args, "cam_prior_zero_eps", 0.0))
        self.cam_prior_mode = str(getattr(args, "cam_prior_mode", "residual")).lower()
        self.cam_prior_alpha = float(getattr(args, "cam_prior_alpha", 0.9))
        self.cam_bg_balance = bool(getattr(args, "cam_bg_balance", False))
        self.cam_bg_weight_min = float(getattr(args, "cam_bg_weight_min", 0.30))
        self.cam_bg_weight_max = float(getattr(args, "cam_bg_weight_max", 1.00))

        # MDAA-adapted SAM expert branch.
        self.use_sam_prior = bool(getattr(args, "use_sam_prior", True))
        self.sam_pseudo_threshold = float(getattr(args, "sam_pseudo_threshold", 0.5))
        self.sam_unsup_weight = float(getattr(args, "sam_unsup_weight", 1.0))
        self.student_a_unsup_weight = float(getattr(args, "student_a_unsup_weight", 1.0))
        self.student_b_labeled_weight = float(getattr(args, "student_b_labeled_weight", 1.0))
        self.sam_loss_on_all_pixels = bool(getattr(args, "sam_loss_on_all_pixels", True))
        self.sam_empty_mask_policy = str(getattr(args, "sam_empty_mask_policy", "use")).lower()

        # Total loss weights.
        self.unlabeled_loss_weight = float(getattr(args, "unlabeled_loss_weight", 0.6))

        # Teacher update from two students.
        self.ema_decay = float(getattr(args, "ema_decay", 0.99))
        self.wema_student_a_weight = float(
            getattr(args, "wema_student_a_weight", getattr(args, "teacher_student_a_weight", 0.75))
        )

        # Which network to evaluate. Recommended: teacher.
        self.eval_model = str(getattr(args, "eval_model", "teacher")).lower()

        metrics_dict = {
            "acc": BinaryAccuracy(),
            "dice": BinaryF1Score(),
            "MIoU": BinaryJaccardIndex(),
        }
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        self.save_hyperparameters()

    def configure_optimizers(self):
        params = list(self.student_a.parameters()) + list(self.student_b.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        return self.student_a.forward(x)

    def _forward_eval_model(self, model_x):
        if self.eval_model == "student_b":
            return self.student_b(model_x)
        if self.eval_model == "ensemble":
            preds_a, bg_a, img_a, neg_a = self.student_a(model_x)
            preds_b, bg_b, img_b, neg_b = self.student_b(model_x)
            return 0.5 * (preds_a + preds_b), 0.5 * (bg_a + bg_b), img_a, neg_a
        if self.eval_model == "teacher":
            return self.teacher(model_x)
        return self.student_a(model_x)

    # ------------------------------------------------------------------
    # Teacher update
    # ------------------------------------------------------------------
    def update_ema_variables_dual(self, global_step, alpha=0.99):
        weighted_ema_update(
            teacher=self.teacher,
            student_a=self.student_a,
            student_b=self.student_b,
            beta=alpha,
            student_a_weight=self.wema_student_a_weight,
        )

    # ------------------------------------------------------------------
    # Common losses and pseudo-label utilities
    # ------------------------------------------------------------------
    def _agreement_confidence_pseudo_label(self, prob_a, prob_b, eps=1e-6):
        return self.acc.calibrate_pair(prob_a, prob_b)

    def build_agreement_confidence_pseudo_labels(self, ema_preds, ema_bg_output):
        teacher_fg = ema_preds.detach()
        teacher_bg = ema_bg_output.detach()

        pseudo_fg, pseudo_bg, mask_fg, mask_bg, debug = self.acc(teacher_fg, teacher_bg)

        return pseudo_fg, pseudo_bg, mask_fg, mask_bg, debug

    def masked_dice_bce_loss(self, pred, target, mask, smooth=1e-5, eps=1e-6):
        pred = pred.float().clamp(eps, 1.0 - eps)
        target = target.float()
        mask = mask.float().detach()

        valid_pixels = mask.sum()
        if valid_pixels.item() < 1:
            return pred.sum() * 0.0

        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        bce = (bce * mask).sum() / (valid_pixels + eps)

        intersection = (pred * target * mask).sum()
        denominator = (pred * mask).sum() + (target * mask).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        return dice + bce

    def build_cam_prior_mask(self, cam_batch, reference_tensor, sample_index=None):
        if cam_batch is None or not torch.is_tensor(cam_batch):
            return torch.ones_like(reference_tensor).float()
        idx = 0 if sample_index is None else int(sample_index)
        cam = cam_batch[idx] if cam_batch.dim() == 4 else cam_batch
        if cam.dim() == 2:
            cam = cam.unsqueeze(0)
        if cam.shape[0] != 1:
            cam = cam[:1]
        cam = cam.to(device=reference_tensor.device, dtype=reference_tensor.dtype).float()
        if cam.shape[-2:] != reference_tensor.shape[-2:]:
            cam = F.interpolate(cam.unsqueeze(0), size=reference_tensor.shape[-2:], mode="bilinear", align_corners=False)[0]
        cam = cam.clamp(0.0, 1.0)
        if self.cam_prior_min_value > 0:
            cam = torch.where(cam <= self.cam_prior_zero_eps, torch.full_like(cam, self.cam_prior_min_value), cam)
        while cam.dim() < reference_tensor.dim():
            cam = cam.unsqueeze(0)
        return cam.to(device=reference_tensor.device, dtype=reference_tensor.dtype).clamp(0.0, 1.0)

    def build_sam_expert_target(self, sam_batch, reference_tensor, sample_index=None):
        if sam_batch is None or not torch.is_tensor(sam_batch):
            return None, None
        idx = 0 if sample_index is None else int(sample_index)
        sam = sam_batch[idx] if sam_batch.dim() == 4 else sam_batch
        if sam.dim() == 2:
            sam = sam.unsqueeze(0)
        if sam.shape[0] != 1:
            sam = sam[:1]
        sam = sam.to(device=reference_tensor.device, dtype=reference_tensor.dtype).float()
        if sam.shape[-2:] != reference_tensor.shape[-2:]:
            sam = F.interpolate(sam.unsqueeze(0), size=reference_tensor.shape[-2:], mode="bilinear", align_corners=False)[0]
        sam = sam.clamp(0.0, 1.0)
        sam_hard = (sam >= self.sam_pseudo_threshold).float()

        if self.sam_loss_on_all_pixels:
            sam_mask = torch.ones_like(sam_hard)
        else:
            # Use confident SAM pixels only; with binary masks this equals all foreground pixels.
            sam_mask = ((sam <= 0.05) | (sam >= self.sam_pseudo_threshold)).float()

        if self.sam_empty_mask_policy == "skip" and sam_hard.sum().item() < 1:
            sam_mask = torch.zeros_like(sam_hard)

        while sam_hard.dim() < reference_tensor.dim():
            sam_hard = sam_hard.unsqueeze(0)
        while sam_mask.dim() < reference_tensor.dim():
            sam_mask = sam_mask.unsqueeze(0)
        return sam_hard.to(reference_tensor.dtype), sam_mask.to(reference_tensor.dtype)

    def _apply_cam_prior_to_foreground_mask(self, cam_batch, mask_fg, mask_bg, sample_index=None):
        """
        Apply exactly the same CAM-prior weighting to a student's foreground
        reliable mask. This is shared by Student-A and Student-B.

        residual mode:
            M_fg = M_fg * [alpha + (1 - alpha) * CAM]

        direct mode:
            M_fg = M_fg * CAM
        """
        cam_bg_weight = torch.tensor(1.0, device=mask_fg.device, dtype=mask_fg.dtype)

        if not (self.use_cam_prior and cam_batch is not None):
            return mask_fg, mask_bg, cam_bg_weight

        base_foreground_mask = mask_fg.detach()
        cam_prior_mask = self.build_cam_prior_mask(cam_batch, mask_fg, sample_index=sample_index)

        if self.cam_prior_mode == "residual":
            alpha = max(0.0, min(1.0, self.cam_prior_alpha))
            cam_weight = alpha + (1.0 - alpha) * cam_prior_mask
            cam_weight = torch.clamp(cam_weight, min=alpha, max=1.0)
        else:
            cam_weight = cam_prior_mask

        if self.cam_bg_balance:
            cam_bg_weight = self._compute_cam_bg_weight(cam_weight, base_foreground_mask)

        mask_fg = mask_fg * cam_weight
        return mask_fg, mask_bg, cam_bg_weight

    def _compute_cam_bg_weight(self, cam_prior_mask, base_foreground_mask, eps=1e-6):
        cam = cam_prior_mask.detach().float()
        mask = base_foreground_mask.detach().float()
        if cam.shape[-2:] != mask.shape[-2:]:
            cam = F.interpolate(cam.unsqueeze(0), size=mask.shape[-2:], mode="bilinear", align_corners=False)[0]
        while cam.dim() < mask.dim():
            cam = cam.unsqueeze(0)
        while mask.dim() < cam.dim():
            mask = mask.unsqueeze(0)
        valid = mask.sum()
        if valid.item() < 1:
            weight = torch.tensor(1.0, device=mask.device, dtype=mask.dtype)
        else:
            weight = (cam * mask).sum() / (valid + eps)
        return torch.clamp(weight.detach(), min=self.cam_bg_weight_min, max=self.cam_bg_weight_max)

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------
    def _parse_batch_x(self, x):
        image, gt = None, None
        cam_batch, sam_batch, text_batch = None, None, None

        if isinstance(x, (list, tuple)):
            if len(x) >= 4 and torch.is_tensor(x[1]) and torch.is_tensor(x[2]) and torch.is_tensor(x[3]):
                image, cam_batch, sam_batch, gt = x[0], x[1], x[2], x[3]
            elif len(x) >= 3 and torch.is_tensor(x[1]) and torch.is_tensor(x[2]):
                image, cam_batch, gt = x[0], x[1], x[2]
            elif len(x) >= 3 and isinstance(x[1], dict):
                image, text_batch, gt = x[0], x[1], x[2]
            elif len(x) >= 2:
                image, gt = x[0], x[1]
            else:
                raise ValueError("Unsupported batch x format.")
        else:
            image, gt = x
        return image, gt, cam_batch, sam_batch, text_batch

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def shared_step(self, batch, batch_idx, exists_zero_image=False):
        x, y, flag = batch
        image, gt, cam_batch, sam_batch, _ = self._parse_batch_x(x)

        if self.trainer.training:
            model_x = [image, gt]
            y = gt

            # Student-A: Teacher/ACC branch.
            preds_a, bg_a, img_a, neg_a = self.student_a(model_x)
            # Student-B: MDAA-adapted SAM expert branch.
            preds_b, bg_b, img_b, neg_b = self.student_b(model_x)

            with torch.no_grad():
                ema_preds, ema_bg_output, ema_img, ema_neg = self.teacher(model_x)

            labeled_loss_sum = 0.0
            unlabeled_a_loss_sum = 0.0
            unlabeled_b_loss_sum = 0.0
            labeled_count = 0
            unlabeled_count = 0

            for i in range(gt.size(0)):
                if flag[i] == 0:
                    # Student-A: pseudo-label source = Teacher.
                    # Then: agreement-confidence filtering -> shared CAM reweight -> masked Dice+BCE.
                    y_fg_a, y_bg_a, mask_fg_a, mask_bg_a, _ = self.build_agreement_confidence_pseudo_labels(
                        ema_preds[i],
                        ema_bg_output[i],
                    )
                    mask_fg_a, mask_bg_a, cam_bg_weight_a = self._apply_cam_prior_to_foreground_mask(
                        cam_batch=cam_batch,
                        mask_fg=mask_fg_a,
                        mask_bg=mask_bg_a,
                        sample_index=i,
                    )
                    loss_a_fg = self.masked_dice_bce_loss(preds_a[i], y_fg_a, mask_fg_a)
                    loss_a_bg = self.masked_dice_bce_loss(bg_a[i], y_bg_a, mask_bg_a)
                    loss_a = loss_a_fg + cam_bg_weight_a * loss_a_bg
                    unlabeled_a_loss_sum += loss_a

                    # Student-B: direct pseudo-label source = MDAA-adapted SAM expert.
                    if self.use_sam_prior and sam_batch is not None:
                        y_fg_b, mask_fg_b = self.build_sam_expert_target(
                            sam_batch=sam_batch,
                            reference_tensor=preds_b[i],
                            sample_index=i,
                        )
                        if y_fg_b is None:
                            loss_b = preds_b[i].sum() * 0.0
                        else:
                            y_bg_b = 1.0 - y_fg_b
                            mask_bg_b = mask_fg_b
                            mask_fg_b, mask_bg_b, cam_bg_weight_b = self._apply_cam_prior_to_foreground_mask(
                                cam_batch=cam_batch,
                                mask_fg=mask_fg_b,
                                mask_bg=mask_bg_b,
                                sample_index=i,
                            )
                            loss_b_fg = self.masked_dice_bce_loss(preds_b[i], y_fg_b, mask_fg_b)
                            loss_b_bg = self.masked_dice_bce_loss(bg_b[i], y_bg_b, mask_bg_b)
                            loss_b = loss_b_fg + cam_bg_weight_b * loss_b_bg
                    else:
                        loss_b = preds_b[i].sum() * 0.0
                    unlabeled_b_loss_sum += loss_b
                    unlabeled_count += 1
                else:
                    # Both students learn GT on labeled data.
                    loss_a = self.loss_fn(preds_a[i].unsqueeze(0), y[i].unsqueeze(0)) + \
                             self.loss_fn(bg_a[i].unsqueeze(0), 1 - y[i].unsqueeze(0))
                    loss_b = self.loss_fn(preds_b[i].unsqueeze(0), y[i].unsqueeze(0)) + \
                             self.loss_fn(bg_b[i].unsqueeze(0), 1 - y[i].unsqueeze(0))
                    labeled_loss = loss_a + self.student_b_labeled_weight * loss_b
                    labeled_loss_sum += labeled_loss
                    labeled_count += 1

            labeled_loss_mean = labeled_loss_sum / labeled_count if labeled_count > 0 else preds_a.sum() * 0.0
            unlabeled_a_loss_mean = unlabeled_a_loss_sum / unlabeled_count if unlabeled_count > 0 else preds_a.sum() * 0.0
            unlabeled_b_loss_mean = unlabeled_b_loss_sum / unlabeled_count if unlabeled_count > 0 else preds_b.sum() * 0.0

            unlabeled_loss = self.student_a_unsup_weight * unlabeled_a_loss_mean + \
                             self.sam_unsup_weight * unlabeled_b_loss_mean
            loss = labeled_loss_mean + self.unlabeled_loss_weight * unlabeled_loss

            # Train metrics: use Student-A predictions and teacher pseudo targets for unlabeled samples.
            processed_y = []
            for i in range(gt.size(0)):
                if flag[i] == 0:
                    y_fg, _, _, _, _ = self.build_agreement_confidence_pseudo_labels(
                        ema_preds[i],
                        ema_bg_output[i],
                    )
                    processed_y.append(y_fg.detach())
                else:
                    processed_y.append(y[i].detach())
            y_metric = torch.stack(processed_y).int()

            preds_metric = preds_a.detach()

            self.log("loss_labeled", labeled_loss_mean.detach(), prog_bar=False, logger=True)
            self.log("loss_u_student_a", unlabeled_a_loss_mean.detach(), prog_bar=False, logger=True)
            self.log("loss_u_student_b_sam", unlabeled_b_loss_mean.detach(), prog_bar=False, logger=True)
        else:
            model_x = [image, gt]
            foreground_probability, background_probability, _, _ = self._forward_eval_model(model_x)
            # Text-free inference from the manuscript: average P_f and (1 - P_b).
            preds_metric = 0.5 * (foreground_probability + (1.0 - background_probability))
            loss = self.loss_fn(preds_metric, y)
            y_metric = y

        return {'loss': loss, 'preds': preds_metric.detach(), 'y': y_metric.detach()}


    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Update the teacher after the optimizer step, using both students.
        self.update_ema_variables_dual(self.trainer.global_step, self.ema_decay)

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self._log_outputs(outputs, "train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self._log_outputs(outputs, "val")
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self._log_outputs(outputs, "test")
        return outputs["loss"]

    def _log_outputs(self, outputs, stage):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)
        self.log(
            f"{stage}_loss",
            outputs["loss"],
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage != "test",
        )
        for name, metric in metrics.items():
            metric.update(outputs["preds"], outputs["y"].int())
            self.log(
                f"{stage}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=stage == "train" and name == "dice",
            )
