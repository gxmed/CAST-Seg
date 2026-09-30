"""
Train the Text-guided Semantic Localization (TSL) module.

Example:
    python train_tsl.py --config ./configs/tsl_qata.yaml

This script trains only the CAM generator:
    image + report text -> predicted CAM

The public configurations use ``labeled_ratio=0.04`` for the paper's 5% setting.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from utils.tsl_model import TSLGenerator, set_text_encoder_frozen
from utils.tsl_dataset import TSLCAMDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CAMLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, mse_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.mse_weight = float(mse_weight)
        self.dice_weight = float(dice_weight)

    @staticmethod
    def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = prob.flatten(1)
        target = target.flatten(1)
        inter = (prob * target).sum(dim=1)
        denom = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target.float().clamp(0.0, 1.0)
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        mse = F.mse_loss(prob, target)
        dice_loss = self.soft_dice_loss(prob, target)
        total = self.bce_weight * bce + self.mse_weight * mse + self.dice_weight * dice_loss
        return {
            "loss": total,
            "bce": bce.detach(),
            "mse": mse.detach(),
            "dice_loss": dice_loss.detach(),
        }


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()
    logs = {"loss": 0.0, "bce": 0.0, "mse": 0.0, "dice_loss": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out = model(
            image=batch["image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss_dict = criterion(out.cam_logits, batch["cam"])
        loss_dict["loss"].backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        bs = batch["image"].shape[0]
        n += bs
        for k in logs:
            value = loss_dict[k]
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            logs[k] += float(value) * bs

    return {k: v / max(n, 1) for k, v in logs.items()}


@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    logs = {"loss": 0.0, "bce": 0.0, "mse": 0.0, "dice_loss": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            image=batch["image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss_dict = criterion(out.cam_logits, batch["cam"])
        bs = batch["image"].shape[0]
        n += bs
        for k in logs:
            value = loss_dict[k]
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            logs[k] += float(value) * bs

    return {k: v / max(n, 1) for k, v in logs.items()}


def make_optimizer(model: TSLGenerator, train_cfg: Dict) -> torch.optim.Optimizer:
    """
    Text encoder is excluded because it is frozen.
    ConvNeXt gets a smaller lr by default.
    Decoder / lateral conv / FiLM get the main lr.
    """
    base_lr = float(train_cfg.get("lr", 1e-4))
    vision_lr = float(train_cfg.get("vision_lr", base_lr * 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    vision_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("text_encoder"):
            # Safety: do not optimize CXR-BERT.
            continue
        if name.startswith("vision_encoder"):
            vision_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if vision_params:
        param_groups.append({"params": vision_params, "lr": vision_lr, "name": "convnext_vision_encoder"})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "cam_decoder_and_fusion"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check freeze_vision/freeze_text settings.")

    print("Optimizer parameter groups:")
    for group in param_groups:
        print(f"  - {group['name']}: lr={group['lr']}, params={sum(p.numel() for p in group['params'])/1e6:.2f}M")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tsl_qata.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("TRAIN", {})
    model_cfg = cfg.get("MODEL", {})
    data_cfg = cfg.get("DATA", {})

    seed = int(cfg.get("seed", 2026))
    seed_everything(seed)

    device_ids = train_cfg.get("device", [0])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0] if isinstance(device_ids, list) else int(device_ids)}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ds_full = TSLCAMDataset(
        csv_path=data_cfg["train_csv_path"],
        root_path=data_cfg["train_root_path"],
        cam_root_path=data_cfg["train_cam_path"],
        bert_type=model_cfg["bert_type"],
        image_size=tuple(train_cfg.get("image_size", [224, 224])),
        text_max_length=int(train_cfg.get("text_max_length", 64)),
        prompt_template=train_cfg.get(
            "prompt_template",
            "Find the infected region in the chest X-ray according to the report: {report}",
        ),
        image_normalize=bool(train_cfg.get("image_normalize", True)),
        strict_cam=bool(train_cfg.get("strict_cam", True)),
        cam_split=train_cfg.get("cam_split", data_cfg.get("cam_split", "labeled")),
        labeled_ratio=float(train_cfg.get("labeled_ratio", data_cfg.get("labeled_ratio", 0.04))),
        valid_ratio=float(train_cfg.get("valid_ratio", data_cfg.get("valid_ratio", 0.2))),
    )

    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    n_val = max(1, int(len(ds_full) * val_ratio)) if len(ds_full) > 1 else 0
    n_train = len(ds_full) - n_val
    if n_train <= 0:
        raise RuntimeError(f"Dataset too small after split. len={len(ds_full)}, val_ratio={val_ratio}")

    generator = torch.Generator().manual_seed(seed)
    if n_val > 0:
        ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=generator)
    else:
        ds_train, ds_val = ds_full, ds_full

    loader_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=int(train_cfg.get("valid_batch_size", train_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    model = TSLGenerator(
        vision_type=model_cfg["vision_type"],
        bert_type=model_cfg["bert_type"],
        cam_dim=int(model_cfg.get("cam_dim", 256)),
        text_pooling=train_cfg.get("text_pooling", "mean"),
        target_size=tuple(train_cfg.get("image_size", [224, 224])),
        # Important requested setting:
        #   ConvNeXt has pretrained weights and is updated.
        #   CXR-BERT has pretrained weights and is frozen.
        freeze_vision=bool(train_cfg.get("freeze_vision", False)),
        freeze_text=bool(train_cfg.get("freeze_text", True)),
        convnext_channels=model_cfg.get("convnext_channels", [96, 192, 384, 768]),
    ).to(device)

    # Safety: always freeze CXR-BERT unless explicitly set freeze_text=False.
    if bool(train_cfg.get("freeze_text", True)):
        set_text_encoder_frozen(model)

    print("Trainable parameter summary:")
    print(f"  text_encoder trainable params   : {count_trainable(model.text_encoder) / 1e6:.2f}M")
    print(f"  vision_encoder trainable params : {count_trainable(model.vision_encoder) / 1e6:.2f}M")
    print(f"  total trainable params          : {count_trainable(model) / 1e6:.2f}M")

    optimizer = make_optimizer(model, train_cfg)
    criterion = CAMLoss(
        bce_weight=float(train_cfg.get("cam_bce_weight", 1.0)),
        mse_weight=float(train_cfg.get("cam_mse_weight", 0.5)),
        dice_weight=float(train_cfg.get("cam_dice_weight", 0.5)),
    )

    save_dir = Path(train_cfg.get("save_dir", "./save_model"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = train_cfg.get("save_name", "tsl_generator.ckpt")
    save_path = save_dir / save_name

    best_val = float("inf")
    patience = int(train_cfg.get("patience", 20))
    bad_epochs = 0

    for epoch in range(1, int(train_cfg.get("max_epochs", 100)) + 1):
        print(f"\nEpoch {epoch}")
        train_logs = train_one_epoch(
            model,
            loader_train,
            criterion,
            optimizer,
            device,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        )
        val_logs = validate(model, loader_val, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} val_bce={val_logs['bce']:.4f} "
            f"val_mse={val_logs['mse']:.4f} val_dice_loss={val_logs['dice_loss']:.4f}"
        )

        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "best_val_loss": best_val,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, save_path)
            print(f"  Saved best checkpoint: {save_path}")
        else:
            bad_epochs += 1
            print(f"  No improvement. bad_epochs={bad_epochs}/{patience}")
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print(f"Done. Best val loss={best_val:.4f}. Checkpoint={save_path}")


if __name__ == "__main__":
    main()
