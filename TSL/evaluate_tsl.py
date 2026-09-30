"""
Evaluate the Text-guided Semantic Localization (TSL) module.

Example:
    python evaluate_tsl.py --config ./configs/tsl_qata.yaml --eval_split validation

Notes:
    - MSE/MAE/BCE/Dice are computed between predicted CAM and CAM label.
    - The selected split must have CAM supervisory targets available.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from utils.tsl_model import TSLGenerator, set_text_encoder_frozen
from utils.tsl_dataset import TSLCAMDataset


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def soft_dice_score(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    return (2.0 * inter + eps) / (denom + eps)


def pearson_corr_map(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = prob.flatten(1)
    y = target.flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x.pow(2).sum(dim=1) + eps) * (y.pow(2).sum(dim=1) + eps))
    return num / den


def topk_hit(prob: torch.Tensor, target: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    """
    Fraction of top-ratio CAM pixels that fall inside target>0.5.
    target can be soft; threshold 0.5 is used here.
    """
    b = prob.shape[0]
    x = prob.flatten(1)
    y = (target.flatten(1) > 0.5).float()
    k = max(1, int(x.shape[1] * ratio))
    idx = torch.topk(x, k=k, dim=1).indices
    hits = torch.gather(y, dim=1, index=idx).mean(dim=1)
    return hits


def mass_in_target(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    CAM energy ratio inside target>0.5.
    """
    mask = (target > 0.5).float()
    inside = (prob * mask).flatten(1).sum(dim=1)
    total = prob.flatten(1).sum(dim=1).clamp_min(eps)
    return inside / total


def save_cam_outputs(prob: torch.Tensor, image_names: List[str], save_dir: Path) -> None:
    """Save predicted CAM as .npy and grayscale .png."""
    npy_dir = save_dir / "pred_npy"
    png_dir = save_dir / "pred_gray_png"
    npy_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    arr = prob.detach().cpu().numpy()
    for i, name in enumerate(image_names):
        stem = Path(str(name)).stem.replace("mask_", "")
        cam = arr[i, 0].astype(np.float32)
        cam = np.clip(cam, 0.0, 1.0)
        np.save(npy_dir / f"{stem}.npy", cam)
        gray = (cam * 255.0).round().astype(np.uint8)
        Image.fromarray(gray, mode="L").save(png_dir / f"{stem}.png")


def build_dataset(cfg: Dict, eval_split: str) -> TSLCAMDataset:
    train_cfg = cfg.get("TRAIN", {})
    model_cfg = cfg.get("MODEL", {})
    data_cfg = cfg.get("DATA", {})

    # For rest, load all first and slice manually below.
    dataset_split = "all" if eval_split == "rest" else eval_split

    # 评估优先用 test 路径，没有则 fallback 到 train 路径
    csv_path = data_cfg.get("test_csv_path", data_cfg["train_csv_path"])
    root_path = data_cfg.get("test_root_path", data_cfg["train_root_path"])

    ds = TSLCAMDataset(
        csv_path=csv_path,
        root_path=root_path,
        cam_root_path=data_cfg.get("train_cam_path", ""),
        bert_type=model_cfg["bert_type"],
        image_size=tuple(train_cfg.get("image_size", [224, 224])),
        text_max_length=int(train_cfg.get("text_max_length", 64)),
        prompt_template=train_cfg.get(
            "prompt_template",
            "Find the infected region in the chest X-ray according to the report: {report}",
        ),
        image_normalize=bool(train_cfg.get("image_normalize", True)),
        strict_cam=bool(train_cfg.get("strict_cam", True)),
        cam_split=dataset_split,
        labeled_ratio=float(train_cfg.get("labeled_ratio", data_cfg.get("labeled_ratio", 0.04))),
        valid_ratio=float(train_cfg.get("valid_ratio", data_cfg.get("valid_ratio", 0.2))),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    if eval_split == "rest":
        n_total = len(ds.df)
        labeled_ratio = float(train_cfg.get("labeled_ratio", data_cfg.get("labeled_ratio", 0.04)))
        n_labeled = int(n_total * labeled_ratio)
        ds.df = ds.df.iloc[n_labeled:].reset_index(drop=True)
        print(f"[Eval split] rest, samples={len(ds.df)} / total={n_total}")

    return ds


def build_model(cfg: Dict, device: torch.device) -> TSLGenerator:
    train_cfg = cfg.get("TRAIN", {})
    model_cfg = cfg.get("MODEL", {})
    model = TSLGenerator(
        vision_type=model_cfg["vision_type"],
        bert_type=model_cfg["bert_type"],
        cam_dim=int(model_cfg.get("cam_dim", 256)),
        text_pooling=train_cfg.get("text_pooling", "mean"),
        target_size=tuple(train_cfg.get("image_size", [224, 224])),
        freeze_vision=bool(train_cfg.get("freeze_vision", False)),
        freeze_text=bool(train_cfg.get("freeze_text", True)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        convnext_channels=model_cfg.get("convnext_channels", [96, 192, 384, 768]),
    )
    if bool(train_cfg.get("freeze_text", True)):
        set_text_encoder_frozen(model)
    return model.to(device)


def load_checkpoint(model: TSLGenerator, ckpt_path: str, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if "epoch" in ckpt:
        print(f"  epoch={ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss')}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


@torch.no_grad()
def evaluate(model: TSLGenerator, loader: DataLoader, device: torch.device, save_pred_dir: Path = None) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    sums = {
        "mse": 0.0,
        "mae": 0.0,
        "bce": 0.0,
        "dice_score": 0.0,
        "dice_loss": 0.0,
        "pearson": 0.0,
        "top1_hit": 0.0,
        "top5_hit": 0.0,
        "top10_hit": 0.0,
        "mass_in_target": 0.0,
    }
    rows: List[Dict[str, object]] = []
    n = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        out = model(
            image=batch["image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = out.cam_logits
        prob = torch.sigmoid(logits).clamp(0.0, 1.0)
        target = batch["cam"].float().clamp(0.0, 1.0)

        bs = prob.shape[0]
        per_mse = F.mse_loss(prob, target, reduction="none").flatten(1).mean(dim=1)
        per_mae = (prob - target).abs().flatten(1).mean(dim=1)
        per_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(dim=1)
        per_dice = soft_dice_score(prob, target)
        per_pearson = pearson_corr_map(prob, target)
        per_top1 = topk_hit(prob, target, ratio=0.01)
        per_top5 = topk_hit(prob, target, ratio=0.05)
        per_top10 = topk_hit(prob, target, ratio=0.10)
        per_mass = mass_in_target(prob, target)

        metric_tensors = {
            "mse": per_mse,
            "mae": per_mae,
            "bce": per_bce,
            "dice_score": per_dice,
            "dice_loss": 1.0 - per_dice,
            "pearson": per_pearson,
            "top1_hit": per_top1,
            "top5_hit": per_top5,
            "top10_hit": per_top10,
            "mass_in_target": per_mass,
        }
        for k, v in metric_tensors.items():
            sums[k] += float(v.sum().item())

        image_names = batch["image_name"]
        cam_paths = batch["cam_path"]
        if save_pred_dir is not None:
            save_cam_outputs(prob, list(image_names), save_pred_dir)

        for i in range(bs):
            row = {
                "image_name": image_names[i],
                "cam_path": cam_paths[i],
            }
            for k, v in metric_tensors.items():
                row[k] = float(v[i].detach().cpu().item())
            rows.append(row)

        n += bs
        if step % 20 == 0:
            print(f"Evaluated {n} samples...")

    summary = {k: v / max(n, 1) for k, v in sums.items()}
    summary["num_samples"] = float(n)
    return summary, rows


def save_results(summary: Dict[str, float], rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary_metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    csv_path = out_dir / "per_sample_metrics.csv"
    fieldnames = [
        "image_name",
        "cam_path",
        "mse",
        "mae",
        "bce",
        "dice_score",
        "dice_loss",
        "pearson",
        "top1_hit",
        "top5_hit",
        "top10_hit",
        "mass_in_target",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-sample csv: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tsl_qata.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["labeled", "unlabeled", "validation", "rest", "all"],
        help="Which part of Train.csv to evaluate. rest excludes the labeled subset.",
    )
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./cam_eval_results")
    parser.add_argument("--save_pred", action="store_true", help="Save predicted CAM npy and grayscale png.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("TRAIN", {})

    device_ids = train_cfg.get("device", [0])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0] if isinstance(device_ids, list) else int(device_ids)}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        save_dir = Path(train_cfg.get("save_dir", "./save_model"))
        save_name = train_cfg.get("save_name", "tsl_generator.ckpt")
        ckpt_path = str(save_dir / save_name)

    ds = build_dataset(cfg, eval_split=args.eval_split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size if args.batch_size > 0 else int(train_cfg.get("valid_batch_size", train_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers >= 0 else int(train_cfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg, device)
    load_checkpoint(model, ckpt_path, device)

    out_dir = Path(args.out_dir) / args.eval_split
    save_pred_dir = out_dir if args.save_pred else None
    summary, rows = evaluate(model, loader, device, save_pred_dir=save_pred_dir)

    print("\nEvaluation summary:")
    for k, v in summary.items():
        if k == "num_samples":
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.6f}")

    save_results(summary, rows, out_dir)


if __name__ == "__main__":
    main()
