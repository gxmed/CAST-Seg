import json
import os
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from scipy import ndimage

import utils.config as config
from data.dataset import CASTSegDataset
from engine.cast_seg import CASTSegModule
from train import load_castseg_checkpoint


def cfg_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes", "y"]
    return default


def get_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate CAST-Seg with CAM-guided decoder-conflict arbitration'
    )
    parser.add_argument('--config', default='./config/qata.yaml', type=str)
    parser.add_argument('--ckpt', default=None, type=str, help='optional explicit checkpoint path')
    parser.add_argument(
        '--cam-threshold',
        default=None,
        type=float,
        help='override eval_cam_threshold in yaml',
    )
    parser.add_argument(
        '--fg-threshold',
        default=None,
        type=float,
        help='override eval_fg_threshold in yaml',
    )
    parser.add_argument(
        '--bg-threshold',
        default=None,
        type=float,
        help='override eval_bg_threshold in yaml',
    )
    parser.add_argument(
        '--disable-cam-conflict-fusion',
        action='store_true',
        help='fall back to the original Lightning test_step',
    )
    parser.add_argument(
        '--output-json',
        default=None,
        type=str,
        help='optional path to write evaluation metrics as JSON',
    )

    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)

    if cli_args.ckpt is not None and str(cli_args.ckpt).strip() != "":
        cfg.eval_ckpt_path = cli_args.ckpt
    if cli_args.cam_threshold is not None:
        cfg.eval_cam_threshold = float(cli_args.cam_threshold)
    if cli_args.fg_threshold is not None:
        cfg.eval_fg_threshold = float(cli_args.fg_threshold)
    if cli_args.bg_threshold is not None:
        cfg.eval_bg_threshold = float(cli_args.bg_threshold)
    if cli_args.disable_cam_conflict_fusion:
        cfg.eval_cam_conflict_fusion = False
    if cli_args.output_json is not None and str(cli_args.output_json).strip() != "":
        cfg.eval_output_json = cli_args.output_json

    return cfg


def build_dataset(args, csv_path, root_path, mode):
    """
    CAM is loaded only for validation/test-time conflict arbitration.
    SAM is not required by this inference fusion rule.
    """
    use_cam_prior = False
    use_sam_prior = False
    cam_prior_path = None
    sam_prior_path = None

    if mode == 'test':
        use_conflict_fusion = cfg_bool(
            getattr(args, "eval_cam_conflict_fusion", True), True
        )
        test_cam_path = getattr(args, "test_cam_prior_path", None)
        test_sam_path = getattr(args, "test_sam_prior_path", None)

        if use_conflict_fusion:
            if test_cam_path is None or str(test_cam_path).strip() == "":
                raise ValueError(
                    "CAM-guided conflict fusion is enabled, but "
                    "DATA.test_cam_prior_path / test_cam_prior_path is empty."
                )
            use_cam_prior = True
            cam_prior_path = test_cam_path
        elif test_cam_path is not None and str(test_cam_path).strip() != "":
            use_cam_prior = cfg_bool(
                getattr(args, "eval_with_cam_prior", False), False
            )
            cam_prior_path = test_cam_path

        if test_sam_path is not None and str(test_sam_path).strip() != "":
            use_sam_prior = cfg_bool(
                getattr(args, "eval_with_sam_prior", False), False
            )
            sam_prior_path = test_sam_path

    return CASTSegDataset(
        csv_path=csv_path,
        root_path=root_path,
        tokenizer=None,
        image_size=args.image_size,
        mode=mode,
        text_column=str(getattr(args, "text_column", "Description")),
        labeled_ratio=float(getattr(args, "labeled_ratio", 0.04)),
        valid_ratio=float(getattr(args, "valid_ratio", 0.2)),
        use_cam_prior=use_cam_prior,
        cam_prior_path=cam_prior_path,
        strict_cam_prior=use_cam_prior,
        use_sam_prior=use_sam_prior,
        sam_prior_path=sam_prior_path,
        strict_sam_prior=False,
        use_spatial_aug=False,
    )


def get_checkpoint_path(args):
    explicit = getattr(args, "eval_ckpt_path", None)
    if explicit is not None and str(explicit).strip() != "":
        return explicit

    filename = getattr(args, "semi_save_filename", "cast_seg")
    if not str(filename).endswith(".ckpt"):
        filename = str(filename) + ".ckpt"
    return os.path.join(
        getattr(args, "semi_save_path", "./semi_supervised"),
        filename,
    )


def _is_image_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.ndim == 4 and x.shape[1] in (1, 3)


def _is_spatial_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.ndim in (3, 4)


def unpack_eval_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Supports the common CASTSegDataset layouts:

      1) ([image, placeholder_gt], gt, flag, cam, sam)
      2) ([image, placeholder_gt, cam, sam], gt, flag)
      3) {"image": ..., "gt": ..., "cam": ...}

    Returns image, ground-truth mask and CAM.
    """
    if isinstance(batch, dict):
        image = next(
            (batch[k] for k in ("image", "images", "x") if k in batch),
            None,
        )
        gt = next(
            (batch[k] for k in ("gt", "mask", "label", "target") if k in batch),
            None,
        )
        cam = next(
            (batch[k] for k in ("cam", "cam_prior", "cam_mask") if k in batch),
            None,
        )
    elif isinstance(batch, (list, tuple)):
        image = None
        gt = None
        cam = None

        first = batch[0] if len(batch) > 0 else None
        if isinstance(first, dict):
            return unpack_eval_batch(first)

        if isinstance(first, (list, tuple)):
            if len(first) > 0 and _is_image_tensor(first[0]):
                image = first[0]
            if len(first) > 1 and _is_spatial_tensor(first[1]):
                gt = first[1]
            # Some datasets put CAM inside model inputs.
            if len(first) > 2 and _is_spatial_tensor(first[2]):
                cam = first[2]
        elif _is_image_tensor(first):
            image = first

        # The second item is normally the actual GT.
        if len(batch) > 1 and _is_spatial_tensor(batch[1]):
            gt = batch[1]

        # In CASTSegDataset, flag normally occupies index 2, CAM index 3.
        if len(batch) > 3 and _is_spatial_tensor(batch[3]):
            cam = batch[3]

        # Fallback: locate spatial tensors outside the model-input container.
        if cam is None:
            candidates = [
                item for idx, item in enumerate(batch)
                if idx >= 2 and _is_spatial_tensor(item)
            ]
            if candidates:
                cam = candidates[0]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    if image is None:
        raise RuntimeError("Could not extract image tensor from the evaluation batch.")
    if gt is None:
        raise RuntimeError("Could not extract GT tensor from the evaluation batch.")
    if cam is None:
        raise RuntimeError(
            "Could not extract CAM tensor from the evaluation batch. "
            "Check CASTSegDataset.__getitem__ and test_cam_prior_path."
        )

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if cam.ndim == 3:
        cam = cam.unsqueeze(1)

    return image, gt, cam


def resolve_eval_network(wrapper: CASTSegModule, eval_model: str):
    """
    Resolve the requested CAST-Seg inference network.
    The first two outputs of the resolved network must be foreground and
    background predictions.
    """
    name = str(eval_model).strip().lower()

    aliases: Dict[str, Sequence[str]] = {
        "teacher": (
            "teacher_model",
            "ema_model",
            "ema_teacher",
            "teacher",
            "model_teacher",
        ),
        "student_a": (
            "student_a",
            "student_model_a",
            "model_a",
            "student1",
            "student_1",
            "model",
        ),
        "student_b": (
            "student_b",
            "student_model_b",
            "model_b",
            "student2",
            "student_2",
        ),
    }

    # Accept common alternative spellings.
    if name in ("ema", "ema_teacher"):
        name = "teacher"
    elif name in ("a", "student1", "student_1"):
        name = "student_a"
    elif name in ("b", "student2", "student_2"):
        name = "student_b"

    for attr_name in aliases.get(name, (name,)):
        candidate = getattr(wrapper, attr_name, None)
        if candidate is not None and callable(candidate):
            return candidate, attr_name

    # Wrapper may expose a selector method.
    for selector_name in ("get_eval_model", "select_eval_model", "_get_eval_model"):
        selector = getattr(wrapper, selector_name, None)
        if callable(selector):
            candidate = selector(name)
            if candidate is not None and callable(candidate):
                return candidate, selector_name

    visible = [
        key for key, value in vars(wrapper).items()
        if callable(value) or isinstance(value, torch.nn.Module)
    ]
    raise AttributeError(
        f"Cannot resolve eval_model={eval_model!r}. "
        f"Available module-like attributes: {visible}"
    )


def call_segmentation_network(
    network: torch.nn.Module,
    image: torch.Tensor,
    gt: torch.Tensor,
) -> Any:
    """Call the underlying model while supporting common project signatures."""
    errors = []
    for model_input in ([image, gt], (image, gt), image):
        try:
            return network(model_input)
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{type(model_input).__name__}: {exc}")

    raise RuntimeError(
        "Failed to call the selected segmentation network. Tried [image, gt], "
        f"(image, gt), and image. Errors: {errors}"
    )


def extract_fg_bg_predictions(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract foreground and background maps from tuple/dict/object outputs."""
    if isinstance(output, dict):
        fg = next(
            (
                output[k]
                for k in ("fg_output", "foreground", "fg", "preds", "pred")
                if k in output
            ),
            None,
        )
        bg = next(
            (
                output[k]
                for k in ("bg_output", "background", "bg", "bg_preds", "neg_pred")
                if k in output
            ),
            None,
        )
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        fg, bg = output[0], output[1]
    else:
        fg = next(
            (
                getattr(output, k)
                for k in ("fg_output", "foreground", "fg", "preds", "pred")
                if hasattr(output, k)
            ),
            None,
        )
        bg = next(
            (
                getattr(output, k)
                for k in ("bg_output", "background", "bg", "bg_preds")
                if hasattr(output, k)
            ),
            None,
        )

    if not torch.is_tensor(fg) or not torch.is_tensor(bg):
        raise RuntimeError(
            "The selected network did not return recognizable foreground and "
            "background tensors. Expected the first two tuple elements or "
            "fg_output/bg_output-style keys."
        )

    if fg.ndim == 3:
        fg = fg.unsqueeze(1)
    if bg.ndim == 3:
        bg = bg.unsqueeze(1)
    return fg, bg


def to_probability(x: torch.Tensor) -> torch.Tensor:
    """Keep sigmoid outputs unchanged; apply sigmoid only when logits are detected."""
    if x.detach().min().item() < 0.0 or x.detach().max().item() > 1.0:
        return torch.sigmoid(x)
    return x.clamp(0.0, 1.0)


def cam_guided_conflict_fusion(
    p_fg: torch.Tensor,
    p_bg: torch.Tensor,
    cam: torch.Tensor,
    fg_threshold: float = 0.5,
    bg_threshold: float = 0.5,
    cam_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-threshold CAM arbitration used only during evaluation.

    fg_vote=1, bg_vote=0 -> foreground (decoder agreement)
    fg_vote=0, bg_vote=1 -> background (decoder agreement)
    fg_vote=bg_vote       -> CAM > cam_threshold decides foreground/background

    Here bg_vote=1 means that the background decoder predicts background.
    """
    if p_fg.shape[-2:] != p_bg.shape[-2:]:
        p_bg = F.interpolate(
            p_bg,
            size=p_fg.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    if cam.shape[-2:] != p_fg.shape[-2:]:
        cam = F.interpolate(
            cam.float(),
            size=p_fg.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    cam = cam.float().clamp(0.0, 1.0)
    fg_vote = p_fg > fg_threshold
    bg_vote = p_bg > bg_threshold

    agreed_foreground = fg_vote & (~bg_vote)
    conflict = fg_vote == bg_vote  # includes (1,1) and (0,0)
    cam_foreground = cam > cam_threshold

    final_mask = agreed_foreground | (conflict & cam_foreground)
    return final_mask.float(), conflict.float()


def _binary_metrics_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection
    total = torch.tensor(pred.shape[1], device=pred.device, dtype=pred.dtype)
    fp = pred_sum - intersection
    fn = target_sum - intersection
    tn = total - intersection - fp - fn

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    acc = (pred == target).float().mean(dim=1)
    return dice, iou, acc, intersection, pred_sum, target_sum, tn, fp


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if not mask_a.any() and not mask_b.any():
        return np.array([0.0], dtype=np.float64)
    if not mask_a.any() or not mask_b.any():
        return np.array([], dtype=np.float64)

    structure = ndimage.generate_binary_structure(mask_a.ndim, 1)
    surface_a = mask_a ^ ndimage.binary_erosion(mask_a, structure=structure, border_value=0)
    surface_b = mask_b ^ ndimage.binary_erosion(mask_b, structure=structure, border_value=0)

    distances_to_b = ndimage.distance_transform_edt(~surface_b)
    distances_to_a = ndimage.distance_transform_edt(~surface_a)
    return np.concatenate([distances_to_b[surface_a], distances_to_a[surface_b]]).astype(np.float64)


def _hd95_assd_per_image(pred: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)
    hd95_values = []
    assd_values = []

    for pred_mask, target_mask in zip(pred_np, target_np):
        pred_mask = np.squeeze(pred_mask)
        target_mask = np.squeeze(target_mask)
        distances = _surface_distances(pred_mask, target_mask)
        if distances.size == 0:
            hd95_values.append(np.nan)
            assd_values.append(np.nan)
        else:
            hd95_values.append(float(np.percentile(distances, 95)))
            assd_values.append(float(np.mean(distances)))

    return np.asarray(hd95_values, dtype=np.float64), np.asarray(assd_values, dtype=np.float64)


def _nanmean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _json_safe_metrics(metrics: Dict[str, float]) -> Dict[str, Optional[float]]:
    safe = {}
    for key, value in metrics.items():
        value = float(value)
        safe[key] = None if not np.isfinite(value) else value
    return safe


@torch.no_grad()
def evaluate_with_cam_conflict_fusion(
    wrapper: CASTSegModule,
    dataloader: DataLoader,
    device: torch.device,
    eval_model: str,
    fg_threshold: float,
    bg_threshold: float,
    cam_threshold: float,
) -> Dict[str, float]:
    wrapper.to(device)
    wrapper.eval()

    network, resolved_name = resolve_eval_network(wrapper, eval_model)
    network.to(device)
    network.eval()
    print(f"Resolved evaluation network: {resolved_name}")

    dice_values = []
    iou_values = []
    acc_values = []
    hd95_values = []
    assd_values = []
    sensitivity_values = []
    specificity_values = []
    global_intersection = 0.0
    global_pred_sum = 0.0
    global_target_sum = 0.0
    global_tn = 0.0
    global_fp = 0.0
    conflict_pixels = 0.0
    total_pixels = 0.0
    conflict_cam_fg_pixels = 0.0

    for batch_idx, batch in enumerate(dataloader):
        image, gt, cam = unpack_eval_batch(batch)
        image = image.to(device, non_blocking=True).float()
        gt = gt.to(device, non_blocking=True).float()
        cam = cam.to(device, non_blocking=True).float()

        output = call_segmentation_network(network, image, gt)
        fg_output, bg_output = extract_fg_bg_predictions(output)
        p_fg = to_probability(fg_output)
        p_bg = to_probability(bg_output)

        final_mask, conflict = cam_guided_conflict_fusion(
            p_fg=p_fg,
            p_bg=p_bg,
            cam=cam,
            fg_threshold=fg_threshold,
            bg_threshold=bg_threshold,
            cam_threshold=cam_threshold,
        )

        if gt.shape[-2:] != final_mask.shape[-2:]:
            gt = F.interpolate(gt, size=final_mask.shape[-2:], mode='nearest')
        gt = (gt > 0.5).float()

        pred_binary = (final_mask > 0.5).float()
        target_binary = (gt > 0.5).float()

        dice, iou, acc, inter, p_sum, t_sum, tn, fp = _binary_metrics_per_image(pred_binary, target_binary)
        dice_values.append(dice.cpu())
        iou_values.append(iou.cpu())
        acc_values.append(acc.cpu())
        global_intersection += inter.sum().item()
        global_pred_sum += p_sum.sum().item()
        global_target_sum += t_sum.sum().item()
        global_tn += tn.sum().item()
        global_fp += fp.sum().item()

        fn = t_sum - inter
        sensitivity = (inter + 1e-7) / (inter + fn + 1e-7)
        specificity = (tn + 1e-7) / (tn + fp + 1e-7)
        sensitivity_values.extend(sensitivity.detach().cpu().tolist())
        specificity_values.extend(specificity.detach().cpu().tolist())

        batch_hd95, batch_assd = _hd95_assd_per_image(pred_binary, target_binary)
        hd95_values.extend(batch_hd95.tolist())
        assd_values.extend(batch_assd.tolist())

        cam_resized = cam
        if cam.shape[-2:] != conflict.shape[-2:]:
            cam_resized = F.interpolate(
                cam,
                size=conflict.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        conflict_pixels += conflict.sum().item()
        total_pixels += float(conflict.numel())
        conflict_cam_fg_pixels += (
            (conflict > 0.5) & (cam_resized > cam_threshold)
        ).sum().item()

        if (batch_idx + 1) % 20 == 0:
            running_dice = torch.cat(dice_values).mean().item()
            print(
                f"[{batch_idx + 1}/{len(dataloader)}] "
                f"running Dice={running_dice:.6f}"
            )

    if not dice_values:
        raise RuntimeError("The test dataloader produced no batches.")

    dice = torch.cat(dice_values).mean().item()
    iou = torch.cat(iou_values).mean().item()
    acc = torch.cat(acc_values).mean().item()
    eps = 1e-7
    global_dice = (2.0 * global_intersection + eps) / (global_pred_sum + global_target_sum + eps)
    global_iou = (global_intersection + eps) / (global_pred_sum + global_target_sum - global_intersection + eps)
    global_sensitivity = (global_intersection + eps) / (global_target_sum + eps)
    global_specificity = (global_tn + eps) / (global_tn + global_fp + eps)
    conflict_ratio = conflict_pixels / max(total_pixels, 1.0)
    cam_fg_in_conflict_ratio = conflict_cam_fg_pixels / max(conflict_pixels, 1.0)

    return {
        "test_dice": dice,
        "test_MIoU": iou,
        "test_acc": acc,
        "test_dice_global": global_dice,
        "test_MIoU_global": global_iou,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "mean_dice": dice,
        "mean_iou": iou,
        "hd95": _nanmean(hd95_values),
        "assd": _nanmean(assd_values),
        "sensitivity": _nanmean(sensitivity_values),
        "specificity": _nanmean(specificity_values),
        "global_sensitivity": global_sensitivity,
        "global_specificity": global_specificity,
        "conflict_pixel_ratio": conflict_ratio,
        "cam_foreground_ratio_in_conflict": cam_fg_in_conflict_ratio,
    }


if __name__ == '__main__':
    args = get_parser()
    ckpt_path = get_checkpoint_path(args)

    model = CASTSegModule(args)
    load_castseg_checkpoint(model, ckpt_path, strict=False)

    eval_model = getattr(args, 'eval_model', 'teacher')
    use_conflict_fusion = cfg_bool(
        getattr(args, "eval_cam_conflict_fusion", True), True
    )
    fg_threshold = float(getattr(args, "eval_fg_threshold", 0.5))
    bg_threshold = float(getattr(args, "eval_bg_threshold", 0.5))
    cam_threshold = float(getattr(args, "eval_cam_threshold", 0.2))

    print(f"Loaded CAST-Seg checkpoint: {ckpt_path}")
    print(f"Evaluation model: {eval_model}")

    ds_test = build_dataset(args, args.test_csv_path, args.test_root_path, mode='test')
    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=int(getattr(args, "eval_num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
    )

    if not use_conflict_fusion:
        print("CAM conflict fusion: disabled; using the original wrapper.test_step().")
        trainer = pl.Trainer(accelerator='gpu', devices=args.device)
        model.eval()
        trainer.test(model, dl_test)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the current evaluation configuration.")

        device_ids = getattr(args, "device", [0])
        if isinstance(device_ids, int):
            device_id = device_ids
        else:
            device_id = int(device_ids[0])
        device = torch.device(f"cuda:{device_id}")

        print("CAM conflict fusion: enabled")
        print(f"Foreground threshold: {fg_threshold}")
        print(f"Background threshold: {bg_threshold}")
        print(f"CAM threshold: {cam_threshold}")
        print("Rule: decoder agreement is retained; CAM resolves only conflict pixels.")

        metrics = evaluate_with_cam_conflict_fusion(
            wrapper=model,
            dataloader=dl_test,
            device=device,
            eval_model=eval_model,
            fg_threshold=fg_threshold,
            bg_threshold=bg_threshold,
            cam_threshold=cam_threshold,
        )

        print("\nEvaluation summary:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.8f}")

        output_json = getattr(args, "eval_output_json", None)
        if output_json is not None and str(output_json).strip() != "":
            output_json = str(output_json)
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(_json_safe_metrics(metrics), f, indent=2, sort_keys=True)
            print(f"Metrics JSON written to: {output_json}")
