import os
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import utils.config as config
from data.dataset import CASTSegDataset
from data.pretrain_dataset import CASTSegPretrainDataset
from engine.cast_seg import CASTSegModule
from engine.pretrain import CASTSegPretrainModule
from modules.asr import AdaptiveSemanticRefinementSchedule


class ETACallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        epochs_remaining = trainer.max_epochs - trainer.current_epoch - 1
        if epochs_remaining > 0:
            eta_seconds = avg_epoch_time * epochs_remaining
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(f"  [Time ETA] Epoch {trainer.current_epoch} finished. Estimated time remaining: {eta_string}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser = argparse.ArgumentParser(description='Train CAST-Seg')
    parser.add_argument('--config', default='./config/qata.yaml', type=str, help='config file')
    args = parser.parse_args()
    assert args.config is not None
    return config.load_cfg_from_cfg_file(args.config)


def build_dataset(args, csv_path, root_path, mode):
    use_cam_prior = cfg_bool(getattr(args, "use_cam_prior", False), default=False)
    use_sam_prior = cfg_bool(getattr(args, "use_sam_prior", False), default=False)

    cam_prior_path = getattr(args, "cam_prior_path", None)
    sam_prior_path = getattr(args, "sam_prior_path", None)

    if mode == 'test':
        test_cam_path = getattr(args, "test_cam_prior_path", None)
        test_sam_path = getattr(args, "test_sam_prior_path", None)
        if test_cam_path is not None and str(test_cam_path).strip() != "":
            cam_prior_path = test_cam_path
        else:
            use_cam_prior = False
        if test_sam_path is not None and str(test_sam_path).strip() != "":
            sam_prior_path = test_sam_path
        else:
            use_sam_prior = False

    # TSL/MDAA priors are precomputed, so tokenization is unnecessary here.
    tokenizer = None if (use_cam_prior or use_sam_prior) else getattr(args, "bert_type", None)

    return CASTSegDataset(
        csv_path=csv_path,
        root_path=root_path,
        tokenizer=tokenizer,
        image_size=args.image_size,
        mode=mode,
        text_column=str(getattr(args, "text_column", "Description")),
        labeled_ratio=float(getattr(args, "labeled_ratio", 0.04)),
        valid_ratio=float(getattr(args, "valid_ratio", 0.2)),
        fold_index=getattr(args, "fold_index", None),
        num_folds=int(getattr(args, "num_folds", 5)),
        use_cam_prior=use_cam_prior,
        cam_prior_path=cam_prior_path,
        cam_prior_min_value=float(getattr(args, "cam_prior_min_value", 0.05)),
        cam_prior_zero_eps=float(getattr(args, "cam_prior_zero_eps", 0.0)),
        strict_cam_prior=cfg_bool(getattr(args, "strict_cam_prior", True), default=True),
        use_sam_prior=use_sam_prior,
        sam_prior_path=sam_prior_path,
        strict_sam_prior=cfg_bool(getattr(args, "strict_sam_prior", True), default=True),
        sam_prior_min_value=float(getattr(args, "sam_prior_min_value", 0.0)),
        sam_prior_zero_eps=float(getattr(args, "sam_prior_zero_eps", 0.0)),
        use_spatial_aug=cfg_bool(getattr(args, "use_spatial_aug", False), default=False),
        spatial_aug_prob=float(getattr(args, "spatial_aug_prob", 0.5)),
        spatial_aug_hflip=cfg_bool(getattr(args, "spatial_aug_hflip", True), default=True),
        spatial_aug_vflip=cfg_bool(getattr(args, "spatial_aug_vflip", False), default=False),
        spatial_aug_rotate=cfg_bool(getattr(args, "spatial_aug_rotate", True), default=True),
        spatial_aug_max_angle=float(getattr(args, "spatial_aug_max_angle", 10.0)),
    )


def build_pretrain_dataset(args, mode):
    return CASTSegPretrainDataset(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        image_size=args.image_size,
        mode=mode,
        labeled_ratio=float(getattr(args, "labeled_ratio", 0.04)),
        valid_ratio=float(getattr(args, "valid_ratio", 0.2)),
        fold_index=getattr(args, "fold_index", None),
        num_folds=int(getattr(args, "num_folds", 5)),
        use_cam_prior=cfg_bool(getattr(args, "use_cam_prior", False), default=False) and mode == "pretrain",
        cam_prior_path=getattr(args, "cam_prior_path", None),
        strict_cam_prior=cfg_bool(getattr(args, "strict_cam_prior", True), default=True),
        cam_prior_min_value=float(getattr(args, "cam_prior_min_value", 0.0)),
        cam_prior_zero_eps=float(getattr(args, "cam_prior_zero_eps", 0.0)),
        use_spatial_aug=cfg_bool(getattr(args, "use_spatial_aug", False), default=False) and mode == "pretrain",
        spatial_aug_prob=float(getattr(args, "spatial_aug_prob", 0.5)),
        spatial_aug_hflip=cfg_bool(getattr(args, "spatial_aug_hflip", True), default=True),
        spatial_aug_vflip=cfg_bool(getattr(args, "spatial_aug_vflip", False), default=False),
        spatial_aug_rotate=cfg_bool(getattr(args, "spatial_aug_rotate", True), default=True),
        spatial_aug_max_angle=float(getattr(args, "spatial_aug_max_angle", 10.0)),
    )


def get_pretrained_ckpt_path(args):
    ckpt_path = getattr(args, "pretrained_ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).strip() != "":
        return ckpt_path
    filename = args.model_save_filename
    if not str(filename).endswith(".ckpt"):
        filename = str(filename) + ".ckpt"
    return os.path.join(args.model_save_path, filename)


def _strip_prefix_state_dict(state_dict, prefix):
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _migrate_legacy_network_names(state_dict):
    migrated = {}
    for key, value in state_dict.items():
        key = key.replace("expert_fg.", "foreground_decoder.")
        key = key.replace("expert_bg.", "background_decoder.")
        if ".moe_gate." in key or ".up1." in key or key.startswith(("moe_gate.", "up1.")):
            continue
        migrated[key] = value
    return migrated


def load_pretrained_segmentation_into_castseg(model, ckpt_path):
    """
    Load a supervised segmentation checkpoint into Student-A/B and Teacher.

    The old checkpoint usually contains keys like:
      model.encoder.*
      model.expert_fg.*
      model.expert_bg.*

    CAST-Seg needs the same initialization in all three networks.
    """
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    if any(k.startswith("student_b.") for k in state_dict.keys()):
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[CAST-Seg load] Missing keys ignored: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[CAST-Seg load] Unexpected keys ignored: {len(incompatible.unexpected_keys)}")
        print(f"Loaded CAST-Seg checkpoint: {ckpt_path}")
        return

    if any(k.startswith("student_sam.") for k in state_dict.keys()):
        legacy_prefixes = {
            "model.": "student_a.",
            "student_sam.": "student_b.",
            "ema_model.": "teacher.",
        }
        migrated = {}
        for key, value in state_dict.items():
            for old_prefix, new_prefix in legacy_prefixes.items():
                if key.startswith(old_prefix):
                    key = new_prefix + key[len(old_prefix):]
                    break
            migrated[key] = value
        model.load_state_dict(_migrate_legacy_network_names(migrated), strict=False)
        print(f"Loaded legacy checkpoint with CAST-Seg parameter names: {ckpt_path}")
        return

    if any(k.startswith("model.") for k in state_dict.keys()):
        seg_state = _strip_prefix_state_dict(state_dict, "model.")
    else:
        seg_state = state_dict
    seg_state = _migrate_legacy_network_names(seg_state)

    miss_a, unexp_a = model.student_a.load_state_dict(seg_state, strict=False)
    miss_b, unexp_b = model.student_b.load_state_dict(seg_state, strict=False)
    miss_t, unexp_t = model.teacher.load_state_dict(seg_state, strict=False)

    bad = list(miss_a) + list(miss_b) + list(miss_t)
    if bad:
        print(f"[CAST-Seg load] Warning: missing segmentation keys: {len(bad)}")
        print("  First missing keys:", bad[:10])
    unexpected = list(unexp_a) + list(unexp_b) + list(unexp_t)
    if unexpected:
        print(f"[CAST-Seg load] Warning: unexpected segmentation keys: {len(unexpected)}")
        print("  First unexpected keys:", unexpected[:10])

    print(f"Loaded pretrained segmentation weights into Student-A, Student-B and Teacher: {ckpt_path}")


def load_castseg_checkpoint(model, ckpt_path, strict=True):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if any(key.startswith("student_sam.") for key in state_dict) and not any(
        key.startswith("student_b.") for key in state_dict
    ):
        legacy_prefixes = {
            "model.": "student_a.",
            "student_sam.": "student_b.",
            "ema_model.": "teacher.",
        }
        migrated = {}
        for key, value in state_dict.items():
            for old_prefix, new_prefix in legacy_prefixes.items():
                if key.startswith(old_prefix):
                    key = new_prefix + key[len(old_prefix):]
                    break
            migrated[key] = value
        state_dict = _migrate_legacy_network_names(migrated)
        strict = False
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"Missing keys ignored: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys ignored: {len(incompatible.unexpected_keys)}")
    print(f"Loaded CAST-Seg checkpoint: {ckpt_path}")


# -----------------------------------------------------------------------------
# Adaptive Semantic Refinement (ASR) utilities
# -----------------------------------------------------------------------------

def _resolve_module_by_name(root_module, module_path):
    """Resolve dotted module path, supporting numeric Sequential indices."""
    module = root_module
    for part in str(module_path).split('.'):
        if part == '':
            continue
        if part.isdigit():
            module = module[int(part)]
        else:
            if not hasattr(module, part):
                raise AttributeError(f"Cannot resolve target layer '{module_path}'. Missing part: {part}")
            module = getattr(module, part)
    return module


def _normalize_cam_batch(cam, eps=1e-6):
    """Per-sample min-max normalize CAM tensor [B,1,H,W] into [0,1]."""
    b = cam.shape[0]
    flat = cam.view(b, -1)
    min_v = flat.min(dim=1)[0].view(b, 1, 1, 1)
    max_v = flat.max(dim=1)[0].view(b, 1, 1, 1)
    cam = (cam - min_v) / (max_v - min_v + eps)
    return cam.clamp(0.0, 1.0)


def _load_existing_cam_for_blend(dataset, image_name, device, dtype):
    """Load current CAM from dataset path for EMA-style dynamic CAM smoothing."""
    try:
        cam = dataset._load_cam_as_tensor(image_name)
        if cam is None:
            return None
        cam = cam.unsqueeze(0).to(device=device, dtype=dtype)
        return cam.clamp(0.0, 1.0)
    except Exception:
        return None


def _save_cam_npy(out_dir, image_name, cam_2d):
    """
    Save CAM using the same filename rule as CASTSegDataset._resolve_cam_path:
        CSV Image = mask_covid_1.png -> covid_1.npy
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_name = Path(str(image_name)).name
    save_name = csv_name.replace('mask_', '')
    save_stem = Path(save_name).stem
    save_path = Path(out_dir) / f"{save_stem}.npy"
    np.save(str(save_path), cam_2d.astype(np.float32))
    return str(save_path)


def _select_asr_source_model(pl_module, args):
    """
    Select which network exports the dynamic CAM bank.

    Recommended for CAST-Seg:
        asr_source: teacher

    Reason:
        Teacher is already the EMA fusion of Student-A and Student-B, so the
        dynamic CAM bank is shared by both students and not biased toward only
        one branch.
    """
    source = str(getattr(args, "asr_source", getattr(args, "dynamic_cam_source", "teacher"))).lower()
    if source in ["teacher", "ema", "ema_model"]:
        return pl_module.teacher, "teacher"
    if source in ["student_b", "sam", "student_sam"]:
        return pl_module.student_b, "student_b"
    return pl_module.student_a, "student_a"


def generate_asr_prior_bank(pl_module, dataset, out_dir, args):
    """
    Generate the Teacher CAM bank used by Adaptive Semantic Refinement.

    The generated CAM bank is not owned by one student. It is written to disk
    and then dataset.cam_prior_path is switched, so the next epoch both
    Student-A and Student-B read exactly the same updated CAM prior.
    """
    device = pl_module.device
    model, source_name = _select_asr_source_model(pl_module, args)
    was_training = model.training
    model.eval()

    # Teacher parameters are frozen during training. Grad-CAM needs gradients
    # through the selected network, so temporarily enable requires_grad when
    # the selected source is the EMA teacher.
    requires_grad_backup = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer_name = str(
        getattr(args, "asr_target_layer", getattr(args, "dynamic_cam_target_layer", "foreground_decoder.up2.nConvs"))
    )
    target_layer = _resolve_module_by_name(model, target_layer_name)

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    batch_size = int(getattr(args, "asr_batch_size", getattr(args, "dynamic_cam_batch_size", getattr(args, "valid_batch_size", 8))))
    grad_weight = float(getattr(args, "asr_grad_weight", getattr(args, "dynamic_cam_grad_weight", 0.50)))
    pred_weight = 1.0 - grad_weight
    ema_momentum = float(getattr(args, "asr_ema_momentum", getattr(args, "dynamic_cam_ema_momentum", 0.0)))
    min_value = float(getattr(args, "asr_min_value", getattr(args, "dynamic_cam_min_value", getattr(args, "cam_prior_min_value", 0.05))))
    eps = 1e-6

    image_names = list(dataset.image_list)
    generated = 0
    manifest_rows = []

    try:
        for start in range(0, len(image_names), batch_size):
            names = image_names[start:start + batch_size]
            images = []
            for name in names:
                image_path = dataset._resolve_image_path(name)
                img = dataset._load_image_as_tensor(image_path)
                images.append(img)

            image = torch.stack(images, dim=0).to(device=device, dtype=torch.float32)
            gt_dummy = torch.zeros(
                (image.shape[0], 1, image.shape[-2], image.shape[-1]),
                device=device,
                dtype=torch.int,
            )

            model.zero_grad(set_to_none=True)
            activations.clear()
            gradients.clear()

            with torch.enable_grad():
                preds, _, _, _ = model([image, gt_dummy])
                target_score = preds.mean(dim=(1, 2, 3)).sum()
                target_score.backward()

            if "value" not in activations or "value" not in gradients:
                raise RuntimeError(
                    f"Grad-CAM hooks did not capture activations/gradients from layer: {target_layer_name}"
                )

            acts = activations["value"].detach()
            grads = gradients["value"].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            gradcam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            gradcam = F.interpolate(
                gradcam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gradcam = _normalize_cam_batch(gradcam, eps=eps)

            pred_cam = preds.detach().clamp(0.0, 1.0)
            pred_cam = _normalize_cam_batch(pred_cam, eps=eps)

            new_cam = grad_weight * gradcam + pred_weight * pred_cam
            new_cam = _normalize_cam_batch(new_cam, eps=eps)

            if ema_momentum > 0:
                blended = []
                for j, name in enumerate(names):
                    old_cam = _load_existing_cam_for_blend(dataset, name, device, new_cam.dtype)
                    if old_cam is not None:
                        old_cam = F.interpolate(
                            old_cam,
                            size=image.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0.0, 1.0)
                        c = ema_momentum * old_cam[0] + (1.0 - ema_momentum) * new_cam[j]
                    else:
                        c = new_cam[j]
                    blended.append(c)
                new_cam = torch.stack(blended, dim=0)
                new_cam = _normalize_cam_batch(new_cam, eps=eps)

            if min_value > 0:
                new_cam = torch.where(new_cam <= 0.0, torch.full_like(new_cam, min_value), new_cam)
                new_cam = new_cam.clamp(0.0, 1.0)

            for j, name in enumerate(names):
                cam_np = new_cam[j, 0].detach().cpu().numpy().astype(np.float32)
                save_path = _save_cam_npy(out_dir, name, cam_np)
                manifest_rows.append(f"{name},{save_path}\n")
                generated += 1

            del image, gt_dummy, preds, target_score, acts, grads, gradcam, pred_cam, new_cam
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        model.zero_grad(set_to_none=True)
        for p, req in zip(model.parameters(), requires_grad_backup):
            p.requires_grad_(req)
        if was_training:
            model.train()

    manifest_path = Path(out_dir) / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Image,CAMPath\n")
        f.writelines(manifest_rows)

    print(f"[ASR] CAM source: {source_name}")
    return generated, str(manifest_path)


class AdaptiveSemanticRefinementCallback(Callback):
    """
    Periodically replace the shared CAM bank with Teacher-generated priors.

    Both students use the updated CAM prior in the next epoch because the
    callback switches ds_semi.cam_prior_path directly.
    """
    def __init__(self, args, train_dataset):
        super().__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.enabled = cfg_bool(getattr(args, "asr_enabled", getattr(args, "use_dynamic_cam_update", False)), default=False)
        self.warmup_epochs = int(getattr(args, "asr_warmup_epochs", getattr(args, "dynamic_cam_warmup_epochs", 0)))
        self.update_interval = int(getattr(args, "asr_update_interval", 1))
        self.output_root = str(getattr(args, "asr_output_root", getattr(args, "dynamic_cam_output_root", "./asr_cam_priors")))
        self.schedule = AdaptiveSemanticRefinementSchedule(
            warmup_epochs=self.warmup_epochs,
            update_interval=self.update_interval,
        )
        self.last_update_epoch = -1

    def _set_dataset_cam_path(self, new_cam_dir):
        if hasattr(self.train_dataset, "update_cam_prior_path"):
            self.train_dataset.update_cam_prior_path(new_cam_dir, strict_cam_prior=True)
        else:
            self.train_dataset.use_cam_prior = True
            self.train_dataset.cam_prior_path = new_cam_dir
            print(f"[ASR] train_dataset.cam_prior_path = {new_cam_dir}")

        if hasattr(self.args, "cam_prior_path"):
            self.args.cam_prior_path = new_cam_dir

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        current_path = getattr(self.train_dataset, "cam_prior_path", None)
        print(f"[ASR] Epoch {trainer.current_epoch} uses shared CAM path: {current_path}")

    def on_validation_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch)
        if not self.schedule.should_update(epoch):
            return

        out_dir = os.path.join(self.output_root, f"epoch_{epoch:03d}")
        print(f"[ASR] Periodic refresh at epoch {epoch}. Generating shared CAM bank...")
        generated, manifest_path = generate_asr_prior_bank(
            pl_module=pl_module,
            dataset=self.train_dataset,
            out_dir=out_dir,
            args=self.args,
        )

        self._set_dataset_cam_path(out_dir)
        setattr(pl_module, "asr_cam_prior_path", out_dir)
        self.last_update_epoch = epoch

        try:
            first_name = self.train_dataset.image_list[0]
            first_cam_path = self.train_dataset._resolve_cam_path(first_name)
            print(f"[ASR] Check first CAM: {first_name} -> {first_cam_path}")
        except Exception as e:
            print(f"[ASR] Warning: failed to verify first CAM after update: {e}")

        print(f"[ASR] Updated shared CAM bank: {out_dir}")
        print(f"[ASR] Generated {generated} CAM files. Manifest: {manifest_path}")


if __name__ == '__main__':
    set_seed(0)
    args = get_parser()
    print("cuda:", torch.cuda.is_available())

    ds_valid = build_dataset(args, args.train_csv_path, args.train_root_path, mode='valid')
    dl_valid = DataLoader(
        ds_valid,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.valid_batch_size,
    )

    do_pretrain = cfg_bool(getattr(args, "pretrain", False), default=False)
    pretrained_ckpt_path = get_pretrained_ckpt_path(args)

    if do_pretrain:
        print("Preparing labeled data for CAST-Seg pre-training...")
        ds_labeled = build_pretrain_dataset(args, mode='pretrain')
        ds_pretrain_valid = build_pretrain_dataset(args, mode='valid')
        dl_labeled = DataLoader(
            ds_labeled,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.train_batch_size,
        )
        dl_pretrain_valid = DataLoader(
            ds_pretrain_valid,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.valid_batch_size,
        )

        print("Start paper-aligned supervised pre-training...")
        model = CASTSegPretrainModule(args)

        model_ckpt_pre = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=args.model_save_filename,
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
        early_stopping_pre = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
        trainer_pre = pl.Trainer(
            logger=True,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=args.device,
            callbacks=[model_ckpt_pre, early_stopping_pre, ETACallback()],
            enable_progress_bar=False,
        )
        trainer_pre.fit(model, dl_labeled, dl_pretrain_valid)
        print("CAST-Seg pre-training complete.")
        if model_ckpt_pre.best_model_path:
            pretrained_ckpt_path = model_ckpt_pre.best_model_path
            print(f"Use best pre-training checkpoint for semi-supervised training: {pretrained_ckpt_path}")
    else:
        print("Skip CAST-Seg pre-training.")
        print(f"Use existing pretrained segmentation checkpoint: {pretrained_ckpt_path}")

    print("Preparing CAST-Seg dual-student semi-supervised training...")
    model = CASTSegModule(args)
    load_pretrained_segmentation_into_castseg(model, pretrained_ckpt_path)

    ds_semi = build_dataset(args, args.train_csv_path, args.train_root_path, mode='semi')
    dl_semi = DataLoader(
        ds_semi,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=False,
    )

    semi_save_path = getattr(args, "semi_save_path", "./semi_supervised")
    semi_save_filename = getattr(args, "semi_save_filename", "cast_seg")
    model_ckpt_semi = ModelCheckpoint(
        dirpath=semi_save_path,
        filename=semi_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )
    early_stopping_semi = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')

    callbacks_semi = [model_ckpt_semi, early_stopping_semi, ETACallback()]
    asr_callback = AdaptiveSemanticRefinementCallback(args, ds_semi)
    if asr_callback.enabled:
        callbacks_semi.append(asr_callback)
        print("[ASR] Enabled periodic semantic-prior refinement.")

    trainer_semi = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.device,
        callbacks=callbacks_semi,
        enable_progress_bar=False,
    )

    print("Start CAST-Seg semi-supervised training...")
    trainer_semi.fit(model, dl_semi, dl_valid)
    print("CAST-Seg semi-supervised training complete.")

    semi_ckpt_path = model_ckpt_semi.best_model_path
    if not semi_ckpt_path:
        semi_filename = semi_save_filename
        if not str(semi_filename).endswith(".ckpt"):
            semi_filename = str(semi_filename) + ".ckpt"
        semi_ckpt_path = os.path.join(semi_save_path, semi_filename)

    print("Testing CAST-Seg...")
    model = CASTSegModule(args)
    load_castseg_checkpoint(model, semi_ckpt_path, strict=True)

    ds_test = build_dataset(args, args.test_csv_path, args.test_root_path, mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer_test = pl.Trainer(accelerator='gpu', devices=args.device)
    model.eval()
    trainer_test.test(model, dl_test)
