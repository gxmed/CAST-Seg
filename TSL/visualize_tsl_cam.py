import os
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


import os
import argparse
import random
import inspect
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据集
from utils.tsl_dataset import TSLCAMDataset
from utils.tsl_model import TSLGenerator


SCRIPT_DIR = Path(__file__).resolve().parent


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cfg_sections(cfg):
    """
    把 YAML 里的 TRAIN / MODEL / DATA 合并成一个简单字典，方便取参数
    """
    out = {}
    for sec in ["TRAIN", "MODEL", "DATA"]:
        if sec in cfg and isinstance(cfg[sec], dict):
            out.update(cfg[sec])
    return out


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_convnext_channels(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return tuple(int(x.strip()) for x in value.split(","))
    return None


def build_model(args, device):
    """
    构建论文中的 TSL CAM 模型。

    注意：TSLGenerator.__init__ 不接收 project_dim，
    所以这里不会再传 project_dim。为了以后兼容，也会自动过滤掉模型构造函数
    不支持的参数。
    """
    target_size = tuple(args.get("image_size", [224, 224]))
    convnext_channels = _parse_convnext_channels(args.get("convnext_channels", None))

    kwargs = {
        "bert_type": args["bert_type"],
        "vision_type": args["vision_type"],
        "cam_dim": args.get("cam_dim", 256),
        "text_pooling": args.get("text_pooling", "mean"),
        "target_size": target_size,
        "freeze_text": True,
        "freeze_vision": False,
        "trust_remote_code": args.get("trust_remote_code", True),
    }
    if convnext_channels is not None:
        kwargs["convnext_channels"] = convnext_channels

    # 只传 TSLGenerator 真正支持的参数，避免 unexpected keyword argument 报错
    sig = inspect.signature(TSLGenerator.__init__)
    valid_keys = set(sig.parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    print("Build TSLGenerator with args:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    model = TSLGenerator(**kwargs)
    model = model.to(device)
    return model


def load_ckpt(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容多种 checkpoint 保存方式
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded ckpt: {ckpt_path}")
    print(f"Missing keys   : {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    if len(incompatible.missing_keys) > 0:
        print("First missing keys:", incompatible.missing_keys[:5])
    if len(incompatible.unexpected_keys) > 0:
        print("First unexpected keys:", incompatible.unexpected_keys[:5])
    return model


@torch.no_grad()
def predict_cam(model, batch, device):
    image = batch["image"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    out = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

    # 兼容三种返回形式：
    # 1) dataclass / object: out.cam_logits
    # 2) dict: out["cam_logits"] / out["logits"]
    # 3) tensor: logits
    if hasattr(out, "cam_logits"):
        logits = out.cam_logits
    elif isinstance(out, dict):
        if "cam_logits" in out:
            logits = out["cam_logits"]
        elif "logits" in out:
            logits = out["logits"]
        else:
            raise ValueError("Model output dict does not contain 'cam_logits' or 'logits'.")
    elif torch.is_tensor(out):
        logits = out
    else:
        raise TypeError(f"Unsupported model output type: {type(out)}")

    prob = torch.sigmoid(logits)
    return prob


def save_side_by_side(label_cam, pred_cam, image_name, save_path):
    """
    左：标签
    右：预测
    """
    label_cam = np.squeeze(label_cam)
    pred_cam = np.squeeze(pred_cam)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(label_cam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[0].set_title("Label CAM")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_cam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted CAM")
    axes[1].axis("off")

    fig.suptitle(image_name, fontsize=10)
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    # 现在可以直接运行，不需要手动传 --config 和 --ckpt
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "configs" / "tsl_qata.yaml"),
        help="Path to a TSL YAML configuration",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(SCRIPT_DIR / "save_model" / "tsl_qata.ckpt"),
        help="Path to trained CAM checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["labeled", "validation", "unlabeled", "all"],
        help="Which split to visualize",
    )
    parser.add_argument("--out_dir", type=str, default="./cam_vis_results")
    parser.add_argument("--num_samples", type=int, default=50, help="How many samples to save")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args_cli = parser.parse_args()

    print(f"Config: {args_cli.config}")
    print(f"Checkpoint: {args_cli.ckpt}")
    print(f"Split: {args_cli.split}")
    print(f"Output dir: {args_cli.out_dir}")
    print(f"Num samples: {args_cli.num_samples}")

    seed_everything(args_cli.seed)

    cfg = load_yaml_config(args_cli.config)
    args = merge_cfg_sections(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集（优先用 test 路径，没有则 fallback 到 train 路径）
    dataset = TSLCAMDataset(
        csv_path=args.get("test_csv_path", args["train_csv_path"]),
        root_path=args.get("test_root_path", args["train_root_path"]),
        cam_root_path=args.get("train_cam_path", ""),
        bert_type=args["bert_type"],
        image_size=tuple(args.get("image_size", [224, 224])),
        text_max_length=args.get("text_max_length", 64),
        prompt_template=args.get(
            "prompt_template",
            "Find the infected region in the chest X-ray according to the report: {report}",
        ),
        image_normalize=True,
        strict_cam=True,
        cam_split=args_cli.split,
        labeled_ratio=args.get("labeled_ratio", 0.04),
        valid_ratio=args.get("valid_ratio", 0.2),
        trust_remote_code=args.get("trust_remote_code", True),
    )

    loader = DataLoader(
        dataset,
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=args_cli.num_workers,
        pin_memory=True,
    )

    # 模型
    model = build_model(args, device)
    model = load_ckpt(model, args_cli.ckpt, device)
    model.eval()

    save_dir = os.path.join(args_cli.out_dir, args_cli.split)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0

    for batch in loader:
        pred = predict_cam(model, batch, device)   # [B,1,H,W]
        gt = batch["cam"]                          # [B,1,H,W]
        image_names = batch["image_name"]

        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        bsz = pred.shape[0]
        for i in range(bsz):
            image_name = image_names[i]
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            save_path = os.path.join(save_dir, f"{saved:04d}_{base_name}.png")

            save_side_by_side(
                label_cam=gt[i],
                pred_cam=pred[i],
                image_name=image_name,
                save_path=save_path,
            )

            saved += 1
            if saved >= args_cli.num_samples:
                print(f"Saved {saved} visualization images to: {save_dir}")
                return

    print(f"Saved {saved} visualization images to: {save_dir}")


if __name__ == "__main__":
    main()
