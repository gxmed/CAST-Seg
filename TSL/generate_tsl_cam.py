"""Generate one TSL CAM prior and save it as a NumPy array."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils.tsl_model import TSLGenerator, set_text_encoder_frozen


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, device):
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
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    # strip module. prefix if needed
    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


@torch.no_grad()
def generate_single_cam(cfg, ckpt_path, image_path, report_text, out_npy_path):
    train_cfg = cfg.get("TRAIN", {})
    model_cfg = cfg.get("MODEL", {})
    device_ids = train_cfg.get("device", [0])
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model & load weights
    model = build_model(cfg, device)
    load_checkpoint(model, ckpt_path, device)

    # Tokenizer
    bert_type = model_cfg["bert_type"]
    tokenizer = AutoTokenizer.from_pretrained(bert_type, trust_remote_code=True)

    # Image transform
    image_size = tuple(train_cfg.get("image_size", [224, 224]))
    image_normalize = bool(train_cfg.get("image_normalize", True))
    tfms = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if image_normalize:
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    image_transform = transforms.Compose(tfms)

    # Load & preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # Tokenize text
    prompt_template = train_cfg.get(
        "prompt_template",
        "Find the infected region in the chest X-ray according to the report: {report}",
    )
    text = prompt_template.format(report=report_text)
    tokens = tokenizer(text, padding="max_length", truncation=True,
                       max_length=int(train_cfg.get("text_max_length", 64)),
                       return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Forward
    out = model(image=image_tensor, input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "cam_logits"):
        prob = torch.sigmoid(out.cam_logits)
    elif hasattr(out, "cam_prob"):
        prob = out.cam_prob
    elif isinstance(out, dict) and "cam_logits" in out:
        prob = torch.sigmoid(out["cam_logits"])
    else:
        prob = torch.sigmoid(out)

    cam = prob.detach().float().clamp(0.0, 1.0).cpu().numpy()[0, 0]
    print(f"CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}, mean: {cam.mean():.4f}")

    # Save
    Path(out_npy_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, cam.astype(np.float32))
    print(f"Saved: {out_npy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "tsl_qata.yaml"),
        help="TSL YAML configuration",
    )
    parser.add_argument("--ckpt", required=True, help="Trained TSL checkpoint")
    parser.add_argument("--image", required=True, help="Input medical image")
    parser.add_argument("--text", required=True, help="Diagnostic report text")
    parser.add_argument("--output", required=True, help="Output .npy path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    generate_single_cam(cfg, args.ckpt, args.image, args.text, args.output)
