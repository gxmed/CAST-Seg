"""
Dataset for Text-guided Semantic Localization training.

Expected CSV columns:
    Image
    Description (or legacy text)

Expected image path:
    root_path/Images/<Image without mask_ prefix>

Expected CAM label path:
    cam_root_path/<corresponding CAM label>

Supports CAM labels in:
    .npy  float32 [H, W] or [1, H, W], range 0~1
    .npz  with key 'cam' or first array, range 0~1
    grayscale .png/.jpg/.bmp/.tif, read as single-channel and normalized to 0~1

The labeled subset size is controlled by ``labeled_ratio``. CAST-Seg uses
``labeled_ratio=0.04`` in code for the paper's 5% setting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class TSLCAMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_path: str,
        cam_root_path: str,
        bert_type: str,
        image_size: Tuple[int, int] = (224, 224),
        text_max_length: int = 64,
        prompt_template: str = "Find the infected region in the chest X-ray according to the report: {report}",
        image_normalize: bool = True,
        strict_cam: bool = True,
        cam_split: str = "labeled",
        labeled_ratio: float = 0.04,
        valid_ratio: float = 0.2,
        trust_remote_code: bool = True,
    ):
        self.csv_path = csv_path
        self.root_path = root_path
        self.cam_root_path = cam_root_path
        self.image_size = tuple(image_size)
        self.text_max_length = int(text_max_length)
        self.prompt_template = prompt_template
        self.strict_cam = bool(strict_cam)
        self.cam_split = str(cam_split)
        self.labeled_ratio = float(labeled_ratio)
        self.valid_ratio = float(valid_ratio)

        self.df = pd.read_csv(csv_path)
        if "Image" not in self.df.columns:
            raise ValueError(f"CSV must contain column 'Image'. Found: {list(self.df.columns)}")
        if "Description" in self.df.columns:
            self.report_column = "Description"
        elif "text" in self.df.columns:
            self.report_column = "text"
        else:
            raise ValueError(f"CSV must contain column 'Description' or 'text'. Found: {list(self.df.columns)}")

        self.df = self._apply_cam_split(self.df)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type, trust_remote_code=trust_remote_code)

        image_tfms = [
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        if image_normalize:
            image_tfms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        self.image_transform = transforms.Compose(image_tfms)

    def _apply_cam_split(self, df: pd.DataFrame) -> pd.DataFrame:
        n_total = len(df)
        n_labeled = int(n_total * self.labeled_ratio)
        n_valid = int(n_total * self.valid_ratio)
        split = self.cam_split.lower()

        if split in ["labeled", "cam_train", "train_labeled"]:
            out = df.iloc[:n_labeled]
        elif split == "unlabeled":
            out = df.iloc[n_labeled:n_total - n_valid]
        elif split in ["validation", "valid", "cam_valid"]:
            out = df.iloc[n_total - n_valid:]
        elif split in ["all", "full"]:
            out = df
        else:
            raise ValueError(
                f"Unknown cam_split={self.cam_split}. Use labeled / unlabeled / validation / all."
            )

        out = out.reset_index(drop=True)
        print(
            f"[TSLCAMDataset] cam_split={self.cam_split}, "
            f"samples={len(out)} / total={n_total}, "
            f"labeled_ratio={self.labeled_ratio}, valid_ratio={self.valid_ratio}"
        )
        return out

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_name: str) -> str:
        return os.path.join(self.root_path, "Images", image_name.replace("mask_", ""))

    def _resolve_cam_path(self, image_name: str) -> Optional[str]:
        candidates = []
        p = Path(self.cam_root_path)
        name = Path(str(image_name)).name
        no_mask = name.replace("mask_", "")
        stem = Path(name).stem
        no_mask_stem = Path(no_mask).stem

        # Exact names first.
        candidates.append(p / name)
        candidates.append(p / no_mask)

        # Prefer float labels if present.
        for ext in [".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidates.append(p / f"{stem}{ext}")
            candidates.append(p / f"{no_mask_stem}{ext}")
            candidates.append(p / f"cam_{stem}{ext}")
            candidates.append(p / f"cam_{no_mask_stem}{ext}")
            candidates.append(p / f"CAM_{stem}{ext}")
            candidates.append(p / f"CAM_{no_mask_stem}{ext}")

        for c in candidates:
            if c.exists():
                return str(c)
        return None

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_cam_tensor(self, path: str) -> torch.Tensor:
        ext = Path(path).suffix.lower()

        if ext == ".npy":
            cam = np.load(path).astype(np.float32)
        elif ext == ".npz":
            data = np.load(path)
            key = "cam" if "cam" in data.files else data.files[0]
            cam = data[key].astype(np.float32)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            cam_img = Image.open(path).convert("L")
            cam_img = cam_img.resize(self.image_size[::-1], resample=Image.BILINEAR)
            cam = np.asarray(cam_img, dtype=np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported CAM label format: {path}")

        if cam.ndim == 3:
            if cam.shape[0] == 1:
                cam = cam[0]
            elif cam.shape[-1] == 1:
                cam = cam[..., 0]
            else:
                raise ValueError(f"CAM label should be single-channel. Got shape {cam.shape} from {path}")

        cam = np.clip(cam, 0.0, 1.0).astype(np.float32)
        cam_t = torch.from_numpy(cam)[None, ...]  # [1, H, W]
        if tuple(cam_t.shape[-2:]) != self.image_size:
            cam_t = torch.nn.functional.interpolate(
                cam_t[None],
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )[0]
        return cam_t.float().clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]

        image_name = str(row["Image"])
        report = str(row[self.report_column])
        text = self.prompt_template.format(report=report)

        image_path = self._resolve_image_path(image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        cam_path = self._resolve_cam_path(image_name)
        if cam_path is None:
            msg = f"CAM label not found for Image={image_name} under cam_root_path={self.cam_root_path}"
            if self.strict_cam:
                raise FileNotFoundError(msg)
            cam = torch.zeros((1, *self.image_size), dtype=torch.float32)
        else:
            cam = self._load_cam_tensor(cam_path)

        image = self.image_transform(self._load_rgb(image_path))
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "cam": cam,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "image_name": image_name,
            "report": report,
            "text": text,
            "image_path": image_path,
            "cam_path": cam_path or "",
        }
