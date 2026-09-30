import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from modules.sca import SynchronizedCrossModalAugmentation


class CASTSegPretrainDataset(Dataset):
    def __init__(self, csv_path=None, root_path=None, mode='train', image_size=[224, 224],
                 is_labeled=True, labeled_ratio=0.04, valid_ratio=0.2,
                 fold_index=None, num_folds=5,
                 use_cam_prior=False, cam_prior_path=None, strict_cam_prior=True,
                 cam_prior_min_value=0.0, cam_prior_zero_eps=0.0,
                 use_spatial_aug=False, spatial_aug_prob=0.5,
                 spatial_aug_hflip=True, spatial_aug_vflip=False,
                 spatial_aug_rotate=True, spatial_aug_max_angle=10.0):
        super().__init__()

        self.mode = mode
        self.is_labeled = is_labeled
        self.root_path = root_path
        self.image_size = image_size
        self.fold_index = None if fold_index is None or str(fold_index).strip() == '' else int(fold_index)
        self.num_folds = int(num_folds)

        self.use_cam_prior = bool(use_cam_prior)
        self.cam_prior_path = cam_prior_path
        self.strict_cam_prior = bool(strict_cam_prior)
        self.cam_prior_min_value = float(cam_prior_min_value)
        self.cam_prior_zero_eps = float(cam_prior_zero_eps)

        self.use_spatial_aug = bool(use_spatial_aug)
        self.spatial_aug_prob = float(spatial_aug_prob)
        self.spatial_aug_hflip = bool(spatial_aug_hflip)
        self.spatial_aug_vflip = bool(spatial_aug_vflip)
        self.spatial_aug_rotate = bool(spatial_aug_rotate)
        self.spatial_aug_max_angle = float(spatial_aug_max_angle)
        self.sca = SynchronizedCrossModalAugmentation(
            probability=self.spatial_aug_prob,
            horizontal_flip=self.spatial_aug_hflip,
            vertical_flip=self.spatial_aug_vflip,
            rotate=self.spatial_aug_rotate,
            max_angle=self.spatial_aug_max_angle,
        )

        if self.use_cam_prior and (self.cam_prior_path is None or str(self.cam_prior_path).strip() == ''):
            raise ValueError("use_cam_prior=True, but cam_prior_path is empty.")

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            self.data = pd.read_csv(f)
        all_images = list(self.data['Image'])

        n = len(all_images)
        if self.fold_index is None:
            valid_start = int((1.0 - float(valid_ratio)) * n)
            train_indices = list(range(valid_start))
            valid_indices = list(range(valid_start, n))
        else:
            if self.fold_index < 0 or self.fold_index >= self.num_folds:
                raise ValueError(f"fold_index must be in [0, {self.num_folds - 1}], got {self.fold_index}")
            valid_start = int(self.fold_index * n / self.num_folds)
            valid_end = int((self.fold_index + 1) * n / self.num_folds)
            valid_indices = list(range(valid_start, valid_end))
            train_indices = list(range(0, valid_start)) + list(range(valid_end, n))

        num_slice = int(float(labeled_ratio) * n)
        num_slice = max(1, min(num_slice, len(train_indices)))
        if mode == 'pretrain':
            self.image_list = [all_images[i] for i in train_indices[:num_slice]]
        elif mode == 'semi':
            train_image_list = [all_images[i] for i in train_indices]
            self.labeled_image_list = train_image_list[:num_slice]
            self.unlabeled_image_list = train_image_list[num_slice:]
            self.image_list = self.labeled_image_list + self.unlabeled_image_list
        elif mode == 'valid':
            self.image_list = [all_images[i] for i in valid_indices]
        else:
            self.image_list = all_images

    def __len__(self):
        return len(self.image_list)

    def _resize_size_pil(self):
        h, w = int(self.image_size[0]), int(self.image_size[1])
        return (w, h)

    @staticmethod
    def _image_file_name_from_csv_name(csv_image_name):
        return str(csv_image_name).replace('mask_', '')

    def _resolve_image_path(self, image_name):
        return os.path.join(self.root_path, 'Images', self._image_file_name_from_csv_name(image_name))

    def _resolve_gt_path(self, image_name):
        return os.path.join(self.root_path, 'GTs', str(image_name))

    def _resolve_prior_path(self, image_name, prior_root):
        root = Path(str(prior_root))
        csv_name = Path(str(image_name)).name
        image_name_no_mask = self._image_file_name_from_csv_name(csv_name)
        csv_stem = Path(csv_name).stem
        image_stem = Path(image_name_no_mask).stem

        stems = [
            image_stem, csv_stem,
            f"sam_{image_stem}", f"sam_{csv_stem}",
            f"SAM_{image_stem}", f"SAM_{csv_stem}",
            f"cam_{image_stem}", f"cam_{csv_stem}",
            f"CAM_{image_stem}", f"CAM_{csv_stem}",
        ]
        for stem in stems:
            for ext in ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                candidate = root / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate)
        for name in [csv_name, image_name_no_mask]:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
        return None

    def _load_image_as_tensor(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize(self._resize_size_pil(), resample=Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        for c in range(tensor.shape[0]):
            ch = tensor[c]
            mean = ch.mean()
            std = ch.std()
            tensor[c] = (ch - mean) / std if std > 1e-6 else ch - mean
        return tensor.float()

    def _load_mask_as_tensor(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"GT file not found: {path}")
        with Image.open(path) as mask:
            mask = mask.resize(self._resize_size_pil(), resample=Image.NEAREST)
            arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int64)
        elif arr.max() > 1:
            arr = (arr == 255).astype(np.int64)
        else:
            arr = arr.astype(np.int64)
        return torch.from_numpy(arr).unsqueeze(0).contiguous().int()

    def _load_cam_as_tensor(self, image_name):
        cam_path = self._resolve_prior_path(image_name, self.cam_prior_path)
        if cam_path is None:
            msg = f"CAM prior not found for Image={image_name} under path={self.cam_prior_path}"
            if self.strict_cam_prior:
                raise FileNotFoundError(msg)
            return torch.ones((1, int(self.image_size[0]), int(self.image_size[1])), dtype=torch.float32)

        ext = Path(cam_path).suffix.lower()
        if ext == '.npy':
            arr = np.load(cam_path).astype(np.float32)
        elif ext == '.npz':
            data = np.load(cam_path)
            key = 'cam' if 'cam' in data.files else data.files[0]
            arr = data[key].astype(np.float32)
        else:
            with Image.open(cam_path) as img:
                img = img.convert('L')
                img = img.resize(self._resize_size_pil(), resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        cam = torch.from_numpy(arr).unsqueeze(0).float()
        if tuple(cam.shape[-2:]) != tuple(self.image_size):
            cam = F.interpolate(
                cam.unsqueeze(0),
                size=tuple(self.image_size),
                mode='bilinear',
                align_corners=False,
            )[0]
        cam = cam.clamp(0.0, 1.0)
        if self.cam_prior_min_value > 0:
            cam = torch.where(cam <= self.cam_prior_zero_eps, torch.full_like(cam, self.cam_prior_min_value), cam)
        return cam.float().clamp(0.0, 1.0)

    @staticmethod
    def _rotate_tensor(x, angle_deg, mode='bilinear'):
        if x is None:
            return None
        dtype = x.dtype
        x_float = x.float().unsqueeze(0)
        angle = float(angle_deg) * np.pi / 180.0
        c, s = np.cos(angle), np.sin(angle)
        theta = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]], dtype=x_float.dtype)
        grid = F.affine_grid(theta, size=x_float.shape, align_corners=False)
        out = F.grid_sample(x_float, grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            return out.round().to(dtype)
        return out.to(dtype)

    def _apply_spatial_aug(self, image, gt, cam):
        if not (self.use_spatial_aug and self.mode in ['train', 'pretrain', 'semi']):
            return image, gt, cam
        image, gt, cam, _ = self.sca(image, gt, cam, None)
        return image, gt, cam

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = self._load_image_as_tensor(self._resolve_image_path(image_name))
        gt = self._load_mask_as_tensor(self._resolve_gt_path(image_name))
        cam = self._load_cam_as_tensor(image_name) if self.use_cam_prior else None

        image, gt, cam = self._apply_spatial_aug(image, gt, cam)

        if self.mode == 'semi' and hasattr(self, 'labeled_image_list') and idx >= len(self.labeled_image_list):
            flag = 0
            placeholder_gt = torch.zeros_like(gt, dtype=torch.int)
            if cam is not None:
                return ([image, placeholder_gt, cam], placeholder_gt, flag)
            return ([image, placeholder_gt], placeholder_gt, flag)

        flag = 1
        if cam is not None:
            return ([image, gt, cam], gt, flag)
        return ([image, gt], gt, flag)
