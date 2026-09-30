import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.sca import SynchronizedCrossModalAugmentation


class CASTSegDataset(Dataset):
    """
    CAST-Seg dataset for dual-student, CAM-guided and SAM-expert training.

    Output formats:
      Semi-supervised training with TSL CAM + MDAA-SAM priors:
        labeled:   ([image, cam, sam, gt], gt, flag=1)
        unlabeled: ([image, cam, sam, placeholder_gt], placeholder_gt, flag=0)

      Evaluation/test without priors:
        ([image, text, gt], gt, flag=1)

    Path rules:
      CSV Image = mask_covid_1.png
      image     = root_path/Images/covid_1.png
      gt        = root_path/GTs/mask_covid_1.png
      cam/sam   = prior_path/covid_1.npy or prior_path/mask_covid_1.*

    Spatial augmentation is synchronized over image, GT, CAM and SAM.
    """

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',
                 image_size=[224, 224], is_labeled=True, text_column='Description',
                 max_text_length=24, labeled_ratio=0.04, valid_ratio=0.2,
                 fold_index=None, num_folds=5,
                 use_spatial_aug=False,
                 use_cam_prior=False, cam_prior_path=None,
                 cam_prior_min_value=0.05, cam_prior_zero_eps=0.0,
                 strict_cam_prior=True,
                 use_sam_prior=False, sam_prior_path=None,
                 strict_sam_prior=True, sam_prior_min_value=0.0,
                 sam_prior_zero_eps=0.0,
                 spatial_aug_prob=0.5, spatial_aug_hflip=True,
                 spatial_aug_vflip=False, spatial_aug_rotate=True,
                 spatial_aug_max_angle=10.0):
        super().__init__()

        self.mode = mode
        self.is_labeled = is_labeled
        self.root_path = root_path
        self.image_size = image_size
        self.text_column = text_column
        self.max_text_length = max_text_length
        self.labeled_ratio = float(labeled_ratio)
        self.valid_ratio = float(valid_ratio)
        self.fold_index = None if fold_index is None or str(fold_index).strip() == '' else int(fold_index)
        self.num_folds = int(num_folds)

        self.use_cam_prior = bool(use_cam_prior)
        self.cam_prior_path = cam_prior_path
        self.cam_prior_min_value = float(cam_prior_min_value)
        self.cam_prior_zero_eps = float(cam_prior_zero_eps)
        self.strict_cam_prior = bool(strict_cam_prior)

        self.use_sam_prior = bool(use_sam_prior)
        self.sam_prior_path = sam_prior_path
        self.strict_sam_prior = bool(strict_sam_prior)
        self.sam_prior_min_value = float(sam_prior_min_value)
        self.sam_prior_zero_eps = float(sam_prior_zero_eps)

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
        if self.use_sam_prior and (self.sam_prior_path is None or str(self.sam_prior_path).strip() == ''):
            raise ValueError("use_sam_prior=True, but sam_prior_path is empty.")

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            self.data = pd.read_csv(f)
        if 'Image' not in self.data.columns:
            raise ValueError(f"CSV must contain column 'Image'. Found: {list(self.data.columns)}")

        self.image_list = list(self.data['Image'])
        if self.text_column in self.data.columns:
            self.caption_list = self.data[self.text_column].fillna('').astype(str).tolist()
        else:
            fallback_columns = [
                'Report', 'report', 'Text', 'text', 'Prompt', 'prompt',
                'Finding', 'Findings', 'Impression', 'Caption', 'caption'
            ]
            found_col = next((c for c in fallback_columns if c in self.data.columns), None)
            if found_col is not None:
                self.caption_list = self.data[found_col].fillna('').astype(str).tolist()
                self.text_column = found_col
            else:
                self.caption_list = [''] * len(self.image_list)

        all_images = self.image_list
        all_captions = self.caption_list
        n = len(all_images)
        if self.fold_index is None:
            valid_start = int((1.0 - self.valid_ratio) * n)
            train_indices = list(range(valid_start))
            valid_indices = list(range(valid_start, n))
        else:
            if self.fold_index < 0 or self.fold_index >= self.num_folds:
                raise ValueError(f"fold_index must be in [0, {self.num_folds - 1}], got {self.fold_index}")
            valid_start = int(self.fold_index * n / self.num_folds)
            valid_end = int((self.fold_index + 1) * n / self.num_folds)
            valid_indices = list(range(valid_start, valid_end))
            train_indices = list(range(0, valid_start)) + list(range(valid_end, n))

        num_labeled = int(self.labeled_ratio * n)
        num_labeled = max(1, min(num_labeled, len(train_indices)))

        if mode == 'pretrain':
            labeled_indices = train_indices[:num_labeled]
            self.image_list = [all_images[i] for i in labeled_indices]
            self.caption_list = [all_captions[i] for i in labeled_indices]
        elif mode == 'semi':
            train_image_list = [all_images[i] for i in train_indices]
            train_caption_list = [all_captions[i] for i in train_indices]
            self.labeled_image_list = train_image_list[:num_labeled]
            self.unlabeled_image_list = train_image_list[num_labeled:]
            self.labeled_caption_list = train_caption_list[:num_labeled]
            self.unlabeled_caption_list = train_caption_list[num_labeled:]
            self.image_list = self.labeled_image_list + self.unlabeled_image_list
            self.caption_list = self.labeled_caption_list + self.unlabeled_caption_list
        elif mode == 'valid':
            self.image_list = [all_images[i] for i in valid_indices]
            self.caption_list = [all_captions[i] for i in valid_indices]

        self.tokenizer = None
        if tokenizer is not None and str(tokenizer).strip() != '':
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def update_cam_prior_path(self, new_cam_prior_path, strict_cam_prior=None):
        if new_cam_prior_path is None or str(new_cam_prior_path).strip() == '':
            raise ValueError("new_cam_prior_path is empty.")
        new_cam_prior_path = str(new_cam_prior_path)
        if not os.path.isdir(new_cam_prior_path):
            raise FileNotFoundError(f"Dynamic CAM directory does not exist: {new_cam_prior_path}")
        self.use_cam_prior = True
        self.cam_prior_path = new_cam_prior_path
        if strict_cam_prior is not None:
            self.strict_cam_prior = bool(strict_cam_prior)
        print(f"[CASTSegDataset] CAM prior path updated to: {self.cam_prior_path}")
        return self.cam_prior_path

    def update_sam_prior_path(self, new_sam_prior_path, strict_sam_prior=None):
        if new_sam_prior_path is None or str(new_sam_prior_path).strip() == '':
            raise ValueError("new_sam_prior_path is empty.")
        new_sam_prior_path = str(new_sam_prior_path)
        if not os.path.isdir(new_sam_prior_path):
            raise FileNotFoundError(f"SAM prior directory does not exist: {new_sam_prior_path}")
        self.use_sam_prior = True
        self.sam_prior_path = new_sam_prior_path
        if strict_sam_prior is not None:
            self.strict_sam_prior = bool(strict_sam_prior)
        print(f"[CASTSegDataset] SAM prior path updated to: {self.sam_prior_path}")
        return self.sam_prior_path

    def _tokenize_caption(self, caption):
        if self.tokenizer is None:
            token = torch.zeros((1, self.max_text_length), dtype=torch.long)
            mask = torch.zeros((1, self.max_text_length), dtype=torch.long)
            return token, mask
        token_output = self.tokenizer.encode_plus(
            caption,
            padding='max_length',
            max_length=self.max_text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return token_output['input_ids'], token_output['attention_mask']

    def _resize_size_pil(self):
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be [H, W], got: {self.image_size}")
        h, w = int(self.image_size[0]), int(self.image_size[1])
        return (w, h)

    @staticmethod
    def _image_file_name_from_csv_name(csv_image_name: str) -> str:
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

        candidates = []
        stems = [
            image_stem, csv_stem,
            f"sam_{image_stem}", f"sam_{csv_stem}",
            f"SAM_{image_stem}", f"SAM_{csv_stem}",
            f"cam_{image_stem}", f"cam_{csv_stem}",
            f"CAM_{image_stem}", f"CAM_{csv_stem}",
        ]
        for stem in stems:
            for ext in ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                candidates.append(root / f"{stem}{ext}")
        candidates.append(root / csv_name)
        candidates.append(root / image_name_no_mask)

        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _resolve_cam_path(self, image_name):
        if not self.use_cam_prior:
            return None
        return self._resolve_prior_path(image_name, self.cam_prior_path)

    def _resolve_sam_path(self, image_name):
        if not self.use_sam_prior:
            return None
        return self._resolve_prior_path(image_name, self.sam_prior_path)

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

    def _load_prior_as_tensor(self, image_name, prior_root, strict=True, prior_name='prior'):
        prior_path = self._resolve_prior_path(image_name, prior_root)
        if prior_path is None:
            msg = f"{prior_name} not found for Image={image_name} under path={prior_root}"
            if strict:
                raise FileNotFoundError(msg)
            return torch.ones((1, int(self.image_size[0]), int(self.image_size[1])), dtype=torch.float32)

        ext = Path(prior_path).suffix.lower()
        if ext == '.npy':
            arr = np.load(prior_path).astype(np.float32)
        elif ext == '.npz':
            data = np.load(prior_path)
            key = 'cam' if 'cam' in data.files else ('sam' if 'sam' in data.files else data.files[0])
            arr = data[key].astype(np.float32)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            with Image.open(prior_path) as img:
                img = img.convert('L')
                img = img.resize(self._resize_size_pil(), resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
        else:
            raise ValueError(f"Unsupported {prior_name} format: {prior_path}")

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"{prior_name} must be single-channel. Got shape {arr.shape} from {prior_path}")
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        prior = torch.from_numpy(arr).unsqueeze(0).float()
        if tuple(prior.shape[-2:]) != tuple(self.image_size):
            prior = F.interpolate(
                prior.unsqueeze(0),
                size=tuple(self.image_size),
                mode='bilinear',
                align_corners=False,
            )[0]
        return prior.float().clamp(0.0, 1.0)

    def _load_cam_as_tensor(self, image_name):
        return self._load_prior_as_tensor(
            image_name,
            self.cam_prior_path,
            strict=self.strict_cam_prior,
            prior_name='CAM prior',
        )

    def _load_sam_as_tensor(self, image_name):
        return self._load_prior_as_tensor(
            image_name,
            self.sam_prior_path,
            strict=self.strict_sam_prior,
            prior_name='SAM expert mask',
        )

    def _apply_min_value(self, x, min_value, zero_eps):
        if x is None:
            return None
        x = x.float().clamp(0.0, 1.0)
        if min_value > 0:
            x = torch.where(x <= zero_eps, torch.full_like(x, min_value), x)
        return x.clamp(0.0, 1.0)

    def _should_apply_spatial_aug(self):
        return self.use_spatial_aug and self.mode in ['train', 'pretrain', 'semi']

    @staticmethod
    def _rotate_tensor(x, angle_deg, mode='bilinear'):
        if x is None:
            return None
        dtype = x.dtype
        x_float = x.float().unsqueeze(0)
        angle = float(angle_deg) * np.pi / 180.0
        c, s = np.cos(angle), np.sin(angle)
        theta = torch.tensor(
            [[[c, -s, 0.0], [s, c, 0.0]]],
            dtype=x_float.dtype,
            device=x_float.device,
        )
        grid = F.affine_grid(theta, size=x_float.shape, align_corners=False)
        out = F.grid_sample(
            x_float,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=False,
        )[0]
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            return out.round().to(dtype)
        return out.to(dtype)

    def _apply_spatial_aug(self, image, gt, cam, sam):
        if not self._should_apply_spatial_aug():
            return image, gt, cam, sam
        return self.sca(image, gt, cam, sam)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = self._resolve_image_path(image_name)
        gt_path = self._resolve_gt_path(image_name)

        caption = str(self.caption_list[idx])
        token, mask = self._tokenize_caption(caption)

        image = self._load_image_as_tensor(image_path)
        gt = self._load_mask_as_tensor(gt_path)
        cam = self._load_cam_as_tensor(image_name) if self.use_cam_prior else None
        sam = self._load_sam_as_tensor(image_name) if self.use_sam_prior else None

        image, gt, cam, sam = self._apply_spatial_aug(image, gt, cam, sam)
        cam = self._apply_min_value(cam, self.cam_prior_min_value, self.cam_prior_zero_eps)
        sam = self._apply_min_value(sam, self.sam_prior_min_value, self.sam_prior_zero_eps)

        text = {
            'input_ids': token.squeeze(dim=0).long(),
            'attention_mask': mask.squeeze(dim=0).long(),
            'raw_text': caption,
        }

        is_unlabeled = self.mode == 'semi' and hasattr(self, 'labeled_image_list') and idx >= len(self.labeled_image_list)
        if is_unlabeled:
            flag = 0
            placeholder_gt = torch.zeros_like(gt, dtype=torch.int)
            if self.use_cam_prior and self.use_sam_prior:
                return ([image, cam, sam, placeholder_gt], placeholder_gt, flag)
            if self.use_cam_prior:
                return ([image, cam, placeholder_gt], placeholder_gt, flag)
            if self.use_sam_prior:
                return ([image, sam, placeholder_gt], placeholder_gt, flag)
            return ([image, text, placeholder_gt], placeholder_gt, flag)

        flag = 1
        if self.use_cam_prior and self.use_sam_prior:
            return ([image, cam, sam, gt], gt, flag)
        if self.use_cam_prior:
            return ([image, cam, gt], gt, flag)
        if self.use_sam_prior:
            return ([image, sam, gt], gt, flag)
        return ([image, text, gt], gt, flag)
