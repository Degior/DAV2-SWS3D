"""Building-height supervision from M4Heights ORTHO + LABEL."""

import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.m4heights_files import load_manifest, read_tiff
from dataset.transform import Crop, NormalizeImage, PrepareForNet, Resize


class M4Heights(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518), height_scale=1.0):
        if mode not in ('train', 'val'):
            raise ValueError(f'Unsupported mode: {mode}')
        if not math.isfinite(height_scale) or height_scale <= 0:
            raise ValueError('height_scale must be finite and positive')
        self.root, self.samples = load_manifest(filelist_path)
        self.mode = mode
        self.height_scale = height_scale
        self._archive_pid = None
        self._archives = {}
        self.transform = Compose([
            Resize(width=size[0], height=size[1], resize_target=(mode == 'train'),
                   keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == 'train' else []))

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        # Spawn workers must open their own handles, even after a parent read.
        return {**self.__dict__, '_archives': {}, '_archive_pid': None}

    def close(self):
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __del__(self):
        if hasattr(self, '_archives'):
            self.close()

    def _read(self, reference):
        pid = os.getpid()
        if self._archive_pid != pid:
            # Fork workers must not share the parent's seek position either.
            self.close()
            self._archive_pid = pid
        return read_tiff(self.root, reference, self._archives)

    def __getitem__(self, index):
        record = self.samples[index]
        image, axes, _ = self._read(record['image'])
        if image.ndim == 3 and axes in ('SYX', 'CYX'):
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
            raise ValueError(f'Expected uint8 RGB/RGBA orthophoto: {record["id"]}, {image.shape}, {image.dtype}')
        image = image[..., :3].astype(np.float32) / 255.0

        height, _, nodata = self._read(record['height'])
        if height.ndim == 3 and 1 in (height.shape[0], height.shape[-1]):
            height = np.squeeze(height, axis=0 if height.shape[0] == 1 else -1)
        if height.ndim != 2:
            raise ValueError(f'Expected single-channel height map: {record["id"]}, {height.shape}')
        height = height.astype(np.float32)
        valid = np.isfinite(height) & (height >= 0)
        if nodata is not None:
            valid &= height != nodata
        height[~valid] = np.nan
        height *= self.height_scale  # M4Heights heights are metres, unlike US3D centimetres.

        # NOGEO tiles are paired by ID and cover the same extent, at 1m and 10m.
        # Nearest-neighbour preserves zeros, invalid pixels, and measured heights.
        if self.mode == 'train' and height.shape != image.shape[:2]:
            height = cv2.resize(height, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        sample = self.transform({'image': image, 'depth': height})
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        sample['valid_mask'] = torch.isfinite(sample['depth']) & (sample['depth'] >= 0)
        sample['image_path'] = str(self.root / record['image']['path']) + (
            '::' + record['image']['member'] if 'member' in record['image'] else '')
        return sample
