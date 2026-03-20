import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class US3D(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size

        with open(filelist_path, 'r') as f:
            lines = f.read().splitlines()

        self.filelist = [line.strip().split() for line in lines]

        net_w, net_h = size

        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=(mode == 'train'),
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        rec = self.filelist[idx]

        image_path = rec[0]
        height_path = rec[1]

        semantic_path = None
        json_path = None

        if len(rec) == 3:
            json_path = rec[2]
        elif len(rec) == 4:
            semantic_path = rec[2]
            json_path = rec[3]
        else:
            raise ValueError(f"Unexpected filelist format: {rec}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        height_map = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        if height_map is None:
            raise FileNotFoundError(f"Height map not found: {height_path}")

        height_map = height_map.astype('float32')
        height_map[height_map == 65535] = np.nan
        height_map = height_map * 0.01

        if json_path is None:
            raise ValueError("JSON path required for scale/angle")

        with open(json_path, 'r') as f:
            meta = json.load(f)

        scale = float(meta["scale"])
        angle = float(meta["angle"])

        scale = np.log(scale + 1e-6)

        semantics = None
        if semantic_path:
            semantics = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            if semantics is not None:
                semantics = semantics.astype('int64')

        sample = {
            'image': image,
            'depth': height_map
        }

        sample = self.transform(sample)

        image = torch.from_numpy(sample['image']).float()
        depth = torch.from_numpy(sample['depth']).float()

        scale = torch.tensor(scale).float()
        angle = torch.tensor(angle).float()

        valid_mask = torch.isfinite(depth) & (depth > 0)

        output = {
            'image': image,
            'depth': depth,
            'scale': scale,
            'angle': angle,
            'valid_mask': valid_mask,
            'image_path': image_path
        }

        if semantics is not None:
            output['semantics'] = torch.from_numpy(semantics).long()

        return output