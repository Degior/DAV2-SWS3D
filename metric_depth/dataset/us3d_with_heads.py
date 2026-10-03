import cv2
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class US3DWH(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518), angle_unit='radians'):
        self.mode = mode
        self.size = size
        if angle_unit not in {'radians', 'degrees'}:
            raise ValueError(f"Unsupported angle unit: {angle_unit}")
        self.angle_unit = angle_unit

        with open(filelist_path, 'r') as f:
            lines = f.read().splitlines()

        self.filelist = [
            line.strip().split()
            for line in lines
            if line.strip() and not line.lstrip().startswith('#')
        ]
        invalid = [record for record in self.filelist if len(record) != 3]
        if invalid:
            raise ValueError(f"Unexpected US3DWH split record: {invalid[0]}")
        if not self.filelist:
            raise ValueError(f"US3DWH split is empty: {filelist_path}")

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

        json_path = rec[2]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        height_map = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        if height_map is None:
            raise FileNotFoundError(f"Height map not found: {height_path}")
        if height_map.ndim != 2 or height_map.shape != image.shape[:2]:
            raise ValueError(
                f"Height map must be single-channel and match the image: "
                f"image={image.shape[:2]}, height={height_map.shape}"
            )

        height_map = height_map.astype('float32')
        height_map[height_map == 65535] = np.nan
        height_map = height_map * 0.01

        with open(json_path, 'r') as f:
            meta = json.load(f)

        scale = float(meta["scale"])
        angle = float(meta["angle"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scale must be finite and positive in {json_path}: {scale}")
        if not math.isfinite(angle):
            raise ValueError(f"Angle must be finite in {json_path}: {angle}")
        if self.angle_unit == 'degrees':
            angle = math.radians(angle)
        angle %= 2 * math.pi

        sample = {
            'image': image,
            'depth': height_map
        }
        sample = self.transform(sample)

        image = torch.from_numpy(sample['image']).float()
        depth = torch.from_numpy(sample['depth']).float()

        scale = torch.tensor(scale).float()
        angle = torch.tensor(angle).float()

        valid_mask = torch.isfinite(depth) & (depth >= 0)

        output = {
            'image': image,
            'depth': depth,
            'scale': scale,
            'angle': angle,
            'valid_mask': valid_mask,
            'image_path': image_path,
        }

        return output
