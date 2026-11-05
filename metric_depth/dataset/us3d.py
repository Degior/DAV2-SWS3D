import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, NativeCrop


def read_us3d_split(filelist_path):
    with open(filelist_path, 'r') as f:
        records = [
            line.strip().split(maxsplit=2)
            for line in f
            if line.strip() and not line.lstrip().startswith('#')
        ]
    invalid = next((record for record in records if len(record) < 2), None)
    if invalid is not None:
        raise ValueError(f"Unexpected US3D split record: {invalid}")
    if not records:
        raise ValueError(f"US3D split is empty: {filelist_path}")
    return [record[:2] for record in records]


def read_us3d_height(height_path):
    height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
    if height is None:
        raise FileNotFoundError(f"Height map not found: {height_path}")
    if height.ndim != 2:
        raise ValueError(f"Height map must be single-channel: {height_path}, shape={height.shape}")
    height = height.astype('float32')
    height[height == 65535] = np.nan
    return height * 0.01


class US3D(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518), sampling_mode='resize'):
        self.mode = mode
        self.size = size
        if sampling_mode not in {'resize', 'native'}:
            raise ValueError(f'Unsupported sampling mode: {sampling_mode}')
        self.sampling_mode = sampling_mode

        # Baseline mode permanently discards every column after RGB and AGL.
        self.filelist = read_us3d_split(filelist_path)

        net_w, net_h = size
        transforms = []
        if sampling_mode == 'resize':
            transforms.append(Resize(
                width=net_w,
                height=net_h,
                resize_target=(mode == 'train'),
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ))
        elif mode == 'train':
            transforms.append(NativeCrop((net_h, net_w)))
        self.transform = Compose(transforms + [
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == 'train' and sampling_mode == 'resize' else []))

    # @staticmethod
    # def transform_height(z):
    #     return np.sign(z) * np.log1p(np.abs(z))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        rec = self.filelist[idx]
        image_path = rec[0]
        height_path = rec[1]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        height_map = read_us3d_height(height_path)
        if height_map.shape != image.shape[:2]:
            raise ValueError(
                f"Height map must be single-channel and match the image: "
                f"image={image.shape[:2]}, height={height_map.shape}"
            )

        # height_map = self.transform_height(height_map)

        sample = {'image': image, 'depth': height_map}

        sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        # Zero is a valid background height for object-height maps. Invalid
        # pixels are represented by the source nodata value and converted to NaN.
        sample['valid_mask'] = torch.isfinite(sample['depth']) & (sample['depth'] >= 0)

        sample['image_path'] = image_path

        return sample
