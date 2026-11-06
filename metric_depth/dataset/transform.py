import random

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = \
                F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width),
                              mode='nearest').numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class NativeCrop:
    UNIFORM, MID_HEIGHT, TALL, FALLBACK = range(4)

    def __init__(self, size, height_crop_prob=0.0, max_depth=250.0, height_crop_mid_first_prob=0.5):
        self.height, self.width = size
        self.height_crop_prob = height_crop_prob
        self.max_depth = max_depth
        self.height_crop_mid_first_prob = height_crop_mid_first_prob
        self.min_target_pixels = max(1, (self.height * self.width + 199) // 200)

    def _sample_target_window(self, mask):
        target_pixels = np.flatnonzero(mask)
        if target_pixels.size < self.min_target_pixels:
            return None
        max_top = mask.shape[0] - self.height
        max_left = mask.shape[1] - self.width
        for _ in range(8):
            y, x = divmod(target_pixels[np.random.randint(target_pixels.size)], mask.shape[1])
            top = np.random.randint(max(0, y - self.height + 1), min(y, max_top) + 1)
            left = np.random.randint(max(0, x - self.width + 1), min(x, max_left) + 1)
            crop = mask[top:top + self.height, left:left + self.width]
            if np.count_nonzero(crop) >= self.min_target_pixels:
                return top, left
        return None

    def _height_window(self, depth):
        bands = [(self.MID_HEIGHT, 5.0, 20.0), (self.TALL, 20.0, np.inf)]
        if self.height_crop_mid_first_prob == 0.5:
            tall_first = np.random.randint(2)
        else:
            tall_first = np.random.random() >= self.height_crop_mid_first_prob
        if tall_first:
            bands.reverse()
        for band, lower, upper in bands:
            mask = (depth >= lower) & (depth < upper) & (depth <= self.max_depth)
            window = self._sample_target_window(mask)
            if window is not None:
                return window, band
        return None, self.FALLBACK

    def __call__(self, sample):
        height, width = sample['image'].shape[:2]
        pad_height = max(0, self.height - height)
        pad_width = max(0, self.width - width)
        if pad_height or pad_width:
            sample['image'] = np.pad(
                sample['image'], ((0, pad_height), (0, pad_width), (0, 0)), mode='edge'
            )
            sample['depth'] = np.pad(
                sample['depth'], ((0, pad_height), (0, pad_width)),
                mode='constant', constant_values=np.nan,
            )
        height, width = sample['image'].shape[:2]
        window, band = None, self.UNIFORM
        if self.height_crop_prob > 0:
            if np.random.random() < self.height_crop_prob:
                window, band = self._height_window(sample['depth'])
            sample['height_crop_band'] = band
        if window is None:
            top = np.random.randint(0, height - self.height + 1)
            left = np.random.randint(0, width - self.width + 1)
        else:
            top, left = window
        sample['image'] = sample['image'][top:top + self.height, left:left + self.width]
        sample['depth'] = sample['depth'][top:top + self.height, left:left + self.width]
        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['image'].shape[-2:]
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'

        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]

        sample['image'] = sample['image'][:, h_start: h_end, w_start: w_end]

        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start: h_end, w_start: w_end]

        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start: h_end, w_start: w_end]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start: h_end, w_start: w_end]

        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.ascontiguousarray(np.flip(sample['image'], axis=1))
            sample['depth'] = np.ascontiguousarray(np.flip(sample['depth'], axis=1))

            if 'semantics' in sample and sample['semantics'] is not None:
                sample['semantics'] = np.ascontiguousarray(
                    np.flip(sample['semantics'], axis=1)
                )

        return sample
