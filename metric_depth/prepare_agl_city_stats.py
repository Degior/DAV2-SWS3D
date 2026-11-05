import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from dataset.us3d import read_us3d_height, read_us3d_split


def us3d_city(image_path):
    city = Path(image_path).name.split('_', 1)[0].upper()
    if city not in {'ARG', 'ATL', 'JAX', 'OMA'}:
        raise ValueError(f"Cannot identify the US3D city from {image_path}")
    return city


def split_sha256(train_split):
    return hashlib.sha256(Path(train_split).read_bytes()).hexdigest()


def validate_agl_city_stats(stats, train_split, max_depth):
    if not isinstance(stats, dict) or stats.get('train_split_sha256') != split_sha256(train_split):
        raise ValueError('AGL city statistics must be computed from the current --train-split')
    if stats.get('max_depth_m') != max_depth:
        raise ValueError('AGL city statistics must use the current --max-depth')
    expected = {us3d_city(image) for image, _ in read_us3d_split(train_split)}
    cities = stats.get('cities')
    if not isinstance(cities, dict) or set(cities) != expected:
        raise ValueError(f"AGL city statistics must cover the training cities: {sorted(expected)}")
    for city, info in cities.items():
        if not isinstance(info, dict):
            raise ValueError(f"Invalid AGL city statistics for {city}")
        mean = info.get('mean_m')
        variance = info.get('variance_m2')
        if not isinstance(mean, (float, int)) or not 0 <= mean <= max_depth:
            raise ValueError(f"Invalid AGL mean for {city}")
        if not isinstance(variance, (float, int)) or not math.isfinite(variance) or variance <= 0:
            raise ValueError(f"AGL variance for {city} must be finite and positive")


def calculate_agl_city_stats(train_split, max_depth):
    digest = split_sha256(train_split)
    records = read_us3d_split(train_split)
    totals = {us3d_city(image): [0, 0.0, 0.0] for image, _ in records}
    for index, (image, height_path) in enumerate(records, start=1):
        height = read_us3d_height(height_path)
        valid = np.isfinite(height) & (height >= 0) & (height <= max_depth)
        values = height[valid].astype(np.float64)
        total = totals[us3d_city(image)]
        total[0] += values.size
        total[1] += values.sum().item()
        total[2] += np.square(values).sum().item()
        if index % 100 == 0 or index == len(records):
            print(f"Read {index}/{len(records)} training height maps", flush=True)

    cities = {}
    for city, (count, total, squared_total) in totals.items():
        if not count:
            raise ValueError(f"No valid training height pixels for {city}")
        mean = total / count
        cities[city] = {
            'mean_m': mean,
            'variance_m2': squared_total / count - mean ** 2,
            'valid_pixels': count,
        }
    stats = {'train_split_sha256': digest, 'max_depth_m': max_depth, 'cities': cities}
    validate_agl_city_stats(stats, train_split, max_depth)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute AGL city statistics from US3D training labels')
    parser.add_argument('--train-split', default='dataset/splits/us3d/train_fold5.txt')
    parser.add_argument('--max-depth', type=float, default=250.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error('--max-depth must be finite and positive')

    stats = calculate_agl_city_stats(args.train_split, args.max_depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        temp_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temp_path, args.output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved AGL city statistics: {args.output}")


if __name__ == '__main__':
    main()
