import argparse
import gc
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset.us3d import US3D
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads, checkpoint_height_activation
from depth_anything_v2.dinov2_layers.attention import set_attention_backend
from run import MODEL_CONFIGS, extract_model_state
from util.inference import predict_height_tiled, tile_starts
from util.metric import eval_depth


HEIGHT_GROUPS = ['zero', 'gt_0_lt_0_1', 'ge_0_1_lt_1', 'ge_1_lt_5', 'ge_5_lt_20', 'ge_20']


def height_masks(target, valid):
    return {
        'all_valid': valid,
        'zero': valid & (target == 0),
        'gt_0_lt_0_1': valid & (target > 0) & (target < 0.1),
        'ge_0_1_lt_1': valid & (target >= 0.1) & (target < 1),
        'ge_1_lt_5': valid & (target >= 1) & (target < 5),
        'ge_5_lt_20': valid & (target >= 5) & (target < 20),
        'ge_20': valid & (target >= 20),
    }


def spatial_masks(shape, tile_size, overlap, device, band=8):
    height, width = shape
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    y_band = torch.zeros(height, device=device, dtype=torch.bool)
    x_band = torch.zeros(width, device=device, dtype=torch.bool)
    for length, coordinates, destination in ((height, ys, y_band), (width, xs, x_band)):
        for start in tile_starts(length, tile_size, tile_size - overlap):
            for boundary in (start, start + tile_size):
                if 0 < boundary < length:
                    destination |= (coordinates - boundary).abs() < band
    outer = (ys[:, None] < band) | (ys[:, None] >= height - band)
    outer = outer | (xs[None, :] < band) | (xs[None, :] >= width - band)
    boundary = (y_band[:, None] | x_band[None, :]) & ~outer
    return {'outer_border_8px': outer, 'tile_boundaries_8px': boundary,
            'away_from_borders': ~(outer | boundary)}


class Statistics:
    def __init__(self):
        self.images = 0
        self.pixels = 0
        self.metric_sums = {}
        self.metric_counts = {}
        self.sums = dict(abs_error=0.0, squared_error=0.0, error=0.0,
                         gt=0.0, pred=0.0, squared_log_error=0.0)
        self.positive_pixels = 0
        self.pred_below_1mm = 0
        self.pred_at_log_floor = 0

    def update(self, prediction, target, mask):
        count = int(mask.sum())
        if not count:
            return None
        pred, gt = prediction[mask], target[mask]
        metrics = {key: float(value) for key, value in eval_depth(pred, gt).items()}
        self.images += 1
        self.pixels += count
        for key, value in metrics.items():
            self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value
            self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
        error = (pred - gt).double()
        self.sums['abs_error'] += error.abs().sum().item()
        self.sums['squared_error'] += error.square().sum().item()
        self.sums['error'] += error.sum().item()
        self.sums['gt'] += gt.double().sum().item()
        self.sums['pred'] += pred.double().sum().item()
        positive = gt > 1e-6
        self.positive_pixels += int(positive.sum())
        self.pred_below_1mm += int((positive & (pred < 0.001)).sum())
        self.pred_at_log_floor += int((positive & (pred <= 1e-6)).sum())
        log_error = pred[positive].clamp_min(1e-6).double().log() - gt[positive].double().log()
        self.sums['squared_log_error'] += log_error.square().sum().item()
        return {'pixels': count, 'metrics': metrics}

    def report(self, total_squared_error=None, total_squared_log_error=None):
        result = {'images': self.images, 'pixels': self.pixels,
                  'mean_per_image': {key: value / self.metric_counts[key]
                                     for key, value in self.metric_sums.items()},
                  'images_per_metric': self.metric_counts.copy(),
                  'positive_pixels': self.positive_pixels}
        if self.pixels:
            result['pixel_weighted'] = {
                'mae': self.sums['abs_error'] / self.pixels,
                'rmse': math.sqrt(self.sums['squared_error'] / self.pixels),
                'bias': self.sums['error'] / self.pixels,
                'mean_gt_m': self.sums['gt'] / self.pixels,
                'mean_prediction_m': self.sums['pred'] / self.pixels,
            }
        if self.positive_pixels:
            result['positive_pixel_diagnostics'] = {
                'rmse_log_pixel_weighted': math.sqrt(self.sums['squared_log_error'] / self.positive_pixels),
                'prediction_below_1mm_fraction': self.pred_below_1mm / self.positive_pixels,
                'prediction_at_log_floor_fraction': self.pred_at_log_floor / self.positive_pixels,
            }
        if total_squared_error:
            result['share_of_total_squared_error'] = self.sums['squared_error'] / total_squared_error
        if total_squared_log_error:
            result['share_of_total_squared_log_error'] = self.sums['squared_log_error'] / total_squared_log_error
        return result


def checkpoint_config(checkpoint):
    saved = checkpoint.get('args', {})
    mode = saved.get('sampling_mode', 'resize')
    if mode not in {'resize', 'native'}:
        raise ValueError(f'Unsupported sampling mode: {mode}')
    config = {
        'encoder': saved.get('encoder', 'vitl'),
        'sampling_mode': mode,
        'img_size': saved.get('img_size', 518),
        'val_tile_overlap': saved.get('val_tile_overlap', 126),
        'max_depth': saved.get('max_depth', 250.0),
        'height_activation': checkpoint_height_activation(checkpoint),
        'attention_backend': saved.get('attention_backend', 'manual'),
    }
    if mode == 'native':
        if config['img_size'] < 14 or config['img_size'] % 14:
            raise ValueError('Native tile size must be a positive multiple of 14')
        if not 0 <= config['val_tile_overlap'] < config['img_size']:
            raise ValueError('Invalid checkpoint tile overlap')
    return config


def inference_config(saved_config, tile_size=None, tile_overlap=None):
    config = saved_config.copy()
    if tile_size is not None or tile_overlap is not None:
        if config['sampling_mode'] != 'native':
            raise ValueError('Tile overrides require a checkpoint with native inference')
        if tile_size is not None:
            config['img_size'] = tile_size
        if tile_overlap is not None:
            config['val_tile_overlap'] = tile_overlap
        if config['img_size'] < 14 or config['img_size'] % 14:
            raise ValueError('Tile size must be a positive multiple of 14')
        if not 0 <= config['val_tile_overlap'] < config['img_size']:
            raise ValueError('Tile overlap must be between 0 and tile size - 1')
    return config


@torch.inference_mode()
def predict(model, image, target_shape, config, tile_batch_size):
    if config['sampling_mode'] == 'native':
        prediction = predict_height_tiled(
            model, image, config['img_size'], config['val_tile_overlap'], tile_batch_size
        )[0]
    else:
        output = model(image)
        output = output['depth'] if isinstance(output, dict) else output
        prediction = F.interpolate(output[:, None], target_shape, mode='bilinear', align_corners=True)[0, 0]
    if prediction.shape != target_shape or not torch.isfinite(prediction).all():
        raise ValueError('Prediction must be finite and match the original target shape')
    return prediction


def render_example(image_path, target, prediction, valid, record, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    gt, pred, mask = target.cpu().numpy(), prediction.cpu().numpy(), valid.cpu().numpy()
    error = pred - gt
    vmax = max(1.0, float(np.percentile(np.concatenate([gt[mask], pred[mask]]), 99.5)))
    error_limit = max(0.1, float(np.percentile(np.abs(error[mask]), 99)))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Original RGB')
    heights = plt.get_cmap('viridis').copy()
    heights.set_bad('#bdbdbd')
    signed_colors = plt.get_cmap('coolwarm').copy()
    signed_colors.set_bad('#bdbdbd')
    absolute_colors = plt.get_cmap('magma').copy()
    absolute_colors.set_bad('#bdbdbd')
    for ax, values, title in ((axes[0, 1], gt, 'Ground truth (m)'),
                              (axes[0, 2], pred, 'Prediction (m)')):
        im = ax.imshow(np.ma.array(values, mask=~mask), cmap=heights, vmin=0, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.75, extend='max')
    im = axes[1, 0].imshow(np.ma.array(error, mask=~mask), cmap=signed_colors,
                           vmin=-error_limit, vmax=error_limit)
    axes[1, 0].set_title('Prediction minus ground truth (m)')
    fig.colorbar(im, ax=axes[1, 0], shrink=0.75, extend='both')
    im = axes[1, 1].imshow(np.ma.array(np.abs(error), mask=~mask), cmap=absolute_colors,
                           vmin=0, vmax=error_limit)
    axes[1, 1].set_title('Absolute error (m)')
    fig.colorbar(im, ax=axes[1, 1], shrink=0.75, extend='max')
    indices = np.flatnonzero(mask)
    if len(indices) > 20000:
        indices = np.random.default_rng(record['index']).choice(indices, 20000, replace=False)
    axes[1, 2].scatter(gt.ravel()[indices], pred.ravel()[indices], s=2, alpha=0.15)
    maximum = max(1.0, float(gt[mask].max()), float(pred[mask].max()))
    axes[1, 2].plot([0, maximum], [0, maximum], 'r--')
    axes[1, 2].set(xlabel='Ground truth (m)', ylabel='Prediction (m)', title='Sampled valid pixels')
    for ax in axes.flat[:5]:
        ax.axis('off')
    metrics = record['all_valid']['metrics']
    fig.suptitle(
        f"{Path(image_path).name} | MAE {metrics['mae']:.4f} m | RMSE {metrics['rmse']:.4f} m | "
        f"RMSE log {metrics.get('rmse_log', float('nan')):.4f}\n"
        'Gray = invalid target; heatmap limits use percentiles; scatter uses the full height range'
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Collect full-image US3D validation statistics by target height')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val-split', default='dataset/splits/us3d/val_fold5.txt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tile-batch-size', type=int, default=4)
    parser.add_argument('--tile-size', type=int, help='Override the native inference tile size without resizing pixels')
    parser.add_argument('--tile-overlap', type=int, help='Override the native inference overlap in pixels')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--worst', type=int, default=6)
    parser.add_argument('--random-examples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.tile_batch_size < 1 or min(args.num_workers, args.max_images, args.worst, args.random_examples) < 0:
        parser.error('Expected positive tile batch size and nonnegative worker/image counts')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run inside a GPU allocation')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_config = checkpoint_config(checkpoint)
    try:
        config = inference_config(saved_config, args.tile_size, args.tile_overlap)
    except ValueError as error:
        parser.error(str(error))
    metadata = {'checkpoint': str(Path(args.checkpoint).resolve()),
                'completed_epochs': int(checkpoint.get('epoch', -1)) + 1,
                'saved_best_rmse': checkpoint.get('best_rmse'),
                'val_split': str(Path(args.val_split).resolve()), 'config': config,
                'checkpoint_config': saved_config,
                'torch': str(torch.__version__), 'device': str(device),
                'tile_batch_size': args.tile_batch_size,
                'averaging': 'mean_per_image matches train.py; pixel_weighted pools pixels separately',
                'error_shares': 'Shares use squared pixel-error sums, not the mean of per-image RMSE values',
                'height_groups': 'Disjoint GT ranges: zero, (0,0.1), [0.1,1), [1,5), [5,20), [20,max_depth]',
                'log_error': 'Positive GT only; prediction clamped to 1e-6, matching train.py',
                'spatial_groups': 'Native inference: bands near tile edges, outer image border, and remaining pixels'}
    state = extract_model_state(checkpoint)
    model_class = DepthAnythingV2withHeads if any(k.startswith('scale_head.') for k in state) else DepthAnythingV2
    model = model_class(**MODEL_CONFIGS[config['encoder']], max_depth=config['max_depth'],
                        height_activation=config['height_activation'])
    model.load_state_dict(state, strict=True)
    set_attention_backend(model, config['attention_backend'])
    model.to(device).eval()
    del checkpoint, state
    gc.collect()
    dataset = US3D(args.val_split, 'val', size=(config['img_size'], config['img_size']),
                   sampling_mode=config['sampling_mode'])
    selected = list(range(min(args.max_images or len(dataset), len(dataset))))
    loader = DataLoader(Subset(dataset, selected), batch_size=1, num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda',
                        prefetch_factor=1 if args.num_workers else None)
    groups = {name: Statistics() for name in ['all_valid'] + HEIGHT_GROUPS}
    cities, spatial, records = {}, {}, []
    print(json.dumps(metadata, indent=2), flush=True)
    with (output / 'per_image.jsonl').open('w', encoding='utf-8') as file:
        for index, sample in zip(selected, loader):
            image = sample['image'].to(device, non_blocking=True)
            target = sample['depth'][0].to(device, non_blocking=True)
            prediction = predict(model, image, target.shape, config, args.tile_batch_size)
            valid = sample['valid_mask'][0].to(device) & (target >= 0) & (target <= config['max_depth'])
            if not valid.any():
                raise ValueError(f"No valid target pixels: {sample['image_path'][0]}")
            image_path = sample['image_path'][0]
            record = {'index': index, 'image': image_path, 'shape': list(target.shape), 'height_groups': {}}
            for name, mask in height_masks(target, valid).items():
                current = groups[name].update(prediction, target, mask)
                if name == 'all_valid':
                    record['all_valid'] = current
                else:
                    record['height_groups'][name] = current
            city = Path(image_path).stem.split('_')[0]
            cities.setdefault(city, Statistics()).update(prediction, target, valid)
            if config['sampling_mode'] == 'native':
                for name, mask in spatial_masks(target.shape, config['img_size'],
                                                 config['val_tile_overlap'], device).items():
                    spatial.setdefault(name, Statistics()).update(prediction, target, valid & mask)
            records.append(record)
            file.write(json.dumps(record, allow_nan=False) + '\n')
            file.flush()
            metrics = record['all_valid']['metrics']
            print(f"[{len(records)}/{len(selected)}] {Path(image_path).name}: "
                  f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"RMSE_log={metrics.get('rmse_log', float('nan')):.4f}", flush=True)
    total = groups['all_valid']
    report = {**metadata, 'validation_images': len(records), 'split_images': len(dataset),
              'complete_split': len(records) == len(dataset),
              'height_groups': {name: group.report(total.sums['squared_error'], total.sums['squared_log_error'])
                                for name, group in groups.items()},
              'cities': {name: group.report() for name, group in cities.items()},
              'spatial_groups': {name: group.report() for name, group in spatial.items()}}
    rankings = {key: sorted(records, key=lambda r: r['all_valid']['metrics'].get(key, -math.inf), reverse=True)
                for key in ('rmse', 'mae', 'rmse_log')}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'rankings.json').write_text(json.dumps(rankings, indent=2, allow_nan=False), encoding='utf-8')
    print('Full-image mean metrics:', json.dumps(total.report()['mean_per_image'], indent=2), flush=True)
    for name in HEIGHT_GROUPS:
        print(name, json.dumps(report['height_groups'][name]), flush=True)
    if not args.no_figures:
        examples = {}
        for key in ('rmse', 'rmse_log'):
            for record in rankings[key][:args.worst]:
                examples.setdefault(record['index'], []).append(f'worst_{key}')
        for record in random.Random(args.seed).sample(records, min(args.random_examples, len(records))):
            examples.setdefault(record['index'], []).append('random')
        figure_dir = output / 'figures'
        figure_dir.mkdir(exist_ok=True)
        lookup = {record['index']: record for record in records}
        for index, reasons in examples.items():
            sample = dataset[index]
            image = sample['image'].unsqueeze(0).to(device)
            target = sample['depth'].to(device)
            valid = sample['valid_mask'].to(device) & (target >= 0) & (target <= config['max_depth'])
            prediction = predict(model, image, target.shape, config, args.tile_batch_size)
            destination = figure_dir / f"{index:04d}_{'_'.join(reasons)}_{Path(sample['image_path']).stem}.png"
            render_example(sample['image_path'], target, prediction, valid, lookup[index], destination)
            print(f'Saved {destination}', flush=True)
    print(f'Saved statistics: {output}', flush=True)


if __name__ == '__main__':
    main()
