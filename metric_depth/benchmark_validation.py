import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

from dataset.us3d import US3D
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads, checkpoint_height_activation
from depth_anything_v2.dinov2_layers.attention import XFORMERS_AVAILABLE, set_attention_backend
from run import MODEL_CONFIGS, extract_model_state
from util.inference import predict_height_tiled, tile_starts
from util.metric import eval_depth


def build_cases(experiment, include_full=False):
    cases = [dict(name='sdpa_518', backend='sdpa', tile_size=518, overlap=126, batch_size=4)]
    if experiment == 'attention':
        cases.insert(0, dict(name='original_518', backend='manual', tile_size=518, overlap=126, batch_size=4))
    else:
        cases += [
            dict(name='sdpa_1036_batch1', backend='sdpa', tile_size=1036, overlap=24, batch_size=1),
            dict(name='sdpa_1036_batch4', backend='sdpa', tile_size=1036, overlap=24, batch_size=4),
        ]
        if include_full:
            cases.append(dict(name='sdpa_2072', backend='sdpa', tile_size=2072, overlap=0, batch_size=1))
    return cases


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def predict(model, image, case):
    return predict_height_tiled(
        model, image, case['tile_size'], case['overlap'], case['batch_size']
    )[0]


def benchmark_case(model, samples, case, repeats, device, max_depth, references, atol, rtol):
    set_attention_backend(model, case['backend'])
    with torch.inference_mode():
        warm_image = samples[0]['image'].unsqueeze(0).to(device)
        warm_prediction = predict(model, warm_image, case)
        synchronize(device)
        del warm_image, warm_prediction
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        metric_values = {}
        predictions = []
        difference_sum = 0.0
        difference_max = 0.0
        difference_count = 0
        outside_tolerance = 0
        valid_pixels = 0
        tile_counts = []
        for repeat in range(repeats):
            total_seconds = 0.0
            for index, sample in enumerate(samples):
                image = sample['image'].unsqueeze(0).to(device)
                synchronize(device)
                started = time.perf_counter()
                prediction = predict(model, image, case)
                synchronize(device)
                total_seconds += time.perf_counter() - started
                if repeat == 0:
                    prediction_cpu = prediction.cpu()
                    target = sample['depth']
                    mask = sample['valid_mask'] & (target >= 0) & (target <= max_depth)
                    if not mask.any():
                        raise ValueError(f"No valid target pixels: {sample['image_path']}")
                    valid_pixels += int(mask.sum())
                    for key, value in eval_depth(prediction_cpu[mask], target[mask]).items():
                        metric_values.setdefault(key, []).append(value.item())
                    if references is None:
                        predictions.append(prediction_cpu)
                    else:
                        difference = (prediction_cpu - references[index]).abs()
                        difference_sum += difference.double().sum().item()
                        difference_max = max(difference_max, difference.max().item())
                        difference_count += difference.numel()
                        outside_tolerance += int((difference > atol + rtol * references[index].abs()).sum())
                    height, width = image.shape[-2:]
                    step = case['tile_size'] - case['overlap']
                    tile_counts.append(
                        len(tile_starts(height, case['tile_size'], step))
                        * len(tile_starts(width, case['tile_size'], step))
                    )
                del prediction, image
                print(f"  {case['name']}: repeat {repeat + 1}/{repeats}, image {index + 1}/{len(samples)}", flush=True)
            durations.append(total_seconds)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else None
    seconds_per_image = statistics.median(durations) / len(samples)
    result = {
        **case,
        'status': 'ok',
        'images': len(samples),
        'valid_pixels': valid_pixels,
        'tiles_per_image': tile_counts,
        'repeat_seconds': durations,
        'median_seconds_per_image': seconds_per_image,
        'estimated_inference_seconds_592_images_4_gpus': seconds_per_image * 592 / 4,
        'peak_allocated_gib': peak_allocated,
        'peak_reserved_gib': peak_reserved,
        'metrics': {key: statistics.mean(values) for key, values in metric_values.items()},
    }
    if references is not None:
        result['prediction_difference_from_first_case'] = {
            'mean_abs_m': difference_sum / difference_count,
            'max_abs_m': difference_max,
            'pixels_outside_tolerance': outside_tolerance,
            'pixel_count': difference_count,
            'all_close': outside_tolerance == 0,
            'atol_m': atol,
            'rtol': rtol,
        }
    return result, predictions


def profile_attention(model, sample, case, device):
    from torch.profiler import ProfilerActivity, profile

    set_attention_backend(model, case['backend'])
    image = sample['image'].unsqueeze(0).to(device)
    image = image[:, :, :case['tile_size'], :case['tile_size']]
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as profiler:
        prediction = predict(model, image, {**case, 'batch_size': 1})
        synchronize(device)
    del prediction, image
    return sorted({event.key for event in profiler.key_averages() if 'scaled_dot_product' in event.key})


def main():
    parser = argparse.ArgumentParser(description='Compare validation speed and predictions with one fixed checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val-split', default='dataset/splits/us3d/val_fold5.txt')
    parser.add_argument('--experiment', choices=['attention', 'tiles'], default='attention')
    parser.add_argument('--samples', type=int, default=12, help='Random validation images; 0 selects the complete split')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder', choices=list(MODEL_CONFIGS))
    parser.add_argument('--include-full', action='store_true', help='Also try 2072-pixel tiles in the tiles experiment')
    parser.add_argument('--profile-attention', action='store_true')
    parser.add_argument('--prediction-atol', type=float, default=0.001)
    parser.add_argument('--prediction-rtol', type=float, default=0.0001)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples < 0 or args.repeats < 1:
        parser.error('Expected samples >= 0 and repeats >= 1')
    if args.prediction_atol < 0 or args.prediction_rtol < 0:
        parser.error('Prediction tolerances must be nonnegative')
    if args.include_full and args.experiment != 'tiles':
        parser.error('--include-full requires --experiment tiles')
    if not torch.cuda.is_available():
        parser.error('Run this benchmark inside a GPU allocation')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_args = checkpoint.get('args', {})
    state = extract_model_state(checkpoint)
    encoder = args.encoder or saved_args.get('encoder', 'vitl')
    max_depth = saved_args.get('max_depth', 250.0)
    model_class = DepthAnythingV2withHeads if any(k.startswith('scale_head.') for k in state) else DepthAnythingV2
    model = model_class(**MODEL_CONFIGS[encoder], max_depth=max_depth,
                        height_activation=checkpoint_height_activation(checkpoint))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    del checkpoint, state
    dataset = US3D(args.val_split, 'val', sampling_mode='native')
    indices = list(range(len(dataset)))
    if args.samples and args.samples < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.samples))
    started = time.perf_counter()
    cached = len(indices) <= 16
    samples = [dataset[index] for index in indices] if cached else torch.utils.data.Subset(dataset, indices)
    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'val_split': str(Path(args.val_split).resolve()),
        'experiment': args.experiment,
        'seed': args.seed,
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(device),
        'precision': 'float32; autocast disabled',
        'xformers_available': XFORMERS_AVAILABLE,
        'original_attention': 'xformers' if XFORMERS_AVAILABLE else 'manual',
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'timing_scope': 'Inference and tile blending only; synchronized; excludes CPU loading, transfers and metrics',
        'projection_scope': '592-image/4-GPU estimate covers inference only and assumes ideal distribution',
        'averaging': 'Mean of per-image metrics, matching train.py',
        'preload_seconds': time.perf_counter() - started,
        'data_loading': 'preloaded CPU samples' if cached else 'streamed CPU samples',
        'indices': indices,
        'images': [dataset.filelist[index][0] for index in indices],
        'image_shapes': [list(sample['depth'].shape) for sample in samples] if cached else None,
        'results': [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    references = None
    reference_result = None
    for case in build_cases(args.experiment, args.include_full):
        print(f"Starting {case['name']}", flush=True)
        try:
            result, predictions = benchmark_case(
                model, samples, case, args.repeats, device, max_depth,
                references, args.prediction_atol, args.prediction_rtol,
            )
            if references is None:
                references = predictions
                reference_result = result
                report['reference_case'] = case['name']
            else:
                result['speedup_vs_reference'] = (
                    reference_result['median_seconds_per_image'] / result['median_seconds_per_image']
                )
                result['metric_delta_from_reference'] = {
                    key: value - reference_result['metrics'][key]
                    for key, value in result['metrics'].items()
                }
            if args.profile_attention:
                result['sdpa_operators'] = profile_attention(model, samples[0], case, device)
        except torch.cuda.OutOfMemoryError:
            result = {**case, 'status': 'out_of_memory'}
        report['results'].append(result)
        print(json.dumps(result, indent=2), flush=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Saved report: {output}', flush=True)
    if any(result['status'] != 'ok' for result in report['results']):
        print('Some variants ran out of memory; see the report.', flush=True)


if __name__ == '__main__':
    main()
