import argparse
import glob
import os

import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def extract_model_state(checkpoint):
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError('Checkpoint does not contain a model state dict')
    state = {
        key.removeprefix('module.'): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not state:
        raise ValueError('Checkpoint model state dict is empty')
    return state


def collect_images(path):
    if os.path.isfile(path):
        if path.lower().endswith('.txt'):
            with open(path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        return [path]
    return sorted(
        filename
        for filename in glob.glob(os.path.join(path, '**', '*'), recursive=True)
        if os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    )


def colorize_height(height, min_height, max_height, cmap, grayscale):
    normalized = np.clip((height - min_height) / max(max_height - min_height, 1e-8), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.round(normalized * 255).astype(np.uint8)
    if grayscale:
        return np.repeat(image[..., None], 3, axis=-1)
    return (cmap(image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def infer_image_tiled(model, raw_image, tile_size=518, overlap=126):
    """Predict at source resolution and blend overlapping square tiles."""
    if tile_size < 14 or tile_size % 14:
        raise ValueError('tile_size must be a positive multiple of 14')
    if not 0 <= overlap < tile_size:
        raise ValueError('overlap must be between 0 and tile_size - 1')

    height, width = raw_image.shape[:2]
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    if pad_height or pad_width:
        raw_image = np.pad(
            raw_image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode='edge',
        )

    padded_height, padded_width = raw_image.shape[:2]
    step = tile_size - overlap
    ys = tile_starts(padded_height, tile_size, step)
    xs = tile_starts(padded_width, tile_size, step)
    print(f'Tiled inference: {len(ys)} x {len(xs)} = {len(ys) * len(xs)} tiles')

    # Positive edge weights avoid uncovered pixels at the image boundary.
    taper = 0.05 + 0.95 * np.hanning(tile_size).astype(np.float32)
    weights = np.outer(taper, taper)
    height_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros_like(height_sum)

    for y in ys:
        for x in xs:
            tile = raw_image[y:y + tile_size, x:x + tile_size]
            prediction = model.infer_image(tile, tile_size)
            if prediction.shape != (tile_size, tile_size):
                raise ValueError(f'Unexpected tile prediction shape: {prediction.shape}')
            height_sum[y:y + tile_size, x:x + tile_size] += prediction * weights
            weight_sum[y:y + tile_size, x:x + tile_size] += weights

    return (height_sum / weight_sum)[:height, :width]


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 US3D height estimation')
    parser.add_argument('--img-path', required=True)
    parser.add_argument('--input-size', type=int, default=518,
                        help='tile size in pixels (multiple of 14); 518 by default')
    parser.add_argument('--tile-overlap', type=int, default=126,
                        help='overlap between adjacent tiles in pixels')
    parser.add_argument('--resize-whole-image', action='store_true',
                        help='use the previous whole-image resizing method')
    parser.add_argument('--outdir', default='./vis_height')
    parser.add_argument('--encoder', default='vitl', choices=list(MODEL_CONFIGS))
    parser.add_argument('--load-from', required=True)
    parser.add_argument('--max-depth', type=float, default=250.0)
    parser.add_argument('--vis-min', type=float, default=0.0)
    parser.add_argument('--vis-max', type=float)
    parser.add_argument('--save-numpy', action='store_true', help='save raw height predictions in meters')
    parser.add_argument('--pred-only', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()
    if not args.resize_whole_image:
        if args.input_size < 14 or args.input_size % 14:
            parser.error('--input-size must be a positive multiple of 14 for tiled inference')
        if not 0 <= args.tile_overlap < args.input_size:
            parser.error('--tile-overlap must be between 0 and --input-size - 1')

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    checkpoint = torch.load(args.load_from, map_location='cpu')
    state = extract_model_state(checkpoint)
    has_aux_heads = any(key.startswith(('scale_head.', 'angle_head.')) for key in state)
    model_class = DepthAnythingV2withHeads if has_aux_heads else DepthAnythingV2
    model = model_class(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth})
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    filenames = collect_images(args.img_path)
    if not filenames:
        raise FileNotFoundError(f'No supported images found at {args.img_path}')
    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    vis_max = args.max_depth if args.vis_max is None else args.vis_max

    for index, filename in enumerate(filenames, start=1):
        print(f'Progress {index}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if raw_image is None:
            print(f'Skipping unreadable image: {filename}')
            continue

        if args.resize_whole_image:
            height = model.infer_image(raw_image, args.input_size)
        else:
            height = infer_image_tiled(model, raw_image, args.input_size, args.tile_overlap)
        stem = os.path.splitext(os.path.basename(filename))[0]
        if args.save_numpy:
            np.save(os.path.join(args.outdir, f'{stem}_height_meter.npy'), height)

        visualization = colorize_height(height, args.vis_min, vis_max, cmap, args.grayscale)
        output_path = os.path.join(args.outdir, f'{stem}.png')
        if args.pred_only:
            cv2.imwrite(output_path, visualization)
        else:
            split_region = np.full((raw_image.shape[0], 50, 3), 255, dtype=np.uint8)
            cv2.imwrite(output_path, cv2.hconcat([raw_image, split_region, visualization]))


if __name__ == '__main__':
    main()
