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


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 US3D height estimation')
    parser.add_argument('--img-path', required=True)
    parser.add_argument('--input-size', type=int, default=518)
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

        height = model.infer_image(raw_image, args.input_size)
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
