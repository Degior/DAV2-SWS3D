import argparse
import gc
import json
import logging
import math
import os
import pprint
import random
from contextlib import nullcontext
from pathlib import Path
from statistics import harmonic_mean

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.us3d import US3D
from dataset.us3d_with_heads import US3DWH
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads, checkpoint_height_activation
from depth_anything_v2.dinov2_layers.attention import set_attention_backend
from prepare_agl_city_stats import us3d_city, validate_agl_city_stats
from util.dist_helper import setup_distributed
from util.loss import BerHuLoss, HeightLoss, L1AGLLoss, L1MSELoss
from util.inference import predict_height_tiled
from util.metric import METRIC_NAMES, eval_depth
from util.utils import depth_to_colormap, init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for US3D height estimation')
parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--attention-backend', choices=['manual', 'sdpa'],
                    help='Fresh training defaults to the original attention; resume inherits the saved backend')
parser.add_argument('--dataset', default='us3d', choices=['us3d', 'us3dwh'])
parser.add_argument('--train-split', default='dataset/splits/us3d/train.txt')
parser.add_argument('--val-split', default='dataset/splits/us3d/val.txt')
parser.add_argument('--freeze-backbone', action='store_true')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--sampling-mode', choices=['resize', 'native'],
                    help='Fresh US3D training uses native crops; resume inherits the checkpoint mode')
parser.add_argument('--height-crop-prob', type=float,
                    help='Probability of targeting objects from 5 m in a native US3D training crop; defaults to 0')
parser.add_argument('--height-crop-mid-first-prob', type=float,
                    help='Probability of trying 5-20 m first on a targeted crop; defaults to 0.5')
parser.add_argument('--val-tile-overlap', type=int,
                    help='Native validation tile overlap (default: 126; resume inherits the saved value)')
parser.add_argument('--min-depth', default=0.1, type=float,
                    help='Minimum training target height before zero-height inclusion; validation includes zero heights')
parser.add_argument('--zero-height-after-epochs', type=int,
                    help='Include zero heights after N completed epochs (default: 15; resume uses checkpoint setting)')
parser.add_argument('--max-depth', default=250.0, type=float)
parser.add_argument('--height-activation', choices=['sigmoid', 'softplus'],
                    help='Fresh training defaults to sigmoid; resume uses the checkpoint activation')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--micro-batch-size', type=int,
                    help='Images per forward pass; --bs remains the effective per-GPU optimizer batch')
parser.add_argument('--height-loss', choices=['berhu', 'l1-mse', 'l1-agl'],
                    help='US3D height loss; fresh/legacy runs use berhu, resume inherits the saved loss')
parser.add_argument('--mse-weight', type=float,
                    help='Squared-error coefficient for l1-mse (default: 0.05; resume inherits it)')
parser.add_argument('--agl-city-stats',
                    help='Training-city statistics JSON from prepare_agl_city_stats.py; resume inherits the data')
parser.add_argument('--agl-weight', type=float,
                    help='City-normalized AGL coefficient; defaults to 0.1 times the harmonic mean of city variances')
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--lr-scheduler', default='constant', choices=['constant', 'poly'])
parser.add_argument('--hflip-prob', default=0.5, type=float)
parser.add_argument('--angle-unit', default='radians', choices=['radians', 'degrees'])
parser.add_argument('--lambda-scale', default=0.05, type=float)
parser.add_argument('--lambda-angle', default=0.05, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--resume', type=str)
parser.add_argument('--resume-lr', type=float,
                    help='Override the backbone learning rate on resume while preserving optimizer state')
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

ACCUMULATING_HEIGHT_LOSSES = {'l1-mse', 'l1-agl'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_structurally_unused_parameters(model):
    # DINO's mask token is only used for masked pretraining. DPT refinenet4 is
    # called without a skip tensor, so its first residual unit is also unused.
    model.pretrained.mask_token.requires_grad_(False)
    for param in model.depth_head.scratch.refinenet4.resConfUnit1.parameters():
        param.requires_grad = False


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


def load_pretrained_backbone(model, path, logger):
    checkpoint = torch.load(path, map_location='cpu')
    state = extract_model_state(checkpoint)
    backbone_state = {key: value for key, value in state.items() if key.startswith('pretrained.')}
    if not backbone_state:
        raise ValueError(f'No pretrained.* weights found in {path}')
    incompatible = model.load_state_dict(backbone_state, strict=False)
    logger.info(
        'Loaded %d backbone tensors from %s (%d missing, %d unexpected)',
        len(backbone_state), path, len(incompatible.missing_keys), len(incompatible.unexpected_keys)
    )


def build_optimizer(model, args):
    groups = []
    if not args.freeze_backbone:
        groups.append({
            'params': list(model.pretrained.parameters()),
            'lr': args.lr,
            'name': 'backbone',
        })
    groups.append({
        'params': list(model.depth_head.parameters()),
        'lr': args.lr * 10.0,
        'name': 'depth_head',
    })
    if args.dataset == 'us3dwh':
        groups.append({
            'params': list(model.scale_head.parameters()) + list(model.angle_head.parameters()),
            'lr': args.lr * 10.0,
            'name': 'aux_heads',
        })

    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
    optimized = {id(param) for group in optimizer.param_groups for param in group['params']}
    missing = [name for name, param in model.named_parameters() if param.requires_grad and id(param) not in optimized]
    if missing:
        raise RuntimeError(f'Trainable parameters missing from optimizer: {missing[:10]}')
    return optimizer


def reduce_metric_dict(metric_sums, metric_counts):
    for name in metric_sums:
        dist.all_reduce(metric_sums[name])
        dist.all_reduce(metric_counts[name])
    return {
        name: (metric_sums[name] / metric_counts[name]).item()
        for name in metric_sums
        if metric_counts[name].item() > 0
    }


def main():
    args = parser.parse_args()
    if args.pretrained_from and args.resume:
        parser.error('--pretrained-from and --resume are mutually exclusive')
    if args.resume_lr is not None:
        if not args.resume or not math.isfinite(args.resume_lr) or args.resume_lr <= 0:
            parser.error('--resume-lr requires --resume and a finite positive learning rate')
        args.lr = args.resume_lr
    if not 0.0 <= args.hflip_prob <= 1.0:
        parser.error('--hflip-prob must be between 0 and 1')
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error('Expected 0 <= min-depth < max-depth')

    resume_checkpoint = torch.load(args.resume, map_location='cpu') if args.resume else None
    saved_args = resume_checkpoint.get('args', {}) if resume_checkpoint is not None else {}
    args.height_loss = args.height_loss or saved_args.get('height_loss', 'berhu')
    if args.mse_weight is None:
        args.mse_weight = saved_args.get('mse_weight', 0.05)
    if args.micro_batch_size is None:
        args.micro_batch_size = saved_args.get('micro_batch_size', args.bs)
    if not math.isfinite(args.mse_weight) or args.mse_weight < 0:
        parser.error('--mse-weight must be finite and nonnegative')
    if args.bs < 1 or args.micro_batch_size < 1 or args.bs % args.micro_batch_size:
        parser.error('--bs must be positive and divisible by --micro-batch-size')
    if args.dataset != 'us3d' and args.height_loss != 'berhu':
        parser.error('--height-loss l1-mse and l1-agl require --dataset us3d')
    if args.micro_batch_size != args.bs and args.height_loss not in ACCUMULATING_HEIGHT_LOSSES:
        parser.error('Gradient accumulation requires --height-loss l1-mse or l1-agl')
    if args.height_loss == 'l1-agl':
        try:
            if args.agl_city_stats is not None:
                city_stats = json.loads(Path(args.agl_city_stats).read_text(encoding='utf-8'))
            else:
                city_stats = saved_args.get('agl_city_stats')
            validate_agl_city_stats(city_stats, args.train_split, args.max_depth)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        args.agl_city_stats = city_stats
        if args.agl_weight is None:
            args.agl_weight = saved_args.get('agl_weight')
        if args.agl_weight is None:
            args.agl_weight = 0.1 * harmonic_mean(
                stats['variance_m2'] for stats in city_stats['cities'].values()
            )
        if not math.isfinite(args.agl_weight) or args.agl_weight < 0:
            parser.error('--agl-weight must be finite and nonnegative')
    elif args.agl_city_stats is not None or args.agl_weight is not None:
        parser.error('--agl-city-stats and --agl-weight require --height-loss l1-agl')
    args.attention_backend = args.attention_backend or saved_args.get('attention_backend', 'manual')
    if args.sampling_mode is None:
        args.sampling_mode = saved_args.get(
            'sampling_mode', 'resize' if resume_checkpoint is not None or args.dataset == 'us3dwh' else 'native'
        )
    if args.height_crop_prob is None:
        args.height_crop_prob = saved_args.get('height_crop_prob', 0.0)
    if args.height_crop_mid_first_prob is None:
        args.height_crop_mid_first_prob = saved_args.get('height_crop_mid_first_prob', 0.5)
    if not 0.0 <= args.height_crop_prob <= 1.0:
        parser.error('--height-crop-prob must be between 0 and 1')
    if not 0.0 <= args.height_crop_mid_first_prob <= 1.0:
        parser.error('--height-crop-mid-first-prob must be between 0 and 1')
    if args.height_crop_prob and (args.dataset != 'us3d' or args.sampling_mode != 'native'):
        parser.error('--height-crop-prob requires --dataset us3d --sampling-mode native')
    if args.val_tile_overlap is None:
        args.val_tile_overlap = saved_args.get('val_tile_overlap', 126)
    if args.sampling_mode == 'native':
        if args.dataset != 'us3d':
            parser.error('Native crops currently require --dataset us3d')
        if args.img_size < 14 or args.img_size % 14:
            parser.error('--img-size must be a positive multiple of 14 for native crops')
        if not 0 <= args.val_tile_overlap < args.img_size:
            parser.error('--val-tile-overlap must be between 0 and --img-size - 1')
    validation_inference = {'sampling_mode': args.sampling_mode, 'img_size': args.img_size}
    if args.sampling_mode == 'native':
        validation_inference['overlap'] = args.val_tile_overlap
    if resume_checkpoint is not None:
        saved_activation = checkpoint_height_activation(resume_checkpoint)
        if args.height_activation is not None and args.height_activation != saved_activation:
            parser.error('--height-activation must match the resumed checkpoint')
        if 'max_depth' in saved_args and args.max_depth != saved_args['max_depth']:
            parser.error('--max-depth must match the resumed checkpoint')
        args.height_activation = saved_activation
    else:
        args.height_activation = args.height_activation or 'sigmoid'
    if args.zero_height_after_epochs is None:
        args.zero_height_after_epochs = saved_args.get('zero_height_after_epochs', 15)
    if args.zero_height_after_epochs < 0:
        parser.error('--zero-height-after-epochs must be nonnegative')

    logger = init_log('global', logging.INFO)
    logger.propagate = False
    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)

    seed_everything(args.seed + rank)
    cudnn.enabled = True
    cudnn.benchmark = True
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    writer = SummaryWriter(args.save_path) if rank == 0 else None
    if rank == 0:
        logger.info('%s\n', pprint.pformat({**vars(args), 'ngpus': world_size}))

    size = (args.img_size, args.img_size)
    dataset_kwargs = {'size': size}
    if args.dataset == 'us3dwh':
        dataset_class = US3DWH
        dataset_kwargs['angle_unit'] = args.angle_unit
        if args.hflip_prob > 0 and rank == 0:
            logger.warning('Horizontal flip is disabled for us3dwh until the angle convention is specified.')
    else:
        dataset_class = US3D
        dataset_kwargs['sampling_mode'] = args.sampling_mode

    train_kwargs = dataset_kwargs.copy()
    if args.height_crop_prob:
        train_kwargs.update(
            height_crop_prob=args.height_crop_prob, max_depth=args.max_depth,
            height_crop_mid_first_prob=args.height_crop_mid_first_prob,
        )
    trainset = dataset_class(args.train_split, 'train', **train_kwargs)
    valset = dataset_class(args.val_split, 'val', **dataset_kwargs)
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(
        valset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    trainloader = DataLoader(
        trainset, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers,
        drop_last=True, sampler=trainsampler, persistent_workers=args.num_workers > 0,
    )
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=args.num_workers,
        drop_last=False, sampler=valsampler, persistent_workers=args.num_workers > 0,
        prefetch_factor=1 if args.num_workers > 0 else None,
    )

    model_class = DepthAnythingV2withHeads if args.dataset == 'us3dwh' else DepthAnythingV2
    model = model_class(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth,
                           'height_activation': args.height_activation})
    set_attention_backend(model, args.attention_backend)

    if args.pretrained_from:
        load_pretrained_backbone(model, args.pretrained_from, logger)
    elif resume_checkpoint is not None:
        model.load_state_dict(extract_model_state(resume_checkpoint), strict=True)
        if rank == 0:
            logger.info('Resumed model from %s', args.resume)

    freeze_structurally_unused_parameters(model)
    if args.freeze_backbone:
        for param in model.pretrained.parameters():
            param.requires_grad = False

    model.to(device)
    optimizer = build_optimizer(model, args)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    )

    if args.dataset == 'us3dwh':
        criterion = HeightLoss(
            lambda_scale=args.lambda_scale,
            lambda_angle=args.lambda_angle,
        ).to(device)
    elif args.height_loss == 'l1-mse':
        criterion = L1MSELoss(mse_weight=args.mse_weight).to(device)
    elif args.height_loss == 'l1-agl':
        cities = args.agl_city_stats['cities']
        city_indices = {city: i for i, city in enumerate(sorted(cities))}
        criterion = L1AGLLoss(
            city_means=[cities[city]['mean_m'] for city in city_indices],
            agl_weight=args.agl_weight,
        ).to(device)
    else:
        criterion = BerHuLoss().to(device)

    start_epoch = 0
    best_rmse = math.inf
    if resume_checkpoint is not None:
        if 'optimizer' in resume_checkpoint:
            configured_lrs = [group['lr'] for group in optimizer.param_groups]
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            if args.resume_lr is not None:
                for group, lr in zip(optimizer.param_groups, configured_lrs):
                    group['lr'] = lr
        start_epoch = int(resume_checkpoint.get('epoch', -1)) + 1
        saved_validation_min_depth = resume_checkpoint.get(
            'validation_min_depth', saved_args.get('min_depth', 'legacy')
        )
        saved_validation_inference = resume_checkpoint.get('validation_inference', {
            'sampling_mode': saved_args.get('sampling_mode', 'resize'),
            'img_size': saved_args.get('img_size', 518),
        })
        agl_config_changed = args.height_loss == 'l1-agl' and (
            saved_args.get('height_loss') != args.height_loss
            or saved_args.get('agl_weight') != args.agl_weight
            or saved_args.get('agl_city_stats') != args.agl_city_stats
        )
        if agl_config_changed:
            if rank == 0:
                logger.info('Tracking best RMSE from the first epoch with this AGL configuration')
        elif saved_validation_inference != validation_inference:
            if rank == 0:
                logger.info('Resetting best RMSE because validation inference changed: %s -> %s',
                            saved_validation_inference, validation_inference)
        elif saved_validation_min_depth in (None, 0.0):
            best_rmse = float(resume_checkpoint.get('best_rmse', math.inf))
        elif rank == 0:
            logger.info('Resetting best RMSE for validation including zero-height targets')

    del resume_checkpoint, saved_args
    gc.collect()

    if rank == 0:
        logger.info('Attention backend: %s', args.attention_backend)
        if args.sampling_mode == 'native':
            logger.info('Training: native %dx%d random crops; validation: native tiles with %d px overlap',
                        args.img_size, args.img_size, args.val_tile_overlap)
            if args.height_crop_prob:
                logger.info('Height-targeted crop attempts: %.0f%%; 5-20 m first: %.0f%% of targeted attempts; '
                            'fallback uses the same image',
                            100 * args.height_crop_prob, 100 * args.height_crop_mid_first_prob)
        else:
            logger.info('Training and validation use whole-image resizing to input size %d', args.img_size)
        logger.info('Global batch size: %d (%d per GPU x %d GPUs)',
                    args.bs * world_size, args.bs, world_size)
        logger.info('Microbatch size: %d per GPU; %d microbatches per optimizer step',
                    args.micro_batch_size, args.bs // args.micro_batch_size)
        logger.info('Optimizer learning rates: %s',
                    {group['name']: group['lr'] for group in optimizer.param_groups})
        if args.dataset == 'us3d' and args.height_loss == 'l1-mse':
            logger.info('Height loss: mean(abs(error) + %.6g * error^2), normalized over global valid pixels',
                        args.mse_weight)
        elif args.height_loss == 'l1-agl':
            logger.info('Height loss: global L1 + %.6g * mean(city SSE / city target SSE)', args.agl_weight)
        elif args.dataset != 'us3dwh':
            logger.info('BerHu uses a shared global threshold and valid-pixel mean across GPUs.')

    total_iters = args.epochs * len(trainloader)

    def update_learning_rate(cur_iter):
        if args.lr_scheduler == 'constant':
            return
        lr = args.lr * (1 - cur_iter / max(total_iters, 1)) ** 0.9
        for group in optimizer.param_groups:
            group['lr'] = lr if group['name'] == 'backbone' else lr * 10.0

    crop_bands = ('uniform', '5-20m', '20m+', 'fallback')
    for epoch in range(start_epoch, args.epochs):
        trainsampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        crop_counts = torch.zeros(len(crop_bands), dtype=torch.long) if args.height_crop_prob else None
        train_min_depth = 0.0 if epoch >= args.zero_height_after_epochs else args.min_depth
        if rank == 0:
            logger.info('Epoch %d: training min-depth %.3f m; validation min-depth 0.000 m',
                        epoch + 1, train_min_depth)
            writer.add_scalar('train/min_depth', train_min_depth, epoch * len(trainloader))

        for i, sample in enumerate(trainloader):
            if crop_counts is not None:
                crop_counts += torch.bincount(sample['height_crop_band'], minlength=len(crop_bands))
            optimizer.zero_grad(set_to_none=True)
            img = sample['image'].to(device, non_blocking=True)
            depth = sample['depth'].to(device, non_blocking=True)
            valid_mask = sample['valid_mask'].to(device, non_blocking=True)

            if args.dataset == 'us3d' and random.random() < args.hflip_prob:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            train_mask = (
                valid_mask.bool() & torch.isfinite(depth)
                & (depth >= train_min_depth) & (depth <= args.max_depth)
            )

            loss_parts = None
            if args.height_loss in ACCUMULATING_HEIGHT_LOSSES:
                if args.height_loss == 'l1-agl':
                    batch_cities = torch.tensor(
                        [city_indices[us3d_city(path)] for path in sample['image_path']], device=device
                    )
                    normalizers = criterion.batch_normalizers(depth, train_mask, batch_cities)
                else:
                    global_valid_count = train_mask.sum()
                    if world_size > 1:
                        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
                loss = img.new_zeros(())
                for start in range(0, img.shape[0], args.micro_batch_size):
                    end = start + args.micro_batch_size
                    micro = slice(start, end)
                    context = model.no_sync() if world_size > 1 and end < img.shape[0] else nullcontext()
                    with context:
                        outputs = model(img[micro])
                        if args.height_loss == 'l1-agl':
                            micro_loss = criterion(
                                outputs, depth[micro], train_mask[micro], batch_cities[micro], normalizers
                            )
                        else:
                            micro_loss = criterion(outputs, depth[micro], train_mask[micro], global_valid_count)
                        if not torch.isfinite(micro_loss):
                            raise FloatingPointError(f'Non-finite loss at epoch {epoch}, iteration {i}')
                        micro_loss.backward()
                    loss += micro_loss.detach()
            else:
                outputs = model(img)
                if args.dataset == 'us3dwh':
                    targets = {
                        'depth': depth,
                        'scale': sample['scale'].to(device, non_blocking=True),
                        'angle': sample['angle'].to(device, non_blocking=True),
                    }
                    loss, loss_parts = criterion(outputs, targets, train_mask, return_components=True)
                else:
                    loss = criterion(outputs, depth, train_mask)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch}, iteration {i}')
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            iters = epoch * len(trainloader) + i
            update_learning_rate(iters)
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), iters)
                writer.add_scalar('train/valid_fraction', train_mask.float().mean().item(), iters)
                if loss_parts is not None:
                    for name, value in loss_parts.items():
                        writer.add_scalar(f'train/loss_{name}', value.item(), iters)
            if rank == 0 and i % 100 == 0:
                logger.info(
                    'Epoch %d/%d, iter %d/%d, LR %.3e, loss %.4f, valid %.1f%%',
                    epoch + 1, args.epochs, i, len(trainloader),
                    optimizer.param_groups[0]['lr'], loss.item(), 100 * train_mask.float().mean().item()
                )

        if crop_counts is not None:
            crop_counts = crop_counts.to(device)
            dist.all_reduce(crop_counts, op=dist.ReduceOp.SUM)
            if rank == 0:
                counts = crop_counts.tolist()
                total = max(sum(counts), 1)
                summary = ', '.join(f'{band}={count / total:.1%} ({count})'
                                    for band, count in zip(crop_bands, counts))
                logger.info('Height crop choices epoch %d: %s', epoch + 1, summary)

        model.eval()
        metric_names = list(METRIC_NAMES)
        if args.dataset == 'us3dwh':
            metric_names += ['scale_log_mae', 'angle_mae_deg']
        metric_sums = {name: torch.zeros((), device=device) for name in metric_names}
        metric_counts = {name: torch.zeros((), device=device) for name in metric_names}

        for i, sample in enumerate(valloader):
            img = sample['image'].to(device, non_blocking=True).float()
            depth = sample['depth'].to(device, non_blocking=True)[0]
            valid_mask = sample['valid_mask'].to(device, non_blocking=True)[0]
            with torch.no_grad():
                if args.sampling_mode == 'native':
                    pred = predict_height_tiled(
                        model.module, img, args.img_size, args.val_tile_overlap, batch_size=args.micro_batch_size
                    )[0]
                else:
                    outputs = model(img)
                    pred = outputs['depth'] if args.dataset == 'us3dwh' else outputs
                    pred = F.interpolate(
                        pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True
                    )[0, 0]

            eval_mask = (
                valid_mask.bool() & torch.isfinite(depth) & torch.isfinite(pred)
                & (depth >= 0.0) & (depth <= args.max_depth)
            )
            current = eval_depth(pred[eval_mask], depth[eval_mask])
            for name, value in current.items():
                metric_sums[name] += value
                metric_counts[name] += 1

            if args.dataset == 'us3dwh':
                scale_gt = sample['scale'].to(device)
                angle_gt = sample['angle'].to(device)
                metric_sums['scale_log_mae'] += torch.abs(
                    outputs['scale'] - torch.log(scale_gt.clamp_min(1e-6))
                ).mean()
                metric_counts['scale_log_mae'] += 1
                angle_diff = torch.atan2(
                    torch.sin(outputs['angle'] - angle_gt),
                    torch.cos(outputs['angle'] - angle_gt),
                ).abs()
                metric_sums['angle_mae_deg'] += torch.rad2deg(angle_diff).mean()
                metric_counts['angle_mae_deg'] += 1

            if writer is not None and i == 0:
                image_vis = img[0].detach().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                image_vis = (image_vis * std + mean).clamp(0, 1)
                writer.add_image('val/image', image_vis, epoch)
                writer.add_image('val/pred_height_color', depth_to_colormap(pred.detach().cpu()), epoch)
                writer.add_image('val/gt_height_color', depth_to_colormap(depth.detach().cpu()), epoch)

        averages = reduce_metric_dict(metric_sums, metric_counts)
        if rank == 0:
            logger.info('Validation epoch %d: %s', epoch + 1, pprint.pformat(averages))
            logger.info('Mean training loss: %.5f', epoch_loss / max(len(trainloader), 1))
            for name, value in averages.items():
                writer.add_scalar(f'eval/{name}', value, epoch)

            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_rmse': min(best_rmse, averages.get('rmse', math.inf)),
                'validation_min_depth': None,
                'validation_inference': validation_inference,
                'args': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if averages.get('rmse', math.inf) < best_rmse:
                best_rmse = averages['rmse']
                checkpoint['best_rmse'] = best_rmse
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
                logger.info('New best RMSE: %.4f', best_rmse)

        best_tensor = torch.tensor(best_rmse, device=device)
        dist.broadcast(best_tensor, src=0)
        best_rmse = best_tensor.item()

    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
