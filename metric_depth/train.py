import argparse
import logging
import math
import os
import pprint
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.us3d import US3D
from dataset.us3d_with_heads import US3DWH
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads
from util.dist_helper import setup_distributed
from util.loss import BerHuLoss, HeightLoss
from util.metric import METRIC_NAMES, eval_depth
from util.utils import depth_to_colormap, init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for overhead height estimation')
parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='us3d', choices=['us3d', 'us3dwh', 'm4heights'])
parser.add_argument('--train-split', default='dataset/splits/us3d/train.txt')
parser.add_argument('--val-split', default='dataset/splits/us3d/val.txt')
parser.add_argument('--m4heights-train-split', help='Add M4Heights manifest to US3D training')
parser.add_argument('--m4heights-val-split', help='Also add M4Heights manifest to US3D validation')
parser.add_argument('--m4heights-height-scale', default=1.0, type=float, help='Multiplier to metres; M4Heights default is 1')
parser.add_argument('--freeze-backbone', action='store_true')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.0, type=float)
parser.add_argument('--max-depth', default=250.0, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--lr-scheduler', default='constant', choices=['constant', 'poly'])
parser.add_argument('--hflip-prob', default=0.5, type=float)
parser.add_argument('--angle-unit', default='radians', choices=['radians', 'degrees'])
parser.add_argument('--lambda-scale', default=0.05, type=float)
parser.add_argument('--lambda-angle', default=0.05, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--resume', type=str)
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


def build_datasets(args, size):
    dataset_kwargs = {'size': size}
    if args.dataset == 'us3dwh':
        dataset_class = US3DWH
        dataset_kwargs['angle_unit'] = args.angle_unit
    elif args.dataset == 'm4heights':
        from dataset.m4heights import M4Heights
        dataset_class = M4Heights
        dataset_kwargs['height_scale'] = args.m4heights_height_scale
    else:
        dataset_class = US3D
    trainset = dataset_class(args.train_split, 'train', **dataset_kwargs)
    valset = dataset_class(args.val_split, 'val', **dataset_kwargs)
    if args.m4heights_train_split or args.m4heights_val_split:
        from dataset.m4heights import M4Heights
        m4_kwargs = {'size': size, 'height_scale': args.m4heights_height_scale}
        if args.m4heights_train_split:
            trainset = ConcatDataset([trainset, M4Heights(args.m4heights_train_split, 'train', **m4_kwargs)])
        if args.m4heights_val_split:
            valset = ConcatDataset([valset, M4Heights(args.m4heights_val_split, 'val', **m4_kwargs)])
    return trainset, valset


def main():
    args = parser.parse_args()
    if args.pretrained_from and args.resume:
        parser.error('--pretrained-from and --resume are mutually exclusive')
    if not 0.0 <= args.hflip_prob <= 1.0:
        parser.error('--hflip-prob must be between 0 and 1')
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error('Expected 0 <= min-depth < max-depth')
    if not math.isfinite(args.m4heights_height_scale) or args.m4heights_height_scale <= 0:
        parser.error('--m4heights-height-scale must be finite and positive')
    if (args.m4heights_train_split or args.m4heights_val_split) and args.dataset != 'us3d':
        parser.error('Additional M4Heights manifests require --dataset us3d (height-only training)')

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
    if args.dataset == 'us3dwh' and args.hflip_prob > 0 and rank == 0:
        logger.warning('Horizontal flip is disabled for us3dwh until the angle convention is specified.')
    trainset, valset = build_datasets(args, size)
    if rank == 0:
        logger.info('Dataset sizes: %d train / %d val', len(trainset), len(valset))
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
    )

    model_class = DepthAnythingV2withHeads if args.dataset == 'us3dwh' else DepthAnythingV2
    model = model_class(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth})

    resume_checkpoint = None
    if args.pretrained_from:
        load_pretrained_backbone(model, args.pretrained_from, logger)
    elif args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
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
    else:
        criterion = BerHuLoss().to(device)

    start_epoch = 0
    best_rmse = math.inf
    if resume_checkpoint is not None:
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        start_epoch = int(resume_checkpoint.get('epoch', -1)) + 1
        best_rmse = float(resume_checkpoint.get('best_rmse', math.inf))

    total_iters = args.epochs * len(trainloader)

    def update_learning_rate(cur_iter):
        if args.lr_scheduler == 'constant':
            return
        lr = args.lr * (1 - cur_iter / max(total_iters, 1)) ** 0.9
        for group in optimizer.param_groups:
            group['lr'] = lr if group['name'] == 'backbone' else lr * 10.0

    for epoch in range(start_epoch, args.epochs):
        trainsampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)
            img = sample['image'].to(device, non_blocking=True)
            depth = sample['depth'].to(device, non_blocking=True)
            valid_mask = sample['valid_mask'].to(device, non_blocking=True)

            if args.dataset != 'us3dwh' and random.random() < args.hflip_prob:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            outputs = model(img)
            train_mask = (
                valid_mask.bool() & torch.isfinite(depth)
                & (depth >= args.min_depth) & (depth <= args.max_depth)
            )

            loss_parts = None
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
                outputs = model(img)
                pred = outputs['depth'] if args.dataset == 'us3dwh' else outputs
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True
                )[0, 0]

            eval_mask = (
                valid_mask.bool() & torch.isfinite(depth) & torch.isfinite(pred)
                & (depth >= args.min_depth) & (depth <= args.max_depth)
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
