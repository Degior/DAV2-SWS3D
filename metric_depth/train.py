import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2withHeads
from dataset.us3d import US3D
from dataset.us3d_with_heads import US3DWH
from util.dist_helper import setup_distributed
from util.loss import HeightLoss, BerHuLoss, MSELoss
from util.metric import eval_depth
from util.utils import init_log, normalize_depth, depth_to_colormap

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='us3d', choices=['hypersim', 'vkitti', 'us3d', 'us3dwh'])
parser.add_argument('--freeze-backbone', action='store_true')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'us3dwh':
        trainset = US3DWH('dataset/splits/us3d/train.txt', 'train', size=size)
    elif args.dataset == 'us3d':
        trainset = US3D('dataset/splits/us3d/train.txt', 'train', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True,
                             sampler=trainsampler)

    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'us3dwh':
        valset = US3DWH('dataset/splits/us3d/val.txt', 'val', size=size)
    elif args.dataset == 'us3d':
        valset = US3D('dataset/splits/us3d/val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)

    local_rank = int(os.environ["LOCAL_RANK"])

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if args.dataset == 'us3dwh':
        model = DepthAnythingV2withHeads(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    elif args.dataset == 'us3d' and args.freeze_backbone:
        model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
        for param in model.pretrained.parameters():
            param.requires_grad = False
    else:
        model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    if args.pretrained_from:
        model.load_state_dict(
            {k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k},
            strict=False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    if args.dataset == 'us3dwh':
        criterion = HeightLoss(lambda_scale=0.5, lambda_angle=0.5).cuda(local_rank)
    else:
        criterion = MSELoss().cuda(local_rank)

    if args.freeze_backbone:
        optimizer = AdamW(
            [
                {
                    'params': model.module.depth_head.parameters(),
                    'lr': args.lr * 10.0,
                    'name': 'head'
                }
            ],
            lr=args.lr * 10.0,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
    else:
        optimizer = AdamW(
            [
                {
                    'params': model.module.pretrained.parameters(),
                    'lr': args.lr,
                    'name': 'backbone'
                },
                {
                    'params': model.module.depth_head.parameters(),
                    'lr': args.lr * 10.0,
                    'name': 'head'
                }
            ],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
                     'log10': 100, 'silog': 100}

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs,
                                                                                                 previous_best['d1'],
                                                                                                 previous_best['d2'],
                                                                                                 previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
                previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0
        if args.dataset == 'us3d':
            for i, sample in enumerate(trainloader):
                optimizer.zero_grad()

                img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

                # ???????????????????????
                if random.random() < 0.5:
                    img = img.flip(-1)
                    depth = depth.flip(-1)
                    valid_mask = valid_mask.flip(-1)

                pred = model(img)

                loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters = epoch * len(trainloader) + i

                lr = args.lr * (1 - iters / total_iters) ** 0.9

                for group in optimizer.param_groups:
                    if group["name"] == "backbone":
                        group["lr"] = lr
                    elif group["name"] == "head":
                        group["lr"] = lr * 10.0

                if rank == 0:
                    writer.add_scalar('train/loss', loss.item(), iters)

                if rank == 0 and i % 100 == 0:
                    logger.info(
                        'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader),
                                                                       optimizer.param_groups[0]['lr'], loss.item()))
        else:
            for i, sample in enumerate(trainloader):
                optimizer.zero_grad()

                img = sample['image'].cuda()
                depth = sample['depth'].cuda()
                valid_mask = sample['valid_mask'].cuda()
                scale_gt = sample['scale'].cuda()
                angle_gt = sample['angle'].cuda()

                # ???????????????????????
                if random.random() < 0.5:
                    img = img.flip(-1)
                    depth = depth.flip(-1)
                    valid_mask = valid_mask.flip(-1)

                outputs = model(img)

                batch_targets = {
                    "depth": depth,
                    "scale": scale_gt,
                    "angle": angle_gt,
                }

                loss = criterion(outputs, batch_targets, valid_mask)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters = epoch * len(trainloader) + i

                lr = args.lr * (1 - iters / total_iters) ** 0.9

                for group in optimizer.param_groups:
                    if group["name"] == "backbone":
                        group["lr"] = lr
                    elif group["name"] == "head":
                        group["lr"] = lr * 10.0

                if rank == 0:
                    writer.add_scalar('train/loss', loss.item(), iters)

                if rank == 0 and i % 100 == 0:
                    logger.info(
                        'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader),
                                                                       optimizer.param_groups[0]['lr'], loss.item()))

        model.eval()

        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(),
                   'rmse': torch.tensor([0.0]).cuda(),
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(),
                   'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()

        for i, sample in enumerate(valloader):

            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], \
                sample['valid_mask'].cuda()[0]

            with torch.no_grad():
                pred = model(img)
                if args.dataset == 'us3dwh':
                    pred = pred["depth"]
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

            if rank == 0 and i == 0:
                img_vis = img[0].detach().cpu()

                pred_vis = depth_to_colormap(pred.detach().cpu())
                depth_vis = depth_to_colormap(depth.detach().cpu())

                if pred_vis.dim() == 4:
                    pred_vis = pred_vis[0]
                if depth_vis.dim() == 4:
                    depth_vis = depth_vis[0]

                writer.add_image("val/image", img_vis, epoch)
                writer.add_image("val/pred_depth_color", pred_vis, epoch)
                writer.add_image("val/gt_depth_color", depth_vis, epoch)

            # if valid_mask.sum() < 10:
            #     continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
                *tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()

            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

        if rank == 0:
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()
