import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class BerHuLoss(nn.Module):
    def __init__(self, eps=1e-6, sync_distributed=True):
        super().__init__()
        self.eps = eps
        self.sync_distributed = sync_distributed

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            pred = pred[valid_mask]
            target = target[valid_mask]

        distributed = (
            self.sync_distributed and dist.is_available()
            and dist.is_initialized() and dist.get_world_size() > 1
        )
        if pred.numel() == 0 and not distributed:
            return pred.sum() * 0.0

        error = pred - target
        abs_error = torch.abs(error)

        max_error = abs_error.detach().max() if abs_error.numel() else abs_error.new_zeros(())
        if distributed:
            dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
        c = (0.2 * max_error).clamp_min(self.eps)

        l1_part = abs_error <= c
        l2_part = abs_error > c

        loss = torch.zeros_like(abs_error)
        loss[l1_part] = abs_error[l1_part]
        loss[l2_part] = (error[l2_part] ** 2 + c.square()) / (2 * c)

        if not distributed:
            return torch.mean(loss)

        reduction_dtype = torch.float64 if loss.dtype == torch.float64 else torch.float32
        local_sum = loss.sum(dtype=reduction_dtype)
        count = torch.tensor(pred.numel(), device=pred.device, dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        denominator = count.clamp_min(1).to(reduction_dtype)
        global_sum = local_sum.detach().clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
        backward_loss = local_sum * dist.get_world_size() / denominator
        global_loss = global_sum / denominator
        return backward_loss + (global_loss - backward_loss.detach())


def r2_loss(pred, target):
    eps = 1e-8

    target_mean = torch.mean(target)

    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)

    loss = (ss_res + eps) / (ss_tot + eps)
    return loss


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred_h = pred[valid_mask]
        gt_h = target[valid_mask]

        if pred_h.numel() == 0:
            return pred_h.sum() * 0.0

        loss = F.mse_loss(pred_h, gt_h)

        return loss


class HeightLoss(nn.Module):
    def __init__(
            self,
            lambda_scale=0.1,
            lambda_angle=0.1,
            depth_beta=1.0,
            scale_beta=0.1,
            eps=1e-6
    ):
        super().__init__()

        self.lambda_scale = lambda_scale
        self.lambda_angle = lambda_angle
        self.depth_beta = depth_beta
        self.scale_beta = scale_beta
        self.eps = eps

    def forward(self, pred, target, valid_mask, return_components=False):
        pred_h = pred["depth"]
        gt_h = target["depth"]

        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            pred_h = pred_h[valid_mask]
            gt_h = gt_h[valid_mask]

        if pred_h.numel() == 0:
            loss_h = pred["depth"].sum() * 0.0
        else:
            loss_h = F.smooth_l1_loss(
                pred_h,
                gt_h,
                beta=self.depth_beta
            )

        pred_scale = pred["scale"]
        gt_scale = torch.log(target["scale"].clamp_min(self.eps))

        loss_scale = F.smooth_l1_loss(
            pred_scale,
            gt_scale,
            beta=self.scale_beta
        )

        pred_angle_vec = pred["angle_vec"]

        gt_angle = target["angle"]
        gt_angle_vec = torch.stack([
            torch.sin(gt_angle),
            torch.cos(gt_angle)
        ], dim=-1)

        gt_angle_vec = F.normalize(gt_angle_vec, p=2, dim=-1, eps=self.eps)
        pred_angle_vec = F.normalize(pred_angle_vec, p=2, dim=-1, eps=self.eps)

        loss_angle = 1.0 - F.cosine_similarity(
            pred_angle_vec,
            gt_angle_vec,
            dim=-1,
            eps=self.eps
        ).mean()

        loss = (
            loss_h
            + self.lambda_scale * loss_scale
            + self.lambda_angle * loss_angle
        )

        if return_components:
            return loss, {
                "depth": loss_h.detach(),
                "scale": loss_scale.detach(),
                "angle": loss_angle.detach(),
            }

        return loss
