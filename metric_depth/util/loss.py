import torch
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
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]

        error = pred - target
        abs_error = torch.abs(error)

        c = 0.2 * torch.max(abs_error).item()

        l1_part = abs_error <= c
        l2_part = abs_error > c

        loss = torch.zeros_like(abs_error)
        loss[l1_part] = abs_error[l1_part]
        loss[l2_part] = (error[l2_part] ** 2 + c ** 2) / (2 * c)

        return torch.mean(loss)


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

    def forward(self, pred, target, valid_mask):
        pred_h = pred["depth"]
        gt_h = target["depth"]

        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            pred_h = pred_h[valid_mask]
            gt_h = gt_h[valid_mask]

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

        return loss