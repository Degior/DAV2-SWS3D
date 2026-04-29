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


def r2_loss(output, target, target_mean):
    eps = 1e-8
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = (ss_res + eps) / (ss_tot + eps)
    return r2


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred_h = outputs[valid_mask]
        gt_h = target[valid_mask]

        loss = F.mse_loss(pred_h, gt_h)

        return loss


class HeightLoss(nn.Module):
    def __init__(self, lambda_scale=0.1, lambda_angle=0.1):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.lambda_angle = lambda_angle

    def forward(self, outputs, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred_h = outputs["depth"][valid_mask]
        gt_h = target["depth"][valid_mask]

        pred_scale = outputs["scale"]
        gt_scale = torch.log(target["scale"] + 1e-6)

        pred_angle = outputs["angle"]
        gt_angle = target["angle"]

        loss_h = r2_loss(pred_h, gt_h)
        loss_scale = F.mse_loss(pred_scale, gt_scale)
        loss_angle = F.mse_loss(pred_angle, gt_angle)

        loss = (
                loss_h
                + self.lambda_scale * loss_scale
                + self.lambda_angle * loss_angle
        )

        return loss
