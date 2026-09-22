import torch


ABSOLUTE_METRICS = ("mae", "rmse", "bias")
RELATIVE_METRICS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse_log", "log10", "silog")
METRIC_NAMES = ABSOLUTE_METRICS + RELATIVE_METRICS


def eval_depth(pred, target, eps=1e-6):
    """Return per-image height metrics as tensors.

    Absolute metrics include valid zero-height pixels. Ratio/log metrics are
    evaluated only where target height is strictly positive.
    """
    assert pred.shape == target.shape

    finite = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[finite]
    target = target[finite]

    if pred.numel() == 0:
        return {}

    diff = pred - target
    result = {
        "mae": torch.mean(torch.abs(diff)),
        "rmse": torch.sqrt(torch.mean(diff.square())),
        "bias": torch.mean(diff),
    }

    positive = target > eps
    if not positive.any():
        return result

    pred_pos = pred[positive].clamp_min(eps)
    target_pos = target[positive]
    diff = pred_pos - target_pos
    diff_log = torch.log(pred_pos) - torch.log(target_pos)
    thresh = torch.maximum(target_pos / pred_pos, pred_pos / target_pos)

    result.update({
        "d1": torch.mean((thresh < 1.25).float()),
        "d2": torch.mean((thresh < 1.25 ** 2).float()),
        "d3": torch.mean((thresh < 1.25 ** 3).float()),
        "abs_rel": torch.mean(torch.abs(diff) / target_pos),
        "sq_rel": torch.mean(diff.square() / target_pos),
        "rmse_log": torch.sqrt(torch.mean(diff_log.square())),
        "log10": torch.mean(torch.abs(torch.log10(pred_pos) - torch.log10(target_pos))),
        "silog": torch.sqrt(
            (torch.mean(diff_log.square()) - 0.5 * torch.mean(diff_log).square()).clamp_min(0.0)
        ),
    })

    return result
