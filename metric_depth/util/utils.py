import os
import re
import numpy as np
import logging
import matplotlib.cm as cm
import torch

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def normalize_depth(depth):
    depth = depth.clone()
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    return depth

def depth_to_colormap(depth):
    depth = normalize_depth(depth)
    depth = depth.numpy()
    colored = cm.viridis(depth)[..., :3]  # RGB
    colored = torch.from_numpy(colored).permute(2, 0, 1)
    return colored